import sys
import collections
import uuid
import os
import math
from math import floor, sqrt
import json
from pathlib import Path

import logging

logging.getLogger("pyboy.plugins.window_headless").setLevel(logging.WARNING)

import numpy as np
from einops import rearrange
import matplotlib.pyplot as plt
from skimage.transform import resize
from pyboy import PyBoy
import hnswlib
import mediapy as media
import pandas as pd

import gymnasium as gym
from gymnasium import spaces

from pyboy.utils import WindowEvent
from memory_addresses import *


class ZeldaGymEnv(gym.Env):

    def __init__(self, config=None):

        self.headless = config["headless"]
        self.save_final_state = config["save_final_state"]
        self.early_stop = config["early_stop"]
        self.action_freq = config["action_freq"]
        self.init_state = config["init_state"]
        self.max_steps = config["max_steps"]
        self.print_rewards = config["print_rewards"]
        self.save_video = config["save_video"]
        self.fast_video = config["fast_video"]
        self.session_path = config["session_path"]
        self.gb_path = config["gb_path"]
        self.debug = config["debug"]
        self.sim_frame_dist = config["sim_frame_dist"]
        self.use_screen_explore = config["use_screen_explore"]
        self.extra_buttons = config["extra_buttons"]
        self.vec_dim = 42 * 42 * 3  # 4320  # 1000 # must match output shape!!!
        self.num_elements = 20000
        self.reward_scale = (
            1 if "reward_scale" not in config else config["reward_scale"]
        )
        self.video_interval = 256 * self.action_freq
        self.downsample_factor = 2
        self.frame_stacks = 3
        self.explore_weight = 1
        # (1 if "explore_weight" not in config else config["explore_weight"])
        self.instance_id = str(uuid.uuid4())[:8]
        self.reset_count = 0
        self.all_runs = []

        # Set this in SOME subclasses
        self.metadata = {"render.modes": []}
        self.reward_range = (0, 15000)

        self.valid_actions = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
            WindowEvent.PRESS_BUTTON_START,
            WindowEvent.PASS,
        ]

        self.release_arrow = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP,
        ]

        self.release_button = [
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
        ]

        self.output_shape = (42, 42, 3)  # (36, 40, 3) # must match vec_dim!!!
        self.mem_padding = 2
        self.memory_height = 8
        self.col_steps = 16
        self.output_full = (
            self.output_shape[0] * self.frame_stacks
            + 2 * (self.mem_padding + self.memory_height),
            self.output_shape[1],
            self.output_shape[2],
        )

        # Define the current objective.
        self.objective = 0
        # Define the highest reached objective for the current episode.
        self.highest_objective = 0

        self.objective_done = False

        self.objective_coordinates = [X_DESTINATION, Y_DESTINATION]

        # objectives completed
        self.objective_completed = 0

        self.killed_enemy_count_last = 0
        self.killed_enemy_count = 0
        self.map_location = 0
        self.target_distance_last = 0
        self.distance_diffrence_last = 0

        # Set these in ALL subclasses
        self.action_space = spaces.Discrete(len(self.valid_actions))
        self.observation_space = spaces.Box(
            low=0, high=255, shape=self.output_shape, dtype=np.uint8
        )
        # self.observation_space = spaces.Box(low=0, high=255, shape=self.output_full, dtype=np.uint8)

        head = "headless" if self.headless else "SDL2"

        self.pyboy = PyBoy(
            self.gb_path,
            debugging=False,
            disable_input=False,
            window_type=head,
            hide_window=True,
            simultaneous_actions=True,  #'--quiet' in sys.argv,
        )

        self.screen = self.pyboy.botsupport_manager().screen()

        self.pyboy.set_emulation_speed(0 if self.headless else 6)
        self.reset()

    def reset(self, *, seed=None, options=None):
        self.seed = seed
        # restart game, skipping credits
        with open(self.init_state, "rb") as f:
            self.pyboy.load_state(f)

        if self.use_screen_explore:
            self.init_knn()

        else:
            self.init_map_mem()

        self.recent_memory = np.zeros(
            (self.output_shape[1] * self.memory_height, 3), dtype=np.uint8
        )

        self.recent_frames = np.zeros(
            (
                self.frame_stacks,
                self.output_shape[0],
                self.output_shape[1],
                self.output_shape[2],
            ),
            dtype=np.uint8,
        )

        self.agent_stats = []

        if self.save_video:
            base_dir = self.session_path / Path("rollouts")
            base_dir.mkdir(exist_ok=True)
            full_name = Path(
                f"full_reset_{self.reset_count}_id{self.instance_id}"
            ).with_suffix(".mp4")
            model_name = Path(
                f"model_reset_{self.reset_count}_id{self.instance_id}"
            ).with_suffix(".mp4")
            self.full_frame_writer = media.VideoWriter(
                base_dir / full_name, (144, 160), fps=60
            )
            self.full_frame_writer.__enter__()
            self.model_frame_writer = media.VideoWriter(
                base_dir / model_name, self.output_full[:2], fps=60
            )
            self.model_frame_writer.__enter__()

        self.levels_satisfied = False
        self.base_explore = 0
        self.max_event_rew = 0
        self.max_level_rew = 0
        self.last_health = 1
        self.total_healing_rew = 0
        self.died_count = self.read_m(NUM_DEATHS)
        self.step_count = 0
        self.progress_reward = self.get_game_state_reward()
        self.total_reward = sum([val for _, val in self.progress_reward.items()])
        self.reset_count += 1
        return self.render(add_memory=False), {}

    def init_knn(self):
        # Declaring index
        self.knn_index = hnswlib.Index(
            space="l2", dim=self.vec_dim
        )  # possible options are l2, cosine or ip
        # Initing index - the maximum number of elements should be known beforehand
        self.knn_index.init_index(
            max_elements=self.num_elements, ef_construction=100, M=16
        )

    def init_map_mem(self):
        self.seen_coords = {}

    def render(self, reduce_res=True, add_memory=True, update_mem=True):
        game_pixels_render = self.screen.screen_ndarray()  # (144, 160, 3)
        if reduce_res:
            game_pixels_render = (
                255 * resize(game_pixels_render, self.output_shape)
            ).astype(np.uint8)
            if update_mem:
                self.recent_frames[0] = game_pixels_render
            if add_memory:
                pad = np.zeros(
                    shape=(self.mem_padding, self.output_shape[1], 3), dtype=np.uint8
                )
                game_pixels_render = np.concatenate(
                    (
                        self.create_exploration_memory(),
                        pad,
                        self.create_recent_memory(),
                        pad,
                        rearrange(self.recent_frames, "f h w c -> (f h) w c"),
                    ),
                    axis=0,
                )
        return game_pixels_render

    def step(self, action):

        self.run_action_on_emulator(action)
        self.append_agent_stats()

        self.recent_frames = np.roll(self.recent_frames, 1, axis=0)
        obs_memory = self.render(add_memory=False)

        # trim off memory from frame for knn index
        # frame_start = 2 * (self.memory_height + self.mem_padding)
        # obs_flat = obs_memory[
        #    frame_start:frame_start+self.output_shape[0], ...].flatten().astype(np.float32)
        obs_flat = obs_memory.flatten().astype(np.float32)

        if self.use_screen_explore:
            self.update_frame_knn_index(obs_flat)
        else:
            self.update_seen_coords

        self.update_heal_reward()

        new_reward, new_prog = self.update_reward()

        self.last_health = self.read_hp_fraction()

        # shift over short term reward memory
        self.recent_memory = np.roll(self.recent_memory, 3)
        self.recent_memory[0, 0] = min(new_prog[0] * 64, 255)
        self.recent_memory[0, 1] = min(new_prog[1] * 64, 255)
        self.recent_memory[0, 2] = min(new_prog[2] * 128, 255)

        done = self.check_if_done()

        self.save_and_print_info(done, obs_memory)

        self.step_count += 1

        return obs_memory, new_reward * 0.1, done, done, {}

    def run_action_on_emulator(self, action):

        # press button then release after some steps
        self.pyboy.send_input(self.valid_actions[action])
        for i in range(self.action_freq):
            # release action, so they are stateless
            if i == 8:
                if action < 4:
                    # release arrow
                    self.pyboy.send_input(self.release_arrow[action])
                if action > 3 and action < 6:
                    # release button
                    self.pyboy.send_input(self.release_button[action - 4])
                if action == WindowEvent.PRESS_BUTTON_START:
                    self.pyboy.send_input(WindowEvent.RELEASE_BUTTON_START)

            if self.save_video and not self.fast_video:
                self.add_video_frame()
            self.pyboy.tick()
        if self.save_video and self.fast_video:
            self.add_video_frame()

    def add_video_frame(self):
        self.full_frame_writer.add_image(
            self.render(reduce_res=False, update_mem=False)
        )
        self.model_frame_writer.add_image(
            self.render(reduce_res=True, update_mem=False)
        )

    def append_agent_stats(self):
        x_pos = self.read_m(X_POS_ADDRESS)
        y_pos = self.read_m(Y_POS_ADDRESS)
        map_n = self.read_m(MAP_STATUS)
        health = self.read_m(HEALTH_LEVEL)
        self.agent_stats.append(
            {
                "step": self.step_count,
                "x": x_pos,
                "y": y_pos,
                "map": map_n,
                "health": health,
                "hp": self.read_hp_fraction(),
                "frames": self.knn_index.get_current_count(),
                "deaths": self.died_count,
                "event": self.progress_reward["event"],
                "healr": self.total_healing_rew,
            }
        )

    def update_frame_knn_index(self, frame_vec):

        if self.get_levels_sum() >= 750 and not self.levels_satisfied:
            self.levels_satisfied = True
            self.base_explore = self.knn_index.get_current_count()
            self.init_knn()

        if self.knn_index.get_current_count() == 0:
            # if index is empty add current frame
            self.knn_index.add_items(
                frame_vec, np.array([self.knn_index.get_current_count()])
            )
        else:
            # check for nearest frame and add if current
            labels, distances = self.knn_index.knn_query(frame_vec, k=1)
            if distances[0] > self.sim_frame_dist:
                self.knn_index.add_items(
                    frame_vec, np.array([self.knn_index.get_current_count()])
                )

    def update_seen_coords(self):
        x_pos = self.read_m(X_POS_ADDRESS)
        y_pos = self.read_m(Y_POS_ADDRESS)
        map_n = self.read_m(MAP_STATUS)
        coord_string = f"x:{x_pos} y:{y_pos} m:{map_n}"
        if self.get_levels_sum() >= 750 and not self.levels_satisfied:
            self.levels_satisfied = True
            self.base_explore = len(self.seen_coords)
            self.seen_coords = {}

        self.seen_coords[coord_string] = self.step_count

    def get_target_distance(self):
        """Returns the absolute distance between Link and the current target, or a specified target."""
        _x_pos = self.read_m(X_POS_ADDRESS)
        _y_pos = self.read_m(Y_POS_ADDRESS)
        _obj_x = self.read_m(X_DESTINATION)
        _obj_y = self.read_m(Y_DESTINATION)
        _x_difference = (_x_pos - _obj_x) ** 2
        _y_difference = (_y_pos - _obj_y) ** 2
        _target_distance = sqrt(_x_difference + _y_difference)
        return _target_distance

    def get_objective_reward(self):
        """Return the reward based on the progress towards the current objective."""
        _target_distance = self.get_target_distance()
        _difference = abs(self.target_distance_last - _target_distance)
        self.target_distance_last = _target_distance
        progress = self.step_count
        _steps = 150
        _reward = 0
        # reduce reward if bot has not made progress
        if progress > _steps:
            _steps += 50
            if self.distance_diffrence_last > _difference:
                self.distance_diffrence_last = _difference
                _reward = -1
            else:
                _reward = 3 * math.atan(_difference)

        return _reward

    def objective_cleared(self):
        """Return reward for a cleared objective."""
        if self.get_target_distance() < 1:
            self.objective += 1
            self.highest_objective += 1
            print("Objective", self.objective, "completed!")

        return 10

    def update_reward(self):
        # compute reward
        old_prog = self.group_rewards()
        self.progress_reward = self.get_game_state_reward()
        new_prog = self.group_rewards()
        new_total = sum(
            [val for _, val in self.progress_reward.items()]
        )  # sqrt(self.explore_reward * self.progress_reward)
        new_step = new_total - self.total_reward
        if new_step < 0 and self.read_hp_fraction() > 0:
            # print(f'\n\nreward went down! {self.progress_reward}\n\n')
            self.save_screenshot("neg_reward")

        self.total_reward = new_total
        return (
            new_step,
            (
                new_prog[0] - old_prog[0],
                new_prog[1] - old_prog[1],
                new_prog[2] - old_prog[2],
                new_prog[3] - old_prog[3],
                new_prog[4] - old_prog[4],
                new_prog[5] - old_prog[5],
            ),
        )

    def group_rewards(self):
        prog = self.progress_reward
        # these values are only used by memory
        return (
            prog["level"],
            prog["explore"],
            prog["objectives"],
            prog["enemies"],
            prog["equipped"],
            prog["died"],
        )

    def create_exploration_memory(self):
        w = self.output_shape[1]
        h = self.memory_height

        def make_reward_channel(r_val):
            col_steps = self.col_steps
            row = floor(r_val / (h * col_steps))
            memory = np.zeros(shape=(h, w), dtype=np.uint8)
            memory[:, :row] = 255
            row_covered = row * h * col_steps
            col = floor((r_val - row_covered) / col_steps)
            memory[:col, row] = 255
            col_covered = col * col_steps
            last_pixel = floor(r_val - row_covered - col_covered)
            memory[col, row] = last_pixel * (255 // col_steps)
            return memory

        level, explore, objectives, enemies, equipped, died = self.group_rewards()
        full_memory = np.stack(
            (
                make_reward_channel(level),
                make_reward_channel(explore),
                make_reward_channel(objectives),
                make_reward_channel(enemies),
                make_reward_channel(equipped),
                make_reward_channel(died),
            ),
            axis=-1,
        )

        return full_memory

    def create_recent_memory(self):
        return rearrange(self.recent_memory, "(w h) c -> h w c", h=self.memory_height)

    def check_if_done(self):
        if self.early_stop:
            done = False
            if self.step_count > 128 and self.recent_memory.sum() < (255 * 1):
                done = True
        else:
            done = self.step_count >= self.max_steps
        # done = self.read_hp_fraction() == 0
        return done

    def save_and_print_info(self, done, obs_memory):
        if self.print_rewards:
            prog_string = f"| step | {self.step_count:6d} |\n"
            for key, val in self.progress_reward.items():
                prog_string += f"| {key} | {val:5.2f} |\n"
            prog_string += f"| sum | {self.total_reward:5.2f} |\n"
            print(f"\r{prog_string}", end="", flush=True)

        if self.print_rewards and done:
            print("", flush=True)
            if self.save_final_state:
                fs_path = self.session_path / Path("final_states")
                fs_path.mkdir(exist_ok=True)
                plt.imsave(
                    fs_path
                    / Path(
                        f"frame_r{self.total_reward:.4f}_{self.reset_count}_small.jpeg"
                    ),
                    obs_memory,
                )
                plt.imsave(
                    fs_path
                    / Path(
                        f"frame_r{self.total_reward:.4f}_{self.reset_count}_full.jpeg"
                    ),
                    self.render(reduce_res=False),
                )

        if self.save_video and done:
            self.full_frame_writer.close()
            self.model_frame_writer.close()

        if done:
            self.all_runs.append(self.progress_reward)
            with open(
                self.session_path / Path(f"all_runs_{self.instance_id}.json"), "w"
            ) as f:
                json.dump(self.all_runs, f)
            pd.DataFrame(self.agent_stats).to_csv(
                self.session_path / Path(f"agent_stats_{self.instance_id}.csv.gz"),
                compression="gzip",
                mode="a",
            )

    def read_m(self, addr):
        return self.pyboy.get_memory_value(addr)

    def get_levels_sum(self):
        shells = self.read_m(SECRET_SHELLS)
        songs = self.read_m(OCARINA_SONGS) * 100
        leaves = self.read_m(GOLDEN_LEAVES) * 10
        max_health = self.read_m(MAX_HEALTH) - 3
        level = shells + songs + leaves + max_health
        return level  # add game progession items together

    def get_levels_reward(self):
        explore_thresh = 750
        scale_factor = 4
        level_sum = self.get_levels_sum()
        if level_sum < explore_thresh:
            scaled = level_sum
        else:
            scaled = (level_sum - explore_thresh) / scale_factor + explore_thresh
        self.max_level_rew = max(self.max_level_rew, scaled)
        return self.max_level_rew

    def get_knn_reward(self):
        pre_rew = 0.004
        post_rew = 0.01
        cur_size = self.knn_index.get_current_count()
        base = (self.base_explore if self.levels_satisfied else cur_size) * pre_rew
        post = (cur_size if self.levels_satisfied else 0) * post_rew
        return base + post

    def kill_reward(self):
        """Return the reward for slaying monsters."""
        self.killed_enemy_count = self.read_m(ENEMIES_KILLED)
        if self.killed_enemy_count < self.killed_enemy_count_last:
            return 0

        _reward = 10 * (self.killed_enemy_count - self.killed_enemy_count_last)
        self.killed_enemy_count_last = self.killed_enemy_count
        return _reward

    def get_instruments(self):
        return self.bit_count(self.read_m(INSTUMENTS))

    def update_heal_reward(self):
        cur_health = self.read_hp_fraction()
        if cur_health > self.last_health:
            if self.last_health > 0:
                heal_amount = cur_health - self.last_health
                if heal_amount > 0.25:
                    print(f"healed: {heal_amount}")
                    self.save_screenshot("healing")
                self.total_healing_rew += heal_amount * 4
            # else:
            #     self.died_count += 1

    def get_all_events_reward(self):
        return max(
            sum(
                [
                    self.bit_count(self.read_m(i))
                    for i in range(EVENT_FLAGS_START_ADDRESS, EVENT_FLAGS_END_ADDRESS)
                ]
            )
            - 13,
            0,
        )

    def get_game_state_reward(self):
        # addresses from https://datacrystal.tcrf.net/w/index.php?title=The_Legend_of_Zelda:_Link%27s_Awakening_(Game_Boy)/RAM_map&oldid=58985

        health = self.read_hp_fraction()
        explore = self.get_knn_reward()
        objectives = self.get_objective_reward()
        event = self.update_max_event_rew()
        enemies = self.kill_reward()
        distance = self.get_target_distance()
        equipped = self.get_equipped_items()

        if self.print_rewards:
            print(
                f"""| Health: {health} |\n| Player_corrdinates: {self.read_m(X_POS_ADDRESS), self.read_m(Y_POS_ADDRESS)}|\n| Target_coordinates: {self.read_m(X_DESTINATION), self.read_m(Y_DESTINATION)} |\n| Distance: {distance} |\n"""
            )

        state_scores = {
            "event": event,
            "level": self.get_levels_reward(),
            "died": self.died_count * -0.8,
            "instruments": self.get_instruments() * 100,
            "explore": explore,
            "enemies": enemies,
            "objectives": objectives,
            "equipped": equipped,
        }

        return state_scores

    def save_screenshot(self, name):
        ss_dir = self.session_path / Path("screenshots/")
        ss_dir.mkdir(exist_ok=True)
        plt.imsave(
            ss_dir
            / Path(
                f"frame{self.instance_id}_r{self.total_reward:.4f}_{self.reset_count}_{name}.jpeg"
            ),
            self.render(reduce_res=False),
        )

    def update_max_event_rew(self):
        cur_rew = self.get_all_events_reward()
        self.max_event_rew = max(cur_rew, self.max_event_rew)
        return self.max_event_rew

    def read_hp_fraction(self):
        hp_sum = self.read_m(HEALTH_LEVEL)
        return hp_sum

    def read_hp(self):
        hp = self.read_m(HEALTH_LEVEL)
        return hp

    def bit_count(self, bits):
        return bin(bits).count("1")

    def read_money(self):
        return self.read_m(RUPREE_ADDRESS_1)

    def get_inventory(self):
        bombs = self.read_m(NUM_BOMBS)
        arrows = self.read_m(NUM_ARROWS)
        items = bombs + arrows
        return items

    def get_equipped_items(self):
        A_ITEM = self.read_m(ITEM_A)
        B_ITEM = self.read_m(ITEM_B)

        if A_ITEM and B_ITEM == 0:
            return -5
        elif A_ITEM == 0:
            return -2
        elif B_ITEM == 0:
            return -2
        else:
            return 0

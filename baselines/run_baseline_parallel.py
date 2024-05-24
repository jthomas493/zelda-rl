from os.path import exists
from pathlib import Path
import uuid
import os
from newest_learning import newest_learning
from zelda_gym_env import ZeldaGymEnv
from stable_baselines3 import A2C, PPO
from stable_baselines3.common import env_checker
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback


current_dir = os.getcwd()  # Get the current working directory
newest_zip_path = newest_learning(current_dir)

if newest_zip_path:
    print(f"The most recent ZIP folder (including subdirectories): {newest_zip_path}")
    # You can now use the zipfile module to access the ZIP content
else:
    print("No valid ZIP folders found in the current directory or its subdirectories.")


def make_env(rank, env_conf, seed=0):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    """

    def _init():
        env = ZeldaGymEnv(env_conf)
        env.reset(seed=(seed + rank))
        return env

    set_random_seed(seed)
    return _init


if __name__ == "__main__":

    ep_length = 5000  # 2048 * 8
    sess_path = Path(f"session_{str(uuid.uuid4())[:8]}")

    env_config = {
        "headless": True,  # False,
        "save_final_state": True,
        "early_stop": False,
        "action_freq": 20,
        "init_state": "./hasSword.state",
        "max_steps": ep_length,
        "print_rewards": True,
        "save_video": False,
        "fast_video": True,
        "session_path": sess_path,
        "gb_path": "./Zelda.gb",  # ./ZeldaDX.gbc,
        "debug": False,
        "sim_frame_dist": 2_000_000.0,
        "use_screen_explore": True,
        "extra_buttons": False,
    }

    num_cpu = 1  # 64 #46  # Also sets the number of episodes per training iteration
    # env = SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])
    env = ZeldaGymEnv(env_config)

    checkpoint_callback = CheckpointCallback(
        save_freq=ep_length, save_path=sess_path, name_prefix="zelda"
    )
    # env_checker.check_env(env)
    self_made_epochs = 10
    file_name = newest_zip_path

    if exists(file_name):
        print("\nloading checkpoint " + file_name)
        model = PPO.load(file_name, env=env)
        model.n_steps = ep_length
        model.n_envs = num_cpu
        model.gamma = 0.001
        model.rollout_buffer.buffer_size = ep_length
        model.rollout_buffer.n_envs = num_cpu
        model.rollout_buffer.reset()
    else:
        model = PPO(
            "CnnPolicy",
            env,
            verbose=1,
            n_steps=ep_length,
            batch_size=512,
            n_epochs=1,
            gamma=0.0009,
            device="cuda",
        )

    for i in range(self_made_epochs):
        model.learn(total_timesteps=(ep_length), callback=checkpoint_callback)

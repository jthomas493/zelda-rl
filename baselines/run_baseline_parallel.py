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
from stable_baselines3.common.callbacks import BaseCallback


# current_dir = os.getcwd()  # Get the current working directory
# newest_zip_path = newest_learning(current_dir)

# if newest_zip_path:
#     print(f"The most recent ZIP folder (including subdirectories): {newest_zip_path}")
# else:
#     print("No valid ZIP folders found in the current directory or its subdirectories.")

LOG_DIR = "./logs/"


class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, "model_{}".format(self.n_calls))
            self.model.save(model_path)

        return True


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

    n_steps = 2048  # 2048 * 8
    sess_path = Path(f"session_{str(uuid.uuid4())[:8]}")

    env_config = {
        "headless": True,  # False Presents a window
        "save_final_state": True,
        "early_stop": False,
        "action_freq": 20,
        "init_state": "./hasSword.state",
        "max_steps": 500_000,
        "print_rewards": False,  # True prints rewards at each time step
        "save_video": False,
        "fast_video": True,
        "session_path": sess_path,
        "gb_path": "./Zelda.gb",  # "./ZeldaDX.gbc",
        "debug": False,
        "sim_frame_dist": 2_000_000.0,
        "use_screen_explore": True,
        "extra_buttons": False,
        "reward_scale": 10,
    }

    num_cpu = 4  # 64 #46  # Also sets the number of episodes per training iteration
    env = SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])
    # env = ZeldaGymEnv(env_config)

    # env = DummyVecEnv([lambda: ZeldaGymEnv(env_config)])

    checkpoint_callback = TrainAndLoggingCallback(
        check_freq=50000,
        save_path=sess_path,
    )

    # env_checker.check_env(env)

    # file_name = newest_zip_path

    # if exists(file_name):
    #     print("\nloading checkpoint " + file_name)
    #     model = PPO.load(file_name, env=env)
    #     model.n_steps = n_steps
    #     model.n_envs = num_cpu
    #     model.gamma = 0.95
    #     model.device = "cuda"
    #     model.rollout_buffer.reset()
    #     model.tensorboard_log = LOG_DIR
    #     model.n_epochs = 10

    # else:
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        n_steps=n_steps,
        batch_size=128,
        n_epochs=10,
        gamma=0.95,
        device="cuda",
        tensorboard_log=LOG_DIR,
    )

    model.learn(
        total_timesteps=5_000_000,
        callback=checkpoint_callback,
    )

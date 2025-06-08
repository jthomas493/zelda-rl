# Zelda-RL

**Train RL agents to play The Legend of Zelda**

This project uses reinforcement learning to train agents to play The Legend of Zelda, inspired by the [gym-zelda-1](https://github.com/Kautenja/gym-zelda-1) and [Zelda1AI](https://github.com/bjotho/Zelda1AI) projects, and drawing inspiration and techniques from projects like the Pokemon Red RL project. It leverages [Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3) and [PyBoy](https://github.com/Baekalfen/PyBoy) to create a powerful RL environment.

## Key Features

* **Gymnasium `Dict` Observation Space:** The environment provides observations as a dictionary, including both pixel data and game memory information. This allows the agent to learn from both visual and state-based inputs using Stable Baselines 3's `MultiInputPolicy`.
* **Parallel Training with `SubprocVecEnv`:** The project utilizes `SubprocVecEnv` from Stable Baselines 3 to parallelize environment instances, significantly speeding up training.
* **Memory Feature Integration:** In addition to the game screen pixels, the agent receives normalized memory values, such as player position and health, to aid in decision-making.
* **Customizable Reward Function:** The reward function is designed to encourage exploration, progress, and survival within the game. (See details below).
* **Integration with PyBoy:** Uses PyBoy for accurate game emulation and control.

## Setup Instructions

1.  **Create a new environment:**

    ```bash
    conda create --name zelda-rl
    ```
2.  **Activate the new environment:**

    ```bash
    conda activate zelda-rl
    ```
3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    It may be necessary in some cases to separately install the SDL libraries.
4.  **Run the training script:**

    ```bash
    python run_baseline_parallel.py
    ```

    This script utilizes `SubprocVecEnv` for parallel training. Ensure that `"headless": True` is set in your `env_config` for stable parallel execution.

## Code Structure

* `zelda_gym_env.py`: Contains the `ZeldaGymEnv` class, defining the Gymnasium environment for The Legend of Zelda. This includes the `Dict` observation space and the logic for interacting with the PyBoy emulator.
* `run_baseline_parallel.py`:  The main training script, using Stable Baselines 3 and `SubprocVecEnv` for parallel environment management.
* `memory_addresses.py`: Defines memory addresses used to extract relevant game state information.
* `requirements.txt`: Lists the Python dependencies required for the project.

## Observation Space

The `ZeldaGymEnv` uses a `gymnasium.spaces.Dict` to provide observations to the agent. This dictionary contains the following keys:

* `"pixels"`:  A `spaces.Box` representing the game screen pixels (shape: `(42, 42, 3)`, `dtype`: `np.uint8`).
* `"memory_values"`: A `spaces.Box` containing normalized game memory values (shape: `(num_memory_features,)`, `dtype`: `np.float32`).  These values include, but are not limited to, player position and health.  The number of memory features is determined by `num_memory_features` in the environment configuration.

## Reward Function

TODO: The reward function is a complicated work in progress. It is designed to encourage:

* **Exploration:** Rewarding the agent for visiting new areas or states.
* **Progress:** Rewarding the agent for advancing in the game, completing objectives, and collecting items.
* **Survival:** Penalizing the agent for taking damage or dying.

## Supporting Libraries

### [Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3)

[![Stable Baselines 3 Logo](https://youtu.be/DcYLT37ImBY)](https://github.com/DLR-RM/stable-baselines3)

### [PyBoy](https://github.com/Baekalfen/PyBoy)

[![PyBoy Logo](https://youtu.be/DcYLT37ImBY)](https://github.com/Baekalfen/PyBoy)

### References

* [The Legend of Zelda: RAM Map](https://datacrystal.romhacking.net/wiki/The_Legend_of_Zelda:RAM_map). _Data Crystal ROM Hacking_.
* [The Legend of Zelda: Memory Addresses](http://thealmightyguru.com/Games/Hacking/Wiki/index.php/The_Legend_of_Zelda#Memory_Addresses). _NES Hacker._

### Inspired By

* [gym-zelda-1](https://github.com/Kautenja/gym-zelda-1)
* [Zelda1AI](https://github.com/bjotho/Zelda1AI)
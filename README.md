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

The reward function in this environment is meticulously designed to guide Link through The Legend of Zelda: Link's Awakening by incentivizing various forms of progress and desired behaviors. It's a composite reward system that combines multiple metrics to provide a comprehensive signal to the reinforcement learning agent.

**Key Reward Components:**

The total reward is an aggregation of scores from several distinct categories, calculated at each step:

* **`event`**: Rewards for triggering various in-game events and flag progressions.
* **`level`**: Incentivizes collection of progression items (e.g., secret shells, ocarina songs, golden leaves, max health upgrades).
* **`died`**: A significant negative penalty incurred upon Link's death.
* **`instruments`**: Rewards for collecting the eight magical instruments required to awaken the Wind Fish.
* **`explore`**: Encourages exploration of new areas/screens, either via screen-based k-NN similarity or visited map coordinates.
* **`enemies`**: Rewards for defeating enemies.
* **`navigation`**: Provides positive feedback for moving closer to the current objective's target coordinates.
* **`objectives`**: A large positive reward for completing major in-game objectives or reaching specific key locations.
* **`equipped`**: A small penalty for having no items equipped in both A and B slots, encouraging active use of the inventory.

**Per-Step Reward Mechanism (Delta-Based):**

It's crucial to understand how the per-step reward (the value returned to the agent) is calculated. Internally, the environment maintains a cumulative score for each of the reward components listed above. However, the actual reward signal given to the agent for any given timestep is the **difference (delta)** between the *current total cumulative reward* and the *total cumulative reward from the previous step*.

This delta-based approach ensures that the agent is primarily rewarded for **making new progress** in that specific step, rather than simply for being in a good state that it already achieved. If no progress is made in a certain category during a step, its contribution to the per-step reward will be zero.

**Observation (`memory_values`) vs. Reward:**

It's important to differentiate between the agent's observation space and the reward function. The `memory_values` provided as part of the observation (e.g., player coordinates, health, item counts, objective coordinates) are **inputs** for the agent's decision-making process. They allow the agent to "see" and understand the current game state.

These `memory_values` themselves **do not directly contribute to the reward signal**. The reward is calculated separately based on *changes* in the game state that these memory values represent, or upon reaching specific objectives/milestones. There is no "double-counting" or accidental reward for simply observing certain memory values.

**Detailed Logging for Insight:**

For better monitoring and debugging of the learning process, the environment now provides detailed reward information in the `info` dictionary returned by the `step` function. This includes:

* **`reward_deltas`**: A breakdown of the individual changes contributed by each reward component for the current step.
* **`current_cumulative_rewards`**: The total accumulated score for each reward component up to the current step.

This `info` dictionary is also printed to the console periodically (e.g., every 1000 steps) to give a clear snapshot of the agent's progress across all reward streams.

## Supporting Libraries

### [Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3)

![alt text](https://github.com/DLR-RM/stable-baselines3/raw/master/docs//_static/img/logo.png)

### [PyBoy](https://github.com/Baekalfen/PyBoy)

![alt text](https://github.com/Baekalfen/PyBoy/raw/master/extras/README/pyboy.svg)

### References

* [The Legend of Zelda: RAM Map](https://datacrystal.romhacking.net/wiki/The_Legend_of_Zelda:RAM_map). _Data Crystal ROM Hacking_.
* [The Legend of Zelda: Memory Addresses](http://thealmightyguru.com/Games/Hacking/Wiki/index.php/The_Legend_of_Zelda#Memory_Addresses). _NES Hacker._

### Inspired By

* [gym-zelda-1](https://github.com/Kautenja/gym-zelda-1)
![alt text](https://user-images.githubusercontent.com/2184469/58208692-dae16580-7caa-11e9-82cf-2e870c681201.gif)
* [Zelda1AI](https://github.com/bjotho/Zelda1AI)
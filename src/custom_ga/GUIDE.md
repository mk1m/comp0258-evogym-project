# How to Test Custom Reward & Fitness Functions (Without Touching EvoGym)

This guide documents how to set up a custom experiment pipeline that keeps your changes isolated from the main `evogym` library.

## 1. Setup & Installation
1.  Clone the repository:
    ```bash
    git clone --recursive <repo_url>
    cd comp0258-evogym-project
    ```
2.  Run the setup script:
    ```bash
    bash scripts/setup_all.sh
    conda activate evogym_env
    ```

## 2. Isolate Your Work
To avoid merge conflicts and keep the repo clean:
1.  Create a `src/` directory.
2.  Add `src/` to your `.gitignore` file.
    *   *Why?* This lets you experiment freely without accidentally committing junk.

## 3. Create Custom GA
Instead of modifying `evogym/` directly, copy the necessary logic to `src/custom_ga/`.

We created three key files:
1.  **`my_utils.py`**:
    *   *Modified copy of* `evogym/examples/utils/algo_utils.py`.
    *   **Purpose:** Defines `Structure` class and `compute_fitness`.
    *   **Change:** Added `TASK_WEIGHTS` dictionary to calculate weighted fitness.
2.  **`my_ga.py`**:
    *   *Modified copy of* `evogym/examples/ga/run.py`.
    *   **Purpose:** The main training loop.
    *   **Change:** Updated to launch PPO jobs for *multiple tasks* per robot.
3.  **`run_experiment.py`**:
    *   *New file*.
    *   **Purpose:** The entry point script you actually run.
    *   **Usage:** Adds `--env-names` argument to specify target tasks.

## 4. How to Customize Fitness
Open `src/custom_ga/my_utils.py` and find `TASK_WEIGHTS` inside `compute_fitness()`:

```python
TASK_WEIGHTS = {
    'Walker-v0': 1.0,   # Maximize performance (Reward is good)
    'Carrier-v0': -1.0, # Minimize performance (Reward is bad)
    'Climber-v0': 0.0   # Ignore this task
}
```
*   Edit these weights to change what the GA optimizes for.

## 5. How to Run the Experiment
Run your custom script from the project root:

```bash
python src/custom_ga/run_experiment.py \
    --exp-name my_custom_test \
    --env-names Walker-v0 Carrier-v0 \
    --pop-size 25
```

## 6. Analyze Results
1.  Check `saved_data/my_custom_test/generation_X/output.txt`.
2.  It will list the **Combined Fitness** and the **Individual Rewards**:
    ```
    ID      Fitness     Rewards
    0       5.0         {'Walker-v0': 10.0, 'Carrier-v0': 5.0}
    ```
    *(In this example with weights 1.0 and -1.0, Fitness = 10.0 - 5.0 = 5.0)*

## 7. Resuming Experiments
If your experiment crashes or you stop it (Ctrl+C), you can continue from where you left off.

1.  Run the **exact same command** again.
2.  The script will detect the existing folder:
    ```
    THIS EXPERIMENT (my_custom_test) ALREADY EXISTS
    Override? (y/n/c):
    ```
3.  Type `c` and press **Enter**.
4.  It will ask: `Enter gen to start training on (0-indexed):`
5.  Enter the number of the last **fully completed** generation (check `saved_data/my_custom_test/generation_X/output.txt`).
    *   Example: If `generation_5` has an `output.txt` file, type `5`. The script will reload everything up to Gen 5 and continue.


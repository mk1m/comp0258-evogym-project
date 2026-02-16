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
1.  Create a `playground/` directory.
2.  Add `playground/` to your `.gitignore` file.
    *   *Why?* This lets you experiment freely without accidentally committing junk.

## 3. Create Custom GA Scaffold
Instead of modifying `evogym/` directly, copy the necessary logic to `playground/custom_ga/`.

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
Open `playground/custom_ga/my_utils.py` and find `TASK_WEIGHTS` inside `compute_fitness()`:

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
python playground/custom_ga/run_experiment.py \
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

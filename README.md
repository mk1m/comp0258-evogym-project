# comp0258-evogym-project

## Setup

### Prerequisites
- Anaconda or Miniconda installed.

### Installation
1. Clone with submodules:
   ```bash
   git clone --recursive https://github.com/ucl-team-35/comp0258-evogym-project.git
   
   cd comp0258-evogym-project
   ```

2. Install Git LFS:
   ```bash
   git lfs install
   ```
   (helps with large file storage)

3. Run Setup:
   ```bash
   bash scripts/setup_all.sh
   ```
   This will create the `evogym_env` conda environment and build the physics engine. It may take a few minutes.

4. Activate Environment:
   ```bash
   conda activate evogym_env
   ```

5. Verify Installation:
   Run the included test script to confirm everything is working correctly:
   ```bash
   python test_installation.py
   ```
   If successful, you will see `EvoGym is correctly installed and working!`.

## Quick Start (Recommended)

We created a simple script `simple_ga.py` that demonstrates the core EvoGym API without any complex file structures. 

**Run it here:**
```bash
python simple_ga.py
```
**What it does:**
1. Generates 5 random robots.
2. Evaluates them in the `Walker-v0` environment.
3. Prints their fitness scores.
4. Visualizes the best robot walking.

**(`simple_ga.py`):**
- `sample_robot((5,5))`: Creates a random robot body.
- `gym.make(..., body=body)`: Creates the environment for that specific robot.
- `env.step(action)`: Steps the physics simulation.

---

### Project Structure
- `simple_ga.py`: **Start here.** A minimal example for your group project.
- `evogym/`: The core library folder (do not modify).
- `evogym/examples/`: Complex examples provided by the library authors.
- `saved_data/`: Created by the complex examples to store training logs.
- `src/eval/`: Custom evaluation scripts for PPO training experiments.
- `src/custom_ga/`: Custom GA implementation for multi-task evolution.

---

## Evaluation Scripts (`src/eval/`)

Custom scripts for running and visualising PPO training experiments. PPO hyperparameters are matched to the official EvoGym defaults (`evogym/examples/ppo/args.py`) so results are directly comparable with `run_ppo.py`.

### Single body evaluation
Train a predefined or random robot on one or more tasks:
```bash
python src/eval/run_eval.py \
    --env-names Walker-v0 UpStepper-v0 \
    --body-type walker \
    --total-timesteps 600000 \
    --eval-interval 10000 \
    --n-envs 4 --n-evals 4 \
    --n-seeds 3 --no-fixed-seed \
    --exp-name walker_cross_task
```

### Cross-Task Transfer Matrix
Evaluate how well a body optimised for task A performs on task B:
```bash
# 1. Download best robots from HuggingFace (one-time)
pip install datasets
python src/eval/download_best_robots.py

# 2. Run the transfer matrix
chmod +x src/eval/run_transfer_matrix.sh
caffeinate -i ./src/eval/run_transfer_matrix.sh
```

### Random Body Baseline
Evaluate a fixed random 5x5 body across all four target environments to establish a performance baseline against evolved morphologies:
```bash
chmod +x src/eval/run_random_baseline.sh
caffeinate -i ./src/eval/run_random_baseline.sh
```

### Plotting results
```bash
python src/eval/plot_curves.py --exp-name walker_cross_task
```

---

## Miscellaneous

### 1. Data Ignored
Updated `.gitignore` to exclude `saved_data/` and `src/eval/logs/`.

### 2. Use Unique Experiment Names
When running evaluations, always use a unique `--exp-name` (e.g., in `run_eval.py`) so you don't accidentally merge or overwrite log files from different runs.

### 3. Create Your Own Folder
Don't edit files in `evogym/examples/` directly. Instead, create your own folder:
```bash
mkdir name_experiments
cp simple_ga.py name_experiments/my_ga.py
```
This way, you can commit your code without conflicting with someone else's code.


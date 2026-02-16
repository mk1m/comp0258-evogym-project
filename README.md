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

## For Full Examples

The `evogym` library includes a full-featured Genetic Algorithm (GA) example in `evogym/examples/run_ga.py`. This is creates a `saved_data` directory.

### 1. Run Full GA
```bash
python evogym/examples/run_ga.py --exp-name test_ga --pop-size 10 --max-evaluations 30
```
*Note: This will create a `saved_data/` folder with results.*

### 2. Visualize Full GA Results
```bash
python evogym/examples/visualize.py --env-name Walker-v0
```
Follow the prompts to select the experiment and robot to view.

---

### Project Structure
- `simple_ga.py`: **Start here.** A minimal example for your group project.
- `evogym/`: The core library folder (do not modify).
- `evogym/examples/`: Complex examples provided by the library authors.
- `saved_data/`: Created by the complex examples to store training logs.

---

## Best Practices

Follow these rules to avoid **Merge Conflicts** and **Data Overwrites**:

### 1. Data Ignored
Updated `.gitignore` to exclude `saved_data/`.

### 2. Use Unique Experiment Names
When running `run_ga.py`, always use a unique `--exp-name` so you don't overwrite your own previous results locally.
```bash
python evogym/examples/run_ga.py --exp-name alice_test_01 ...
```

### 3. Create Your Own Folder
Don't edit files in `evogym/examples/` directly. Instead, create your own folder:
```bash
mkdir alice_experiments
cp simple_ga.py alice_experiments/my_ga.py
```
This way, you can commit your code without conflicting with someone else's code.

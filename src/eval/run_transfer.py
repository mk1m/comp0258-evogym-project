"""
run_transfer_walker.py — Run Walker-v0 best body transfer experiments.

Identical in protocol to run_transfer_upstepper.py:
  - Training:     n_envs=1
  - Evaluation:   n_eval_envs=4, n_evals=4, eval_interval=10000
  - Duration:     600,000 timesteps
  - Runs:         3 independent (random seeds)
"""

import sys
import os

# Path setup
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'evogym', 'examples')))

import re
import argparse
import random
import numpy as np
import evogym.envs
from ppo.run import run_ppo

# ======================= CONFIG =======================
ROBOT_NPZ = os.path.join(os.path.dirname(__file__), "robots", "Walker_v0_best.npz")
SOURCE_ENV = "Walker-v0"
TARGET_ENVS = ["Walker-v0", "Balancer-v0", "Pusher-v0", "Carrier-v0"]
TOTAL_TIMESTEPS = 600_000
EVAL_INTERVAL = 10_000
N_EVALS = 4
N_EVAL_ENVS = 4
N_RUNS = 3
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs", "other_envs_walker_transfer")
# ======================================================

class TeeStream:
    def __init__(self, file_path, original_stdout):
        self.file = open(file_path, 'w')
        self.stdout = original_stdout
    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
        self.file.flush()
    def flush(self):
        self.file.flush()
        self.stdout.flush()
    def close(self):
        self.file.close()

def make_args():
    return argparse.Namespace(
        verbose_ppo=1, learning_rate=2.5e-4, n_steps=128, batch_size=4, n_epochs=4,
        gamma=0.99, gae_lambda=0.95, vf_coef=0.5, max_grad_norm=0.5, ent_coef=0.01,
        clip_range=0.1, total_timesteps=TOTAL_TIMESTEPS, log_interval=50,
        n_envs=1, n_eval_envs=N_EVAL_ENVS, n_evals=N_EVALS, eval_interval=EVAL_INTERVAL,
    )

def count_eval_lines(log_path):
    try:
        with open(log_path, 'r') as f:
            content = f.read()
        return len(re.findall(r'\[.*?\]\s*Mean:', content))
    except: return 0

if __name__ == "__main__":
    data = np.load(ROBOT_NPZ)
    body, connections = data['arr_0'], data['arr_1']

    print(f"Source: {SOURCE_ENV}, Log dir: {LOG_DIR}\n")
    os.makedirs(LOG_DIR, exist_ok=True)
    args = make_args()
    expected_evals = TOTAL_TIMESTEPS // EVAL_INTERVAL

    for target_env in TARGET_ENVS:
        for run_idx in range(N_RUNS):
            log_name = f"run_ppo_{SOURCE_ENV}_to_{target_env}_t{TOTAL_TIMESTEPS}_run{run_idx}.log"
            log_path = os.path.join(LOG_DIR, log_name)

            if count_eval_lines(log_path) >= expected_evals - 1:
                print(f"SKIP  {log_name}")
                continue

            print(f"\nSTART {log_name}")
            exp_name = log_name.replace('.log', '')
            save_dir = os.path.join(LOG_DIR, "saved_data")
            model_save_dir = os.path.join(save_dir, exp_name, "controller")
            structure_save_dir = os.path.join(save_dir, exp_name, "structure")
            os.makedirs(model_save_dir, exist_ok=True)
            os.makedirs(structure_save_dir, exist_ok=True)
            np.savez(os.path.join(structure_save_dir, target_env), body, connections)

            seed = random.randint(0, 2**31 - 1)
            tee = TeeStream(log_path, sys.stdout)
            old_stdout = sys.stdout
            sys.stdout = tee

            try:
                run_ppo(args=args, body=body, connections=connections, env_name=target_env,
                        model_save_dir=model_save_dir, model_save_name=target_env, seed=seed)
            finally:
                sys.stdout = old_stdout
                tee.close()

    print(f"\nRuns complete. Logs in: {LOG_DIR}")

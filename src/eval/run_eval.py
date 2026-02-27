"""
run_eval.py - Evaluate PPO training performance over time.

Custom script to evaluate robot morphologies across EvoGym tasks, featuring 
custom logging, matrix run support, and resume-capability.

Adapted from official EvoGym code (evogym/examples/ppo/run.py and eval.py), 
specifically the PPO setup and env wrapping with make_vec_env.

Usage (from project root):
    python src/eval/run_eval.py \
        --env-names Walker-v0 Balancer-v0 \
        --body-type walker \
        --total-timesteps 500000 \
        --eval-interval 35000 \
        --n-envs 4 \
        --n-evals 3 \
        --n-seeds 2 \
        --exp-name my_experiment
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'evogym', 'examples')))

import csv
import random
import argparse
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
import evogym.envs
from evogym import sample_robot
from evogym.utils import get_full_connectivity

# ---------- Predefined robot bodies ----------
# Voxel types: 0=empty, 1=rigid, 2=soft, 3=h_act, 4=v_act
# 'speed_bot' from evogym world_data/speed_bot.json. 
# Others ('walker', 'balancer', 'climber') are custom designs.
    # note: balancer body performs terribly even on balancer tasks

PREDEFINED_BODIES = {
    'walker': np.array([
        [0, 0, 3, 3, 0, 0],
        [0, 0, 3, 3, 0, 0],
        [3, 3, 3, 3, 3, 3],
        [3, 0, 0, 0, 0, 3],
        [4, 0, 0, 0, 0, 4],
    ]),
    'balancer': np.array([
        [0, 4, 4, 4, 0],
        [3, 1, 1, 1, 3],
        [3, 1, 1, 1, 3],
        [1, 1, 1, 1, 1],
    ]),
    'climber': np.array([
        [0, 4, 0],
        [3, 1, 3],
        [4, 1, 4],
        [3, 1, 3],
        [4, 1, 4],
    ]),
    'speed_bot': np.array([
        [3, 0, 0, 0, 3],
        [3, 1, 0, 1, 3],
        [1, 3, 3, 3, 1],
        [2, 3, 3, 3, 1],
    ]),
}

# Logging Callback

class LoggingCallback(BaseCallback):
    """
    Evaluates the policy every `eval_interval` steps and writes
    (timestep, mean_reward) rows to a CSV file.
    """
    def __init__(self, body, connections, env_name, eval_interval, n_evals, csv_path, verbose=0):
        super().__init__(verbose)
        self.body = body
        self.connections = connections
        self.env_name = env_name
        self.eval_interval = eval_interval
        self.n_evals = n_evals
        self.csv_path = csv_path
        self.model_path = csv_path.replace('.csv', '_model.zip')
        self._last_eval_step = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval_step >= self.eval_interval:
            self._last_eval_step = self.num_timesteps
            mean_reward = self._evaluate()
            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([self.num_timesteps, mean_reward])
            if self.verbose:
                print(f"  [{self.env_name}] t={self.num_timesteps:>7d}  mean_reward={mean_reward:.3f}")
            if self.model is not None:
                self.model.save(self.model_path)
        return True

    def _on_training_end(self):
        # Log final checkpoint
        mean_reward = self._evaluate()
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([self.num_timesteps, mean_reward])

    def _evaluate(self) -> float:
        # Env setup adapted from evogym/examples/ppo/eval.py
        eval_env = make_vec_env(self.env_name, n_envs=1, env_kwargs={
            'body': self.body,
            'connections': self.connections,
        })
        rewards = []
        for _ in range(self.n_evals):
            obs = eval_env.reset()
            total_r = 0.0
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, r, done, _ = eval_env.step(action)
                total_r += r[0]
            rewards.append(total_r)
        eval_env.close()
        return float(np.mean(rewards))


# Training

def train(body, connections, env_name, total_timesteps, eval_interval, n_evals, n_envs, csv_path, seed):
    """Run a single PPO training run and log to csv_path."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    start_timestep = 0
    model_path = csv_path.replace('.csv', '_model.zip')

    # Check if we can resume (need BOTH csv data AND a saved model)
    if os.path.exists(csv_path) and os.path.exists(model_path):
        import pandas as pd
        try:
            df = pd.read_csv(csv_path)
            if not df.empty and 'timestep' in df.columns:
                start_timestep = int(df['timestep'].max())
                print(f"  Resuming from timestep {start_timestep}")
        except Exception:
            pass
    elif os.path.exists(csv_path):
        print(f"  CSV exists but no model checkpoint found. Starting fresh.")

    if start_timestep >= total_timesteps:
        print(f"  Already completed {total_timesteps} steps. Skipping.")
        return

    # Create a fresh CSV if starting from 0
    if start_timestep == 0:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestep', 'mean_reward'])

    # Env setup adapted from evogym/examples/ppo/run.py
    env = make_vec_env(env_name, n_envs=n_envs, seed=seed, env_kwargs={
        'body': body,
        'connections': connections,
    })

    callback = LoggingCallback(
        body=body,
        connections=connections,
        env_name=env_name,
        eval_interval=eval_interval,
        n_evals=n_evals,
        csv_path=csv_path,
        verbose=1,
    )
    # Give callback the correct initial timestep
    callback.num_timesteps = start_timestep
    callback._last_eval_step = start_timestep

    # Model instantiation adapted from evogym/examples/ppo/run.py
    model_path = csv_path.replace('.csv', '_model.zip')
    if start_timestep > 0 and os.path.exists(model_path):
        print(f"  Loading model from {model_path}")
        model = PPO.load(model_path, env=env, custom_objects={'n_envs': n_envs})
        model.verbose = 1
    else:
        model = PPO("MlpPolicy", env, verbose=1, seed=seed)

    remaining_steps = total_timesteps - start_timestep
    if remaining_steps > 0:
        print(f"  Training for {remaining_steps} more steps...")
        # Reset callback num_timesteps property so it doesn't get confused
        model.learn(total_timesteps=remaining_steps, callback=callback, reset_num_timesteps=False)
    
    # Save the model for future resumption
    model.save(model_path)
    env.close()


# Entry point

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-names', nargs='+', required=True,
                        help='Environments to run, e.g. Walker-v0 Balancer-v0')
    parser.add_argument('--total-timesteps', type=int, default=1_000_000)
    parser.add_argument('--eval-interval',   type=int, default=10_000,
                        help='How often (steps) to evaluate and log')
    parser.add_argument('--n-envs',         type=int, default=4,
                        help='Number of parallel training environments')
    parser.add_argument('--n-evals',         type=int, default=5,
                        help='Number of episodes per evaluation')
    parser.add_argument('--n-seeds',         type=int, default=3,
                        help='Number of random seeds (for std band)')
    parser.add_argument('--structure-shape', type=int, nargs=2, default=[5, 5])
    parser.add_argument('--exp-name',        type=str, default='eval_experiment')
    parser.add_argument('--body-type',       type=str, default='random',
                        choices=['random'] + list(PREDEFINED_BODIES.keys()),
                        help='Robot body type (default: random)')
    parser.add_argument('--multi-task',      action='store_true',
                        help='Also run a joint multi-task condition on the same robot')
    args = parser.parse_args()

    # Fix one robot body (same for all runs)
    if args.body_type == 'random':
        random.seed(0)
        np.random.seed(0)
        body, connections = sample_robot(tuple(args.structure_shape))
    else:
        body = PREDEFINED_BODIES[args.body_type]
        connections = get_full_connectivity(body)
    print(f"Body type: {args.body_type}")
    print(f"Robot body:\n{body}\n")

    log_dir = os.path.join('src', 'eval', 'logs', args.exp_name)
    os.makedirs(log_dir, exist_ok=True)

    # ----- Single-task runs -----
    for env_name in args.env_names:
        label = env_name.replace('-', '_').replace('/', '_')
        print(f"\n=== Single-task: {env_name} ===")
        for seed in range(args.n_seeds):
            print(f"  Seed {seed+1}/{args.n_seeds}")
            csv_path = os.path.join(log_dir, f'{label}_seed{seed}.csv')
            train(body, connections, env_name,
                  args.total_timesteps, args.eval_interval, args.n_evals, args.n_envs,
                  csv_path, seed)

    # ----- Multi-task run (optional) -----
    # Trains on each task in sequence, logging separately.
    # This emulates allocating equal time to each task.
    if args.multi_task and len(args.env_names) >= 2:
        per_task_steps = args.total_timesteps // len(args.env_names)
        for seed in range(args.n_seeds):
            print(f"\n=== Multi-task (seed {seed+1}/{args.n_seeds}) ===")
            for env_name in args.env_names:
                label = env_name.replace('-', '_').replace('/', '_')
                csv_path = os.path.join(log_dir, f'multitask_{label}_seed{seed}.csv')
                print(f"  Task: {env_name}  steps: {per_task_steps}")
                train(body, connections, env_name,
                      per_task_steps, args.eval_interval, args.n_evals, args.n_envs,
                      csv_path, seed)

    print(f"\nDone. Logs saved to: {log_dir}")
    print(f"Now run: python src/eval/plot_curves.py --exp-name {args.exp_name}")

"""
run_eval.py - Evaluate PPO training performance across EvoGym tasks.

Custom script to evaluate robot morphologies, featuring custom logging,
matrix run support, and resume-capability.

PPO hyperparameters and training setup are matched EXACTLY to the official
EvoGym implementation (evogym/examples/ppo/run.py and ppo/args.py):
  - n_envs=1        (single training env, so total_timesteps = actual env steps)
  - n_steps=128     (rollout buffer = 128 samples)
  - batch_size=4    (32 mini-batches per epoch, 128 gradient updates per rollout)
  - n_epochs=4
  - learning_rate=2.5e-4
  - clip_range=0.1
  - ent_coef=0.01

Usage (from project root):
    # Using a predefined body (1M timesteps, matching original defaults):
    python src/eval/run_eval.py \
        --env-names Walker-v0 UpStepper-v0 \
        --body-type walker \
        --total-timesteps 1000000 \
        --eval-interval 10000 \
        --n-seeds 3 --no-fixed-seed \
        --exp-name my_experiment

    # Using a robot from HuggingFace dataset (.npz from download_best_robots.py):
    python src/eval/run_eval.py \
        --robot-npz src/eval/robots/UpStepper_v0_best.npz \
        --env-names Walker-v0 BridgeWalker-v0 UpStepper-v0 ObstacleTraverser-v0 \
        --total-timesteps 1000000 \
        --eval-interval 10000 \
        --n-seeds 3 --no-fixed-seed \
        --exp-name transfer_UpStepper_v0
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
import pandas as pd

# ---------- Predefined robot bodies ----------
# Voxel types: 0=empty, 1=rigid, 2=soft, 3=h_act, 4=v_act
# 'speed_bot' from evogym world_data/speed_bot.json. 
# Others ('walker', 'balancer', 'climber') are custom designs.
    # note: balancer body performs terribly even on balancer tasks so probably shouldn't use it

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
        """
        Evaluate the current policy using the EXACT same logic as
        evogym/examples/ppo/eval.py :: eval_policy().

        Key details matched to the original:
          - deterministic=False  (original default)
          - Reward zeroed AFTER cum_done is set (terminal step reward excluded)
          - Rewards summed over time axis, then averaged across evals
        """
        from ppo.eval import eval_policy
        rewards = eval_policy(
            model=self.model,
            body=self.body,
            connections=self.connections,
            env_name=self.env_name,
            n_evals=self.n_evals,
            n_envs=1,
            deterministic_policy=False,
        )
        return float(np.mean(rewards))


# Transfer score computation

def compute_transfer_score(csv_path, n_last=10):
    """
    Compute the final transfer score as the mean of the last `n_last`
    evaluation means from a CSV log. Matches the group mate's protocol:
    "mean of the last 10 evaluation means (500k-600k timesteps)".
    """
    try:
        df = pd.read_csv(csv_path)
        if df.empty or len(df) < n_last:
            return float(df['mean_reward'].mean()) if not df.empty else float('nan')
        return float(df['mean_reward'].tail(n_last).mean())
    except Exception:
        return float('nan')


# Training

def train(body, connections, env_name, total_timesteps, eval_interval, n_evals, n_envs, csv_path, seed):
    """Run a single PPO training run and log to csv_path."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    start_timestep = 0
    model_path = csv_path.replace('.csv', '_model.zip')

    # Check if we can resume (need BOTH csv data AND a saved model)
    if os.path.exists(csv_path) and os.path.exists(model_path):
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

    # PPO hyperparameters matched to evogym/examples/ppo/args.py defaults
    model_path = csv_path.replace('.csv', '_model.zip')
    if start_timestep > 0 and os.path.exists(model_path):
        print(f"  Loading model from {model_path}")
        model = PPO.load(model_path, env=env, custom_objects={'n_envs': n_envs})
        model.verbose = 1
    else:
        model = PPO(
            "MlpPolicy", env,
            verbose=1,
            seed=seed,
            learning_rate=2.5e-4,
            n_steps=128,
            batch_size=4,
            n_epochs=4,
            gamma=0.99,
            gae_lambda=0.95,
            vf_coef=0.5,
            max_grad_norm=0.5,
            ent_coef=0.01,
            clip_range=0.1,
        )

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
    parser.add_argument('--total-timesteps', type=int, default=1_000_000,
                        help='Total env steps (with n_envs=1 this equals actual '
                             'per-env experience). Original default: 1M')
    parser.add_argument('--eval-interval',   type=int, default=10_000,
                        help='How often (steps) to evaluate and log')
    parser.add_argument('--n-envs',         type=int, default=1,
                        help='Number of parallel training environments. '
                             'Original EvoGym default: 1. WARNING: changing '
                             'this alters gradient update frequency with '
                             'batch_size=4.')
    parser.add_argument('--n-evals',         type=int, default=1,
                        help='Number of episodes per evaluation '
                             '(original default: 1)')
    parser.add_argument('--n-seeds',         type=int, default=3,
                        help='Number of random seeds (for std band)')
    parser.add_argument('--structure-shape', type=int, nargs=2, default=[5, 5])
    parser.add_argument('--exp-name',        type=str, default='eval_experiment')
    parser.add_argument('--body-type',       type=str, default='random',
                        choices=['random'] + list(PREDEFINED_BODIES.keys()),
                        help='Robot body type (default: random)')
    parser.add_argument('--robot-npz',       type=str, default=None,
                        help='Path to .npz file with robot body/connections '
                             '(from download_best_robots.py or GA output). '
                             'Overrides --body-type.')
    parser.add_argument('--no-fixed-seed',   action='store_true',
                        help='Use random seeds for each trial instead of '
                             'deterministic 0,1,2... (for independent runs)')
    args = parser.parse_args()

    # Fix one robot body (same for all runs)
    if args.robot_npz:
        # Load from .npz file (HuggingFace dataset or GA output)
        data = np.load(args.robot_npz)
        body = data['arr_0']
        connections = data['arr_1']
        body_label = os.path.basename(args.robot_npz).replace('.npz', '')
        print(f"Loaded robot from: {args.robot_npz}")
    elif args.body_type == 'random':
        random.seed(0)
        np.random.seed(0)
        body, connections = sample_robot(tuple(args.structure_shape))
        body_label = 'random'
    else:
        body = PREDEFINED_BODIES[args.body_type]
        connections = get_full_connectivity(body)
        body_label = args.body_type
    print(f"Body type: {body_label}")
    print(f"Robot body:\n{body}\n")

    log_dir = os.path.join('src', 'eval', 'logs', args.exp_name)
    os.makedirs(log_dir, exist_ok=True)

    # ----- Single-task runs -----
    for env_name in args.env_names:
        label = env_name.replace('-', '_').replace('/', '_')
        print(f"\n=== Single-task: {env_name} ===")
        for trial in range(args.n_seeds):
            # Use random seeds if --no-fixed-seed, otherwise deterministic
            seed = random.randint(0, 2**31) if args.no_fixed_seed else trial
            print(f"  Trial {trial+1}/{args.n_seeds} (seed={seed})")
            csv_path = os.path.join(log_dir, f'{label}_seed{trial}.csv')
            train(body, connections, env_name,
                  args.total_timesteps, args.eval_interval, args.n_evals, args.n_envs,
                  csv_path, seed)


    # ----- Compute final transfer scores -----
    print(f"\n{'='*60}")
    print(f"FINAL TRANSFER SCORES (mean of last 10 evaluations)")
    print(f"{'='*60}")
    for env_name in args.env_names:
        label = env_name.replace('-', '_').replace('/', '_')
        scores = []
        for trial in range(args.n_seeds):
            csv_path = os.path.join(log_dir, f'{label}_seed{trial}.csv')
            score = compute_transfer_score(csv_path)
            scores.append(score)
        mean_score = np.nanmean(scores)
        std_score = np.nanstd(scores)
        print(f"  {env_name:25s}  mean={mean_score:.3f}  std={std_score:.3f}  (n={len(scores)})")

    print(f"\nDone. Logs saved to: {log_dir}")
    print(f"Now run: python src/eval/plot_curves.py --exp-name {args.exp_name}")

"""
run_eval.py - Evaluate PPO training performance across EvoGym tasks.

Custom script to evaluate robot morphologies, featuring custom logging,
matrix run support, and resume-capability.

PPO hyperparameters and env wrapping are matched to the official EvoGym
implementation (evogym/examples/ppo/run.py and ppo/args.py) so results
are directly comparable with run_ppo.py runs.

Usage (from project root):
    # Using a predefined body:
    python src/eval/run_eval.py \
        --env-names Walker-v0 Balancer-v0 \   # Which tasks to train on
        --body-type walker \                  # Robot morphology
        --total-timesteps 600000 \            # Total PPO training steps per env
        --eval-interval 10000 \               # Evaluate every N steps
        --n-envs 4 \                          # Parallel training envs
        --n-evals 4 \                         # Episodes averaged per eval
        --n-seeds 3 --no-fixed-seed \         # 3 independent trials
        --exp-name my_experiment              # Folder name for CSVs/models

    # Using a robot from HuggingFace dataset (.npz from download_best_robots.py):
    python src/eval/run_eval.py \
        --robot-npz src/eval/robots/UpStepper_v0_best.npz \
        --env-names Walker-v0 BridgeWalker-v0 UpStepper-v0 ObstacleTraverser-v0 \
        --total-timesteps 600000 \
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
        # Env setup adapted exactly from evogym/examples/ppo/eval.py
        # We run self.n_evals episodes and average their total rewards.
        # It's crucial we don't count rewards after an environment is 'done'.
        
        eval_env = make_vec_env(self.env_name, n_envs=1, env_kwargs={
            'body': self.body,
            'connections': self.connections,
        })
        
        rewards = []
        obs = eval_env.reset()
        cum_done = np.array([False])
        
        # We run until the environment signals done. Because we only use n_envs=1 for
        # evaluation here to keep it simple, we just run n_evals times.
        for _ in range(self.n_evals):
            obs = eval_env.reset()
            done = False
            total_r = 0.0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, r, dones, _ = eval_env.step(action)
                
                # In SB3 vec envs, 'dones' is an array of booleans.
                done = dones[0]
                
                # We only add reward if we weren't already done
                # (SB3 auto-resets, so the next step after done belongs to the *next* episode)
                if not done:
                    total_r += r[0]
                else:
                    # If it's the terminal step, we include the final terminal reward,
                    # but then the loop breaks so we don't include the auto-reset observation
                    total_r += r[0]

            rewards.append(total_r)
            
        eval_env.close()
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

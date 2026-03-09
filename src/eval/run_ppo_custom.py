import os
import shutil
import json
import argparse
import numpy as np
import csv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback

import evogym.envs
from evogym import WorldObject

# Need to adjust system path so imports work correctly
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'evogym', 'examples')))

from ppo.args import add_ppo_args
from ppo.eval import eval_policy


class CsvEvalCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.
    Logs the progress to a progress.csv in the model_save_dir.
    """
    def __init__(
        self,
        body: np.ndarray,
        env_name: str,
        eval_every: int,
        n_evals: int,
        n_envs: int,
        model_save_dir: str,
        model_save_name: str,
        connections=None,
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.body = body
        self.connections = connections
        self.env_name = env_name
        self.eval_every = eval_every
        self.n_evals = n_evals
        self.n_envs = n_envs
        self.model_save_dir = model_save_dir
        self.model_save_name = model_save_name
        
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
            
        self.csv_path = os.path.join(model_save_dir, "progress.csv")
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestep', 'mean_reward'])
            
        self.best_reward = -float('inf')
        
    def _on_training_start(self) -> None:
        pass

    def _on_rollout_start(self) -> None:
        pass

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_every == 0:
            self._validate_and_save()
        return True
        
    def _on_rollout_end(self) -> None:
        pass

    def _on_training_end(self) -> None:
        self._validate_and_save()
    
    def _validate_and_save(self) -> None:
        rewards = eval_policy(
            model=self.model,
            body=self.body,
            connections=self.connections,
            env_name=self.env_name,
            n_evals=self.n_evals,
            n_envs=self.n_envs,
        )
        out = f"[{self.model_save_name}] Mean: {np.mean(rewards):.3f}, Std: {np.std(rewards):.3f}, Min: {np.min(rewards):.3f}, Max: {np.max(rewards):.3f}"
        
        mean_reward = float(np.mean(rewards))
        
        # Log to CSV
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([self.num_timesteps, mean_reward])
            
        if mean_reward > self.best_reward:
            out += f" NEW BEST ({mean_reward:.3f} > {self.best_reward:.3f})"
            self.best_reward = mean_reward
            self.model.save(os.path.join(self.model_save_dir, self.model_save_name))
        if self.verbose > 0:
            print(out)


def run_ppo_custom(
    args: argparse.Namespace,
    body: np.ndarray,
    env_name: str,
    model_save_dir: str,
    model_save_name: str,
    connections=None,
    seed: int = 42,
) -> float:
    """
    Run ppo with the custom CSV callback and return the best reward achieved during evaluation.
    """
    
    # Parallel environments
    vec_env = make_vec_env(env_name, n_envs=args.n_envs, seed=seed, env_kwargs={
        'body': body,
        'connections': connections,
    })
    
    # Eval Callback
    callback = CsvEvalCallback(
        body=body,
        connections=connections,
        env_name=env_name,
        eval_every=args.eval_interval,
        n_evals=args.n_evals,
        n_envs=args.n_eval_envs,
        model_save_dir=model_save_dir,
        model_save_name=model_save_name,
        verbose=args.verbose_ppo,
    )

    # Train
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=args.verbose_ppo,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        ent_coef=args.ent_coef,
        clip_range=args.clip_range
    )
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callback,
        log_interval=args.log_interval
    )
    
    return callback.best_reward


if __name__ == "__main__":    
    
    # Args
    parser = argparse.ArgumentParser(description='Arguments for PPO script')
    parser.add_argument(
        "--env-name", default="Walker-v0", type=str, help="Environment name (default: Walker-v0)"
    )
    parser.add_argument(
        "--save-dir", default="saved_data", type=str, help="Parent directory in which to save data(default: saved_data)"
    )
    parser.add_argument(
        "--exp-name", default="test_ppo", type=str, help="Name of experiment. Data saved to <save-dir/exp-name> (default: test_ppo)"
    )
    parser.add_argument(
        "--robot-path", default=os.path.join("world_data", "speed_bot.json"), type=str, help="Path to the robot json file (default: world_data/speed_bot.json)"
    )
    add_ppo_args(parser)
    args = parser.parse_args()
    
    # Manage dirs
    exp_dir = os.path.join(args.save_dir, args.exp_name)
    if os.path.exists(exp_dir):
        print(f'THIS EXPERIMENT ({args.exp_name}) ALREADY EXISTS')
        print("Delete and override? (y/n): ", end="")
        ans = input()
        if ans.lower() != "y":
            exit()
        shutil.rmtree(exp_dir)
    model_save_dir = os.path.join(args.save_dir, args.exp_name, "controller")
    structure_save_dir = os.path.join(args.save_dir, args.exp_name, "structure")
    save_name = f"{args.env_name}"

    # Get Robot
    # Adjusted to load npz files or json files based on extensions
    if args.robot_path.endswith('.npz'):
        data = np.load(args.robot_path)
        body = data['arr_0']
        connections = data['arr_1'] if data['arr_1'].size > 0 else np.array([])
    else:    
        robot = WorldObject.from_json(args.robot_path)
        body = robot.get_structure()
        connections = robot.get_connections()

    os.makedirs(structure_save_dir, exist_ok=True)
    np.savez(os.path.join(structure_save_dir, save_name), body, connections)

    # Train
    best_reward = run_ppo_custom(
        args=args,
        body=body,
        connections=connections,
        env_name=args.env_name,
        model_save_dir=model_save_dir,
        model_save_name=save_name,
    )
    
    # Save result file
    with open(os.path.join(args.save_dir, args.exp_name, "ppo_result.json"), "w") as f:
        json.dump({
            "best_reward": best_reward,
            "env_name": args.env_name,
        }, f, indent=4)

    # Evaluate
    model = PPO.load(os.path.join(model_save_dir, save_name))
    rewards = eval_policy(
        model=model,
        body=body,
        connections=connections,
        env_name=args.env_name,
        n_evals=args.n_evals,
        n_envs=args.n_eval_envs,
        render_mode="human",
    )
    print(f"Mean reward: {np.mean(rewards)}")

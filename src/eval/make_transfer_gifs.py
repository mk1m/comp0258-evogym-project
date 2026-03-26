"""
make_transfer_gifs.py - Generate GIFs for the best controllers in a transfer study.

Usage:
    python src/eval/make_transfer_gifs.py --exp-name other_envs_walker_transfer
"""

import os
import sys
import re
import glob
import argparse
import numpy as np
import imageio
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Path setup for evogym imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'evogym', 'examples')))
import evogym.envs

EVAL_LINE_RE = re.compile(
    r"\[([^\]]+)\]\s*Mean:\s*([-0-9.eE+]+)"
)

def get_best_run(log_dir, target_env):
    """Find the run with the highest final evaluation mean for a given target env."""
    logs = glob.glob(os.path.join(log_dir, f"run_ppo_*_to_{target_env}_*_run*.log"))
    best_score = -float('inf')
    best_run_name = None
    
    for log_path in logs:
        with open(log_path, 'r') as f:
            content = f.read()
        eval_matches = EVAL_LINE_RE.findall(content)
        if not eval_matches:
            continue
        
        # Filter for target env
        target_means = []
        for m in eval_matches:
            if m[0].strip() == target_env:
                try:
                    # Strip any trailing commas or spaces just in case
                    val = float(m[1].strip().strip(','))
                    target_means.append(val)
                except ValueError:
                    continue
        
        if not target_means:
            continue
            
        final_mean = np.mean(target_means[-1:]) # Last evaluation
        if final_mean > best_score:
            best_score = final_mean
            best_run_name = os.path.basename(log_path).replace('.log', '')
            
    return best_run_name

def save_robot_gif(out_path, env_name, body_path, ctrl_path, seed=42):
    print(f"  --> Loading robot from: {body_path}")
    data = np.load(body_path)
    body, connections = data['arr_0'], data['arr_1']
    
    print(f"  --> Loading model from: {ctrl_path}")
    model = PPO.load(ctrl_path)
    
    print(f"  --> Initializing environment: {env_name}")
    # Setup environment
    vec_env = make_vec_env(env_name, n_envs=1, seed=seed, env_kwargs={
        'body': body,
        'connections': connections,
        'render_mode': 'img',
    })
    
    print(f"  --> Environment created. Resetting...")
    obs = vec_env.reset()
    
    print(f"  --> Initial render...")
    imgs = [vec_env.env_method('render')[0]]
    
    print(f"  --> Starting simulation loop (max 500 steps)...")
    done = False
    step = 0
    max_steps = 500 
    
    while not done and step < max_steps:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        
        # Render
        img = vec_env.env_method('render')[0]
        imgs.append(img)
        step += 1
        if step % 100 == 0:
            print(f"      - Step {step}")

    print(f"  --> Saving GIF: {out_path}.gif")
    if imgs:
        imageio.mimsave(f'{out_path}.gif', imgs, duration=(1/50.0))
        print(f"  Success: {out_path}.gif")
    else:
        print(f"  Failed: No images captured for {env_name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, required=True)
    parser.add_argument('--max-envs', type=int, default=10)
    args = parser.parse_args()

    log_dir = os.path.join('src', 'eval', 'logs', args.exp_name)
    save_dir = os.path.join('src', 'eval', 'plots', 'gifs', args.exp_name)
    os.makedirs(save_dir, exist_ok=True)

    print(f"Searching for logs in: {log_dir}")
    # Identify all target environments in this study
    log_files = glob.glob(os.path.join(log_dir, "*.log"))
    if not log_files:
        print(f"ERROR: No log files found matching run_ppo_*.log in {log_dir}")
        return

    targets = set()
    for f in log_files:
        # run_ppo_Source_to_Target_t...
        m = re.search(r'_to_(.+?)_t', os.path.basename(f))
        if m:
            targets.add(m.group(1))
    
    print(f"Detected {len(targets)} target environments: {sorted(list(targets))}")
    
    for target in sorted(list(targets)):
        best_run = get_best_run(log_dir, target)
        if not best_run:
            print(f"No completed runs found for {target}")
            continue
            
        print(f"Best run for {target}: {best_run}")
        
        body_path = os.path.join(log_dir, "saved_data", best_run, "structure", f"{target}.npz")
        ctrl_path = os.path.join(log_dir, "saved_data", best_run, "controller", f"{target}.zip")
        out_path = os.path.join(save_dir, f"{target}_best")
        
        if os.path.exists(ctrl_path):
            try:
                save_robot_gif(out_path, target, body_path, ctrl_path)
            except Exception as e:
                print(f"Error generating GIF for {target}: {e}")
        else:
            print(f"Controller not found at {ctrl_path}")

if __name__ == "__main__":
    main()

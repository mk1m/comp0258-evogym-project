"""
visualize_transfer.py - Parse transfer logs and plot learning curves.

Usage (from project root):
    python src/eval/visualize_transfer.py --exp-name other_envs_jumper_transfer
"""

import os
import re
import argparse
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# --- CONFIG ---
EVAL_INTERVAL = 10000 

ENV_COLORS = {
    'Walker-v0': '#2171b5',           # Blue
    'BridgeWalker-v0': '#6baed6',     # Light Blue
    'Pusher-v0': '#e6550d',           # Orange
    'Carrier-v0': '#31a354',          # Green
    'Flipper-v0': '#756bb1',          # Purple
    'Jumper-v0': '#c72e2e',           # Red
    'Balancer-v0': '#636363',         # Dark Grey
    'UpStepper-v0': '#fd8d3c',        # Light Orange
    'ObstacleTraverser-v0': '#8c6d31' # Brown
}

# Fallback palette for any unexpected environments
COLORS = ['#2171b5', '#e6550d', '#31a354', '#756bb1', '#c72e2e', '#636363', '#969696']

EVAL_LINE_RE = re.compile(
    r"\[([^\]]+)\]\s*Mean:\s*([0-9.+-eE]+),\s*Std:\s*([0-9.+-eE]+),\s*Min:\s*([0-9.+-eE]+),\s*Max:\s*([0-9.+-eE]+)"
)

FNAME_RE = re.compile(
    r"run_ppo_(?P<Source>.+?)_to_(?P<Target>.+?)_t(?P<T>[0-9]+)_run(?P<Run>[0-9]+)\.log$"
)

def parse_log(path):
    fn = os.path.basename(path)
    m = FNAME_RE.search(fn)
    if not m:
        return None
    
    source = m.group("Source")
    target = m.group("Target")
    run = int(m.group("Run"))
    
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    
    eval_matches = EVAL_LINE_RE.findall(content)
    if not eval_matches:
        return None
    
    # Filter only evaluations matching the target env tag
    means = []
    for tag, mean, std, min_val, max_val in eval_matches:
        if tag.lower().strip() == target.lower().strip():
            means.append(float(mean))
    
    if not means:
        return None
        
    return {
        "source": source,
        "target": target,
        "run": run,
        "means": np.array(means)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, required=True)
    parser.add_argument('--title', type=str, default=None)
    parser.add_argument('--smoothing', type=float, default=1.0, help="Smoothing alpha (1.0 = no smoothing)")
    args = parser.parse_args()

    log_dir = os.path.join('src', 'eval', 'logs', args.exp_name)
    plot_dir = os.path.join('src', 'eval', 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    log_files = glob.glob(os.path.join(log_dir, "*.log"))
    if not log_files:
        print(f"No log files found in {log_dir}")
        return

    data = []
    for f in log_files:
        parsed = parse_log(f)
        if parsed:
            data.append(parsed)
            
    if not data:
        print("No valid data parsed from logs.")
        return

    # Group by target environment
    targets = sorted(list(set(d['target'] for d in data)))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, target in enumerate(targets):
        run_data = [d['means'] for d in data if d['target'] == target]
        if not run_data:
            continue
            
        # Determine common length
        min_len = min(len(rd) for rd in run_data)
        run_data = [rd[:min_len] for rd in run_data]
        
        matrix = np.vstack(run_data)
        mean_curve = matrix.mean(axis=0)
        std_curve = matrix.std(axis=0)
        
        # Apply smoothing if requested
        if args.smoothing < 1.0:
            mean_curve = pd.Series(mean_curve).ewm(alpha=args.smoothing, adjust=False).mean().values
            std_curve = pd.Series(std_curve).ewm(alpha=args.smoothing, adjust=False).mean().values
            
        timesteps = np.arange(1, min_len + 1) * EVAL_INTERVAL
        
        label = target.replace('-v0', '')
        color = ENV_COLORS.get(target, COLORS[i % len(COLORS)])
        
        ax.plot(timesteps, mean_curve, label=f"{label} (n={len(run_data)})", color=color, linewidth=2)
        ax.fill_between(timesteps, mean_curve - std_curve, mean_curve + std_curve, alpha=0.15, color=color)

    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Mean Reward')
    
    source_env = data[0]['source']
    title = args.title or f"Transfer Performance: {source_env} Body across Enviornments"
    ax.set_title(title)
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x/1e3)}k'))
    
    plt.tight_layout()
    out_path = os.path.join(plot_dir, f"{args.exp_name}.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to: {out_path}")

if __name__ == "__main__":
    main()

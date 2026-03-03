"""
plot_curves.py - Plot learning curves from run_eval.py output.

Usage (from project root):
    python src/eval/plot_curves.py --exp-name single_task_comparison

This reads CSVs from:
    src/eval/logs/<exp-name>/<condition>_seed<N>.csv

And produces:
    src/eval/plots/<exp-name>.png
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import argparse
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Plotting helpers

COLORS = ['#2171b5', '#e6550d', '#31a354', '#756bb1', '#636363']

def load_condition(csv_paths, smoothing=0.0):
    """
    Load multiple seed CSVs for one condition and interpolate onto a
    common timestep grid so we can average across seeds.
    Applies EMA smoothing if smoothing > 0.
    """
    dfs = [pd.read_csv(p) for p in csv_paths]
    
    # Common x grid: finest resolution across seeds
    all_steps = sorted(set(np.concatenate([df['timestep'].values for df in dfs])))
    grid = np.array(all_steps)
    
    interp_rewards = []
    for df in dfs:
        x = df['timestep'].values
        y = df['mean_reward'].values
        if smoothing > 0:
            y = pd.Series(y).ewm(alpha=smoothing, adjust=False).mean().values
        interp = np.interp(grid, x, y)
        interp_rewards.append(interp)
    
    return grid, np.array(interp_rewards)  # shape: (n_seeds, n_steps)


def plot_condition(ax, grid, rewards_matrix, label, color, hide_std=False):
    mean = rewards_matrix.mean(axis=0)
    std  = rewards_matrix.std(axis=0)
    
    # Smooth the std to prevent jagged fill bands that cause noise artifacting
    std = pd.Series(std).ewm(alpha=0.1, adjust=False).mean().values
    
    ax.plot(grid, mean, label=label, color=color, linewidth=1.5)
    if not hide_std:
        ax.fill_between(grid, mean - std, mean + std, alpha=0.15, color=color)


# Main 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, required=True)
    parser.add_argument('--title',    type=str, default=None)
    parser.add_argument('--smoothing', type=float, default=0.2,
                        help='EMA smoothing alpha vs past values (e.g. 0.1=highly smoothed, 1.0=no smoothing). Default: 0.2')
    parser.add_argument('--hide-std', action='store_true', help='Do not plot standard deviation bands')
    args = parser.parse_args()

    log_dir  = os.path.join('src', 'eval', 'logs', args.exp_name)
    plot_dir = os.path.join('src', 'eval', 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    # Discover all CSVs and group them by condition (everything before _seed<N>)
    all_csvs = sorted(glob.glob(os.path.join(log_dir, '*.csv')))
    if not all_csvs:
        print(f"No CSV files found in {log_dir}")
        sys.exit(1)

    # Group by condition name
    conditions = {}
    for path in all_csvs:
        fname = os.path.basename(path)
        # Strip trailing _seed<N>.csv
        m = re.match(r'^(.+)_seed\d+\.csv$', fname)
        if m:
            cond = m.group(1)
            conditions.setdefault(cond, []).append(path)
        else:
            print(f"Skipping unrecognised file: {fname}")

    print(f"Found {len(conditions)} condition(s): {list(conditions.keys())}")

    # Plot
    fig, ax = plt.subplots(figsize=(9, 5))

    for i, (cond, paths) in enumerate(sorted(conditions.items())):
        n_seeds = len(paths)
        label = cond.replace('_', ' ').replace('v0', 'v0').replace('v1', 'v1')
        # Pretty-print env names
        label = re.sub(r'([A-Z])', r' \1', label).strip()

        color = COLORS[i % len(COLORS)]
        grid, rewards_matrix = load_condition(paths, smoothing=args.smoothing)
        plot_condition(ax, grid, rewards_matrix, label=f"{label} (n={n_seeds})", color=color, hide_std=args.hide_std)

    ax.set_xlabel('Training timesteps', fontsize=12)
    ax.set_ylabel('Evaluation reward', fontsize=12)
    
    if args.title:
        title = args.title
    else:
        # Generate a nice title from the exp-name
        clean_name = args.exp_name.replace('transfer_', '').replace('verification_', '')
        clean_name = re.sub(r'([A-Z])', r' \1', clean_name).strip()
        title = f'Evaluation Performance: {clean_name} Body\n'
        
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M' if x >= 1e6 else f'{int(x/1e3)}k'))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(plot_dir, f'{args.exp_name}.png')
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to: {out_path}")


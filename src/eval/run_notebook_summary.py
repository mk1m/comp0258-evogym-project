import re, os, json, glob, math
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Optional
import argparse

# ---- CONFIG (Defaults) ----
PATTERN = "run_ppo_*_run*.log"  
LAST_K_EVALS = 10      
TARGET_ENV = None      
EVAL_INTERVAL_FALLBACK = 10000 

EVAL_LINE_RE = re.compile(
    r"\[([^\]]+)\]\s*Mean:\s*([0-9.+-eE]+),\s*Std:\s*([0-9.+-eE]+),\s*Min:\s*([0-9.+-eE]+),\s*Max:\s*([0-9.+-eE]+)"
)
TIMESTEP_RE = re.compile(r"\|\s*total_timesteps\s*\|\s*([0-9]+)\s*\|")
FNAME_RE = re.compile(
    r"run_ppo_(?P<A>.+?)_to_(?P<B>.+?)_t(?P<T>[0-9]+)_run(?P<R>[0-9]+)\.log$"
)

def parse_one_log(path: str, target_env: Optional[str] = None):
    p = Path(path)
    m = FNAME_RE.search(p.name)
    if not m:
        raise ValueError(f"Filename doesn't match expected pattern: {p.name}")

    A = m.group("A")
    B = m.group("B")
    T = int(m.group("T"))
    run = int(m.group("R"))

    with open(p, "rb") as f:
        text = f.read().decode("utf-8", errors="ignore")

    eval_matches = EVAL_LINE_RE.findall(text)
    if not eval_matches:
        raise ValueError(f"No evaluation lines found in {p.name}")

    env_tags = [em[0] for em in eval_matches]
    means = [float(em[1]) for em in eval_matches]
    
    if target_env is None:
        target_env = B
    
    keep_idx = [i for i, tag in enumerate(env_tags) if tag == target_env]
    if not keep_idx:
         keep_idx = [i for i, tag in enumerate(env_tags) if target_env in tag or tag in target_env]

    if keep_idx:
        means = [means[i] for i in keep_idx]

    df = pd.DataFrame({"mean": means}).dropna(subset=["mean"])
    tail = df["mean"].tail(LAST_K_EVALS).to_numpy()
    final_score = float(np.mean(tail)) if len(tail) else float("nan")

    return {
        "file": str(p),
        "A": A,
        "B": B,
        "T": T,
        "run": run,
        "final_score_mean_lastK": final_score,
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Summarize EvoGym PPO logs using notebook logic.')
    parser.add_argument('--exp-name', type=str, default="other_envs_walker_transfer", help='Folder name in src/eval/logs/')
    args = parser.parse_args()

    LOG_DIR = os.path.join('src', 'eval', 'logs', args.exp_name)
    if not os.path.exists(LOG_DIR):
        print(f"Error: {LOG_DIR} not found.")
        exit(1)

    paths = sorted(glob.glob(os.path.join(LOG_DIR, PATTERN)))
    if not paths:
        print(f"No logs found in {LOG_DIR} matching {PATTERN}")
        exit(1)

    parsed = []
    for path in paths:
        try:
            parsed.append(parse_one_log(path, target_env=TARGET_ENV))
        except Exception as e:
            print(f"Skipping {Path(path).name}: {e}")

    if not parsed:
        print("No logs successfully parsed.")
        exit(1)

    rows = []
    df_parsed = pd.DataFrame(parsed)
    for key, group in df_parsed.groupby(["A", "B"]):
        finals = group["final_score_mean_lastK"].to_numpy(dtype=float)
        mean_final = float(np.mean(finals))
        std_final = float(np.std(finals, ddof=1)) if len(finals) > 1 else 0.0

        rows.append({
            "Source": key[0],
            "Target": key[1],
            "n": int(len(group)),
            f"Mean (last {LAST_K_EVALS})": round(mean_final, 3),
            "Std": round(std_final, 3)
        })

    summary_df = pd.DataFrame(rows)
    print(f"\n--- Results for {args.exp_name} (Notebook Logic) ---")
    print(summary_df.to_string(index=False))

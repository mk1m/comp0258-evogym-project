#!/bin/bash
# Run the transfer matrix rows for UpStepper-v0 and ObstacleTraverser-v0.
# (already ran Walker-v0 and BridgeWalker-v0 rows using run_ppo.py.)
#
# Prerequisites:
#   pip install datasets
#   python src/eval/download_best_robots.py
#
# Usage:
#   chmod +x src/eval/run_transfer_matrix.sh
#   caffeinate -i ./src/eval/run_transfer_matrix.sh
#
# Protocol (matches group mate's experimental design):
#   - Source bodies: best robot per task from HuggingFace dataset
#   - 4 target environments
#   - 3 independent trials per pair (no fixed seeds)
#   - 600k timesteps, 10k eval interval, 4 eval episodes
#   - PPO hyperparameters matched to evogym/examples/ppo/args.py
#   - Final score = mean of last 10 evaluation means

set -e

ROBOT_DIR="src/eval/robots"
TARGET_ENVS="Walker-v0 BridgeWalker-v0 UpStepper-v0 ObstacleTraverser-v0"
TIMESTEPS=1000000
EVAL_INTERVAL=10000
N_SEEDS=1
# run_eval.py defaults (n_envs=1, n_evals=1).

SOURCE_TASKS=("UpStepper_v0" "ObstacleTraverser_v0")

for SOURCE in "${SOURCE_TASKS[@]}"; do
    ROBOT_PATH="${ROBOT_DIR}/${SOURCE}_best.npz"

    if [ ! -f "$ROBOT_PATH" ]; then
        echo "ERROR: Robot file not found: $ROBOT_PATH"
        echo "Run: python src/eval/download_best_robots.py"
        exit 1
    fi

    echo ""
    echo "============================================="
    echo "  Source body: ${SOURCE} -> all target envs"
    echo "============================================="
    echo ""

    python src/eval/run_eval.py \
        --robot-npz "$ROBOT_PATH" \
        --env-names $TARGET_ENVS \
        --total-timesteps $TIMESTEPS \
        --eval-interval $EVAL_INTERVAL \
        --n-seeds $N_SEEDS \
        --no-fixed-seed \
        --exp-name "transfer_${SOURCE}_final"
done

echo ""
echo "============================================="
echo "  Transfer matrix rows complete!"
echo "============================================="
echo ""
echo "Results printed above. To generate plots:"
echo "  python src/eval/plot_curves.py --exp-name transfer_UpStepper_v0"
echo "  python src/eval/plot_curves.py --exp-name transfer_ObstacleTraverser_v0"

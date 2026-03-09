#!/bin/bash
# run_random_baseline.sh
# Tests a random 5x5 body on all four target environments

# Define target environments
declare -a ENVS=(
    "Walker-v0"
    "BridgeWalker-v0"
    "UpStepper-v0"
    "ObstacleTraverser-v0"
)

# Number of trials per environment pair
TRIALS=3

# Output directory base name
EXP_NAME="transfer_random_baseline"

echo "==================================================="
echo "  Starting Random Body Baseline Evaluation"
echo "==================================================="

# Note: We use the exact same random seed to generate the shape, 
# but different seeds for the PPO controllers. The python script
# does this automatically when body-type=random and no-fixed-seed is on.

for TARGET in "${ENVS[@]}"; do
    echo "==================================================="
    echo "  Evaluating Random Body on Task: $TARGET"
    echo "==================================================="
    
    
    source ~/opt/anaconda3/etc/profile.d/conda.sh
    conda activate evogym_env
    
    python src/eval/run_eval.py \
        --exp-name "${EXP_NAME}_final" \
        --env-names "${TARGET}" \
        --body-type "random" \
        --structure-shape 5 5 \
        --eval-interval 10000 \
        --total-timesteps 1000000 \
        --n-seeds ${TRIALS} \
        --no-fixed-seed
        
    echo "[Done] Random Body -> $TARGET"
    echo ""
done

echo "==================================================="
echo "  All random baseline evaluations complete!"
echo "  You can plot results using:"
echo "  python src/eval/plot_curves.py --exp-name ${EXP_NAME}"
echo "==================================================="

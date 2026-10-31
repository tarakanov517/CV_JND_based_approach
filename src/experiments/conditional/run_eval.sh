#!/bin/bash
#SBATCH --job-name=cornet-conditional-eval
#SBATCH --partition=rocky
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --time=1-00:00:00
#SBATCH --array=0-2%3
#SBATCH --output=logs/conditional_eval_%A_%a.out
#SBATCH --mail-user=arilmusaev@edu.hse.ru
#SBATCH --mail-type=END,FAIL

set -euo pipefail

PROJECT_DIR="$SLURM_SUBMIT_DIR"
cd "$PROJECT_DIR"

source "$HOME/venvs/cornet/bin/activate"

export PYTHONUNBUFFERED=1
export HF_HOME="/home/arilmusaev/19.10.2026/.cache/huggingface"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

SEEDS=(42 43 44)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"

mkdir -p logs results/conditional_yoto

echo "conditional evaluation seed=$SEED"

python3 evaluate_conditional.py \
    --model-dir "results/conditional_yoto/conditional_axon/seed_$SEED" \
    --baseline-dir "results/profiles/baseline/seed_$SEED" \
    --output-dir results/conditional_yoto \
    --dataset "ilee0022/Caltech-256" \
    --seed "$SEED" \
    --batch-size 8 \
    --workers 8 \
    --mc-samples 3 \
    --eot-samples 4 \
    --whitebox-examples 512 \
    --attack-epsilons 2 4 8 \
    --pgd-steps 10

#!/bin/bash
#SBATCH --job-name=cornet-adaptive-ext-transfer
#SBATCH --partition=rocky
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --time=1-00:00:00
#SBATCH --array=0-2%3
#SBATCH --output=logs/adaptive_extended_transfer_%A_%a.out
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

mkdir -p logs

python3 transfer_adaptive.py \
    --source-results-dir results/profiles \
    --target-results-dir results/adaptive_extended \
    --seed "$SEED" \
    --batch-size 16 \
    --workers 8 \
    --attack-epsilons 2 4 8 \
    --pgd-steps 10

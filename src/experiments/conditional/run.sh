#!/bin/bash
#SBATCH --job-name=cornet-conditional
#SBATCH --partition=rocky
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --time=1-00:00:00
#SBATCH --array=0-2%3
#SBATCH --output=logs/conditional_train_%A_%a.out
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

echo "conditional training seed=$SEED"

python3 main_conditional.py \
    --name conditional_axon \
    --baseline-dir "results/profiles/baseline/seed_$SEED" \
    --output-dir results/conditional_yoto \
    --dataset "ilee0022/Caltech-256" \
    --seed "$SEED" \
    --batch-size 32 \
    --workers 8 \
    --epochs 100 \
    --learning-rate 0.0001 \
    --weight-decay 0.0001 \
    --max-sigma 1.0 \
    --anchor-sigma 0.85 \
    --conditioner-hidden 128 \
    --noise-warmup-epochs 5 \
    --noise-ramp-epochs 25 \
    --zero-probability 0.20 \
    --shared-probability 0.30 \
    --single-probability 0.30

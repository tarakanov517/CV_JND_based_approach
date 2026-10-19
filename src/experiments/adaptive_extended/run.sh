#!/bin/bash
#SBATCH --job-name=cornet-adaptive-ext
#SBATCH --partition=rocky
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --time=1-00:00:00
#SBATCH --array=0-5%3
#SBATCH --output=logs/adaptive_extended_%A_%a.out
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
SEED_INDEX=$((SLURM_ARRAY_TASK_ID % 3))
ORDER_INDEX=$((SLURM_ARRAY_TASK_ID / 3))
SEED="${SEEDS[$SEED_INDEX]}"

mapfile -t ORDERS < experiments/adaptive_extended/orders.txt
ORDER="${ORDERS[$ORDER_INDEX]}"
IFS='|' read -r NAME BLOCK1 BLOCK2 BLOCK3 BLOCK4 <<< "$ORDER"

mkdir -p results/adaptive_extended logs

echo "experiment=$NAME seed=$SEED order=[$BLOCK1,$BLOCK2,$BLOCK3,$BLOCK4]"

python3 main_adaptive.py \
    --name "$NAME" \
    --order "$BLOCK1" "$BLOCK2" "$BLOCK3" "$BLOCK4" \
    --baseline-dir "results/profiles/baseline/seed_$SEED" \
    --output-dir results/adaptive_extended \
    --dataset "ilee0022/Caltech-256" \
    --seed "$SEED" \
    --batch-size 32 \
    --workers 8 \
    --levels 0.10 0.25 0.40 0.55 0.70 0.85 1.00 1.25 1.50 1.75 2.00 2.25 2.50 \
    --epochs-per-level 4 \
    --consolidation-epochs 42 \
    --clean-drop-limit 0.10 \
    --controller-examples 512 \
    --controller-epsilon 4 \
    --learning-rate 0.0001 \
    --weight-decay 0.0001 \
    --attack-epsilons 2 4 8 \
    --pgd-steps 10

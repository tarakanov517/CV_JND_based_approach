#!/bin/bash
#SBATCH --job-name=cornet-adaptive
#SBATCH --partition=rocky
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --time=1-00:00:00
#SBATCH --array=0-8%3
#SBATCH --output=logs/adaptive_%A_%a.out
#SBATCH --mail-user=arilmusaev@edu.hse.ru
#SBATCH --mail-type=END,FAIL

set -euo pipefail

PROJECT_DIR="$SLURM_SUBMIT_DIR"
cd "$PROJECT_DIR"

source "$HOME/venvs/cornet/bin/activate"

export PYTHONUNBUFFERED=1
export HF_HOME="$PROJECT_DIR/.cache/huggingface"

SEEDS=(42 43 44)
SEED_INDEX=$((SLURM_ARRAY_TASK_ID % 3))
ORDER_INDEX=$((SLURM_ARRAY_TASK_ID / 3))
SEED="${SEEDS[$SEED_INDEX]}"

mapfile -t ORDERS < experiments/adaptive/orders.txt
ORDER="${ORDERS[$ORDER_INDEX]}"
IFS='|' read -r NAME BLOCK1 BLOCK2 BLOCK3 BLOCK4 <<< "$ORDER"

mkdir -p results/adaptive logs "$HF_HOME"

echo "experiment=$NAME seed=$SEED order=[$BLOCK1,$BLOCK2,$BLOCK3,$BLOCK4]"

python3 main_adaptive.py \
    --name "$NAME" \
    --order "$BLOCK1" "$BLOCK2" "$BLOCK3" "$BLOCK4" \
    --baseline-dir "results/profiles/baseline/seed_$SEED" \
    --output-dir results/adaptive \
    --dataset "ilee0022/Caltech-256" \
    --seed "$SEED" \
    --batch-size 32 \
    --workers 8 \
    --levels 0.25 0.40 0.55 0.70 0.85 \
    --epochs-per-level 4 \
    --consolidation-epochs 10 \
    --clean-drop-limit 0.10 \
    --controller-examples 512 \
    --controller-epsilon 4 \
    --no-improvement-patience 2 \
    --min-transfer-improvement 0.002 \
    --learning-rate 0.0001 \
    --weight-decay 0.0001 \
    --attack-epsilons 2 4 8 \
    --pgd-steps 10

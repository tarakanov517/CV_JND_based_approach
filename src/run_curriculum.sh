#!/bin/bash
#SBATCH --job-name=cornet-curriculum
#SBATCH --partition=rocky
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --time=1-00:00:00
#SBATCH --array=0-20%5
#SBATCH --output=logs/curriculum_%A_%a.out
#SBATCH --mail-user=arilmusaev@edu.hse.ru
#SBATCH --mail-type=END,FAIL

set -euo pipefail

PROJECT_DIR="$SLURM_SUBMIT_DIR"
cd "$PROJECT_DIR"

source "$HOME/venvs/cornet/bin/activate"

export PYTHONUNBUFFERED=1
export HF_HOME="$PROJECT_DIR/.cache/huggingface"

mapfile -t EXPERIMENTS < curriculum_configs.txt
SEEDS=(42 43 44)
SEED_COUNT="${#SEEDS[@]}"
CONFIG_INDEX=$((SLURM_ARRAY_TASK_ID / SEED_COUNT))
SEED_INDEX=$((SLURM_ARRAY_TASK_ID % SEED_COUNT))
SEED="${SEEDS[$SEED_INDEX]}"

IFS='|' read -r NAME SCHEDULE LEVEL SIGMA_AXON SIGMA_DENDRITE <<< "${EXPERIMENTS[$CONFIG_INDEX]}"

mkdir -p curriculum_results logs "$HF_HOME"

echo "experiment=$NAME schedule=$SCHEDULE seed=$SEED task=$SLURM_ARRAY_TASK_ID"

python3 main.py \
    --name "$NAME" \
    --sweep-kind "$SCHEDULE" \
    --sweep-level "$LEVEL" \
    --seed "$SEED" \
    --weights weights.pth \
    --output-dir curriculum_results \
    --dataset "ilee0022/Caltech-256" \
    --batch-size 32 \
    --workers 8 \
    --head-epochs 10 \
    --full-epochs 70 \
    --head-lr 0.001 \
    --full-lr 0.0001 \
    --sigma-axon "$SIGMA_AXON" \
    --sigma-dendrite "$SIGMA_DENDRITE" \
    --noise-schedule "$SCHEDULE" \
    --noise-warmup-epochs 10 \
    --noise-ramp-epochs 30

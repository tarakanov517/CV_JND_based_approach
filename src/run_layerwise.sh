#!/bin/bash
#SBATCH --job-name=cornet-layerwise
#SBATCH --partition=rocky
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --time=1-00:00:00
#SBATCH --array=0-32%5
#SBATCH --output=logs/layerwise_%A_%a.out
#SBATCH --mail-user=arilmusaev@edu.hse.ru
#SBATCH --mail-type=END,FAIL

set -euo pipefail

PROJECT_DIR="$SLURM_SUBMIT_DIR"
cd "$PROJECT_DIR"

source "$HOME/venvs/cornet/bin/activate"

export PYTHONUNBUFFERED=1
export HF_HOME="$PROJECT_DIR/.cache/huggingface"

mapfile -t EXPERIMENTS < layerwise_configs.txt
SEEDS=(42 43 44)
SEED_COUNT="${#SEEDS[@]}"
CONFIG_INDEX=$((SLURM_ARRAY_TASK_ID / SEED_COUNT))
SEED_INDEX=$((SLURM_ARRAY_TASK_ID % SEED_COUNT))
SEED="${SEEDS[$SEED_INDEX]}"

IFS='|' read -r NAME NOISE_KIND NOISE_BLOCK SIGMA_AXON SIGMA_DENDRITE <<< "${EXPERIMENTS[$CONFIG_INDEX]}"

mkdir -p layerwise_results logs "$HF_HOME"

echo "experiment=$NAME kind=$NOISE_KIND block=$NOISE_BLOCK seed=$SEED"

python3 main_layerwise.py \
    --name "$NAME" \
    --noise-kind "$NOISE_KIND" \
    --noise-block "$NOISE_BLOCK" \
    --seed "$SEED" \
    --weights weights.pth \
    --output-dir layerwise_results \
    --dataset "ilee0022/Caltech-256" \
    --batch-size 32 \
    --workers 8 \
    --head-epochs 50 \
    --full-epochs 100 \
    --head-lr 0.001 \
    --full-lr 0.0001 \
    --sigma-axon "$SIGMA_AXON" \
    --sigma-dendrite "$SIGMA_DENDRITE" \
    --attack-epsilons 2 4 \
    --pgd-steps 10

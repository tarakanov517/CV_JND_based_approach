#!/bin/bash
#SBATCH --job-name=cornet-intensity
#SBATCH --partition=rocky
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --time=1-00:00:00
#SBATCH --array=0-55%5
#SBATCH --output=logs/intensity_%A_%a.out
#SBATCH --mail-user=arilmusaev@edu.hse.ru
#SBATCH --mail-type=END,FAIL

set -euo pipefail

PROJECT_DIR="$SLURM_SUBMIT_DIR"
cd "$PROJECT_DIR"

source "$HOME/venvs/cornet/bin/activate"

export PYTHONUNBUFFERED=1
export HF_HOME="$PROJECT_DIR/.cache/huggingface"

mapfile -t EXPERIMENTS < experiments/intensity/configs.txt
EXPERIMENT="${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}"
IFS='|' read -r NAME NOISE_KIND NOISE_BLOCK SIGMA_AXON SIGMA_DENDRITE <<< "$EXPERIMENT"

mkdir -p results/intensity logs "$HF_HOME"

echo "experiment=$NAME kind=$NOISE_KIND block=$NOISE_BLOCK seed=42 axon=$SIGMA_AXON dendrite=$SIGMA_DENDRITE"

python3 main_layerwise.py \
    --name "$NAME" \
    --noise-kind "$NOISE_KIND" \
    --noise-block "$NOISE_BLOCK" \
    --seed 42 \
    --weights weights.pth \
    --output-dir results/intensity \
    --dataset "ilee0022/Caltech-256" \
    --batch-size 32 \
    --workers 8 \
    --head-epochs 10 \
    --full-epochs 40 \
    --head-lr 0.001 \
    --full-lr 0.0001 \
    --sigma-axon "$SIGMA_AXON" \
    --sigma-dendrite "$SIGMA_DENDRITE" \
    --attack-epsilons 2 4 8 \
    --pgd-steps 10

#!/bin/bash
#SBATCH --job-name=cornet-intensity-transfer
#SBATCH --partition=rocky
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/intensity_transfer_%j.out
#SBATCH --mail-user=arilmusaev@edu.hse.ru
#SBATCH --mail-type=END,FAIL

set -euo pipefail

PROJECT_DIR="$SLURM_SUBMIT_DIR"
cd "$PROJECT_DIR"

source "$HOME/venvs/cornet/bin/activate"

export PYTHONUNBUFFERED=1
export HF_HOME="$PROJECT_DIR/.cache/huggingface"

mkdir -p logs "$HF_HOME"

python3 transfer_layerwise.py \
    --results-dir results/intensity \
    --seed 42 \
    --batch-size 16 \
    --workers 8 \
    --attack-epsilons 2 4 8 \
    --pgd-steps 10

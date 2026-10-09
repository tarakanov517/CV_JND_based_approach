#!/bin/bash
#SBATCH --job-name=cornet-layerwise-transfer
#SBATCH --partition=rocky
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --time=1-00:00:00
#SBATCH --array=0-2%3
#SBATCH --output=logs/layerwise_transfer_%A_%a.out
#SBATCH --mail-user=arilmusaev@edu.hse.ru
#SBATCH --mail-type=END,FAIL

set -euo pipefail

PROJECT_DIR="$SLURM_SUBMIT_DIR"
cd "$PROJECT_DIR"

source "$HOME/venvs/cornet/bin/activate"

export PYTHONUNBUFFERED=1
export HF_HOME="$PROJECT_DIR/.cache/huggingface"

SEEDS=(42 43 44)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"

python3 transfer_layerwise.py \
    --results-dir layerwise_results \
    --seed "$SEED" \
    --batch-size 16 \
    --workers 8 \
    --attack-epsilons 2 4 \
    --pgd-steps 10

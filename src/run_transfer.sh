#!/bin/bash
#SBATCH --job-name=cornet-transfer
#SBATCH --partition=rocky
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --time=08:00:00
#SBATCH --array=0-2%3
#SBATCH --output=logs/transfer_%A_%a.out
#SBATCH --mail-user=arilmusaev@edu.hse.ru
#SBATCH --mail-type=END,FAIL

set -euo pipefail

PROJECT_DIR="$SLURM_SUBMIT_DIR"
cd "$PROJECT_DIR"

if [[ ! -f "$HOME/venvs/cornet/bin/activate" ]]; then
    echo "Не найдено окружение $HOME/venvs/cornet"
    exit 1
fi
source "$HOME/venvs/cornet/bin/activate"

export PYTHONUNBUFFERED=1
export HF_HOME="$PROJECT_DIR/.cache/huggingface"

SEEDS=(42 43 44)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"

python3 transfer_attacks.py \
    --results-dir experiments_cornet \
    --seed "$SEED" \
    --batch-size 16 \
    --workers 8

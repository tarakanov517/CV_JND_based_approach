#!/bin/bash
#SBATCH --job-name=cornet-compare
#SBATCH --partition=rocky
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/comparison_%A_%a.out
set -euo pipefail
source "${CORNET_VENV:-$HOME/venvs/cornet}/bin/activate"
export PYTHONUNBUFFERED=1
cd "$SLURM_SUBMIT_DIR"
python3 experiments/comparison/run.py "$1" --index "${SLURM_ARRAY_TASK_ID:-0}" "${@:2}"

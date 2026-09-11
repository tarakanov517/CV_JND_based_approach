#!/bin/bash
#SBATCH --job-name=cornet-human-noise
#SBATCH --partition=rocky
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --time=1-00:00:00
#SBATCH --array=0-20%3
#SBATCH --output=logs/%A_%a.out
#SBATCH --mail-user=arilmusaev@edu.hse.ru
#SBATCH --mail-type=END,FAIL

set -euo pipefail

PROJECT_DIR="$SLURM_SUBMIT_DIR"
RESULTS_DIR="$PROJECT_DIR/experiments_cornet"
cd "$PROJECT_DIR"

if [[ ! -f "$HOME/venvs/cornet/bin/activate" ]]; then
    echo "Не найдено окружение $HOME/venvs/cornet"
    exit 1
fi
source "$HOME/venvs/cornet/bin/activate"

export PYTHONUNBUFFERED=1
export HF_HOME="$PROJECT_DIR/.cache/huggingface"

python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA недоступна'"

SEEDS=(42 43 44)
EXPERIMENTS=(
    "baseline|0.0|0.0|0.0|0|0.0|1.0|1.0|0.0|0.0"
    "lateral_005|0.05|0.0|0.0|0|0.0|1.0|1.0|0.0|0.0"
    "contrast_p010_a001|0.0|0.10|0.01|0|0.0|1.0|1.0|0.0|0.0"
    "pyramidal_002|0.0|0.0|0.0|1|0.02|1.0|1.0|0.0|0.0"
    "axon_002|0.0|0.0|0.0|0|0.0|1.0|1.0|0.02|0.0"
    "dendrite_002|0.0|0.0|0.0|0|0.0|1.0|1.0|0.0|0.02"
    "combined_low|0.03|0.05|0.005|1|0.01|1.0|1.0|0.01|0.01"
)

EXPERIMENT_COUNT="${#EXPERIMENTS[@]}"
EXPERIMENT_INDEX=$((SLURM_ARRAY_TASK_ID % EXPERIMENT_COUNT))
SEED_INDEX=$((SLURM_ARRAY_TASK_ID / EXPERIMENT_COUNT))
SEED="${SEEDS[$SEED_INDEX]}"

IFS='|' read -r NAME SIGMA_LATERAL SIGMA_PROP SIGMA_ADD PYRAMID SIGMA_PYRAMID GAMMA B SIGMA_AXON SIGMA_DENDRITE <<< "${EXPERIMENTS[$EXPERIMENT_INDEX]}"

mkdir -p "$RESULTS_DIR" "$PROJECT_DIR/logs" "$HF_HOME"

COMMAND=(
    python3 main.py
    --name "$NAME"
    --seed "$SEED"
    --weights weights.pth
    --output-dir experiments_cornet
    --dataset "ilee0022/Caltech-256"
    --batch-size 32
    --workers 8
    --head-epochs 10
    --full-epochs 40
    --head-lr 0.001
    --full-lr 0.0001
    --sigma-lateral "$SIGMA_LATERAL"
    --sigma-prop "$SIGMA_PROP"
    --sigma-add "$SIGMA_ADD"
    --sigma-pyramid "$SIGMA_PYRAMID"
    --gamma "$GAMMA"
    --b "$B"
    --sigma-axon "$SIGMA_AXON"
    --sigma-dendrite "$SIGMA_DENDRITE"
)

if [[ "$PYRAMID" == "1" ]]; then
    COMMAND+=(--pyramid)
fi

echo "experiment=$NAME seed=$SEED task=$SLURM_ARRAY_TASK_ID"
"${COMMAND[@]}"

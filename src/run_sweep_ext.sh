#!/bin/bash
#SBATCH --job-name=cornet-noise-sweep-ext
#SBATCH --partition=rocky
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --time=1-00:00:00
#SBATCH --array=0-44%5
#SBATCH --output=logs/sweep_ext_%A_%a.out
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

python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA недоступна'"

SEEDS=(42 43 44)
EXPERIMENTS=(
    "lateral|0.16|lateral_016|0.16|0.0|0.0|0|0.0|0.0|0.0"
    "lateral|0.24|lateral_024|0.24|0.0|0.0|0|0.0|0.0|0.0"
    "lateral|0.32|lateral_032|0.32|0.0|0.0|0|0.0|0.0|0.0"
    "contrast|3.00|contrast_k300|0.0|0.300|0.030|0|0.0|0.0|0.0"
    "contrast|4.00|contrast_k400|0.0|0.400|0.040|0|0.0|0.0|0.0"
    "contrast|6.00|contrast_k600|0.0|0.600|0.060|0|0.0|0.0|0.0"
    "pyramidal|0.12|pyramidal_012|0.0|0.0|0.0|1|0.12|0.0|0.0"
    "pyramidal|0.16|pyramidal_016|0.0|0.0|0.0|1|0.16|0.0|0.0"
    "pyramidal|0.24|pyramidal_024|0.0|0.0|0.0|1|0.24|0.0|0.0"
    "axon|0.12|axon_012|0.0|0.0|0.0|0|0.0|0.12|0.0"
    "axon|0.20|axon_020|0.0|0.0|0.0|0|0.0|0.20|0.0"
    "axon|0.30|axon_030|0.0|0.0|0.0|0|0.0|0.30|0.0"
    "dendrite|0.12|dendrite_012|0.0|0.0|0.0|0|0.0|0.0|0.12"
    "dendrite|0.20|dendrite_020|0.0|0.0|0.0|0|0.0|0.0|0.20"
    "dendrite|0.30|dendrite_030|0.0|0.0|0.0|0|0.0|0.0|0.30"
)

EXPERIMENT_COUNT="${#EXPERIMENTS[@]}"
EXPERIMENT_INDEX=$((SLURM_ARRAY_TASK_ID % EXPERIMENT_COUNT))
SEED_INDEX=$((SLURM_ARRAY_TASK_ID / EXPERIMENT_COUNT))
SEED="${SEEDS[$SEED_INDEX]}"

IFS='|' read -r KIND LEVEL NAME SIGMA_LATERAL SIGMA_PROP SIGMA_ADD PYRAMID SIGMA_PYRAMID SIGMA_AXON SIGMA_DENDRITE <<< "${EXPERIMENTS[$EXPERIMENT_INDEX]}"

mkdir -p sweep_cornet logs "$HF_HOME"

COMMAND=(
    python3 main.py
    --name "$NAME"
    --sweep-kind "$KIND"
    --sweep-level "$LEVEL"
    --seed "$SEED"
    --weights weights.pth
    --output-dir sweep_cornet
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
    --gamma 1.0
    --b 1.0
    --sigma-axon "$SIGMA_AXON"
    --sigma-dendrite "$SIGMA_DENDRITE"
)

if [[ "$PYRAMID" == "1" ]]; then
    COMMAND+=(--pyramid)
fi

echo "kind=$KIND level=$LEVEL experiment=$NAME seed=$SEED task=$SLURM_ARRAY_TASK_ID"
"${COMMAND[@]}"

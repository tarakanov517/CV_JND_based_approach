#!/bin/bash
#SBATCH --job-name=cornet-noise-sweep
#SBATCH --partition=rocky
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --time=1-00:00:00
#SBATCH --array=0-80%5
#SBATCH --output=logs/sweep_%A_%a.out
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
    "baseline|0.0|baseline|0.0|0.0|0.0|0|0.0|0.0|0.0"
    "lateral|0.01|lateral_001|0.01|0.0|0.0|0|0.0|0.0|0.0"
    "lateral|0.03|lateral_003|0.03|0.0|0.0|0|0.0|0.0|0.0"
    "lateral|0.05|lateral_005|0.05|0.0|0.0|0|0.0|0.0|0.0"
    "lateral|0.08|lateral_008|0.08|0.0|0.0|0|0.0|0.0|0.0"
    "lateral|0.12|lateral_012|0.12|0.0|0.0|0|0.0|0.0|0.0"
    "contrast|0.25|contrast_k025|0.0|0.025|0.0025|0|0.0|0.0|0.0"
    "contrast|0.50|contrast_k050|0.0|0.050|0.0050|0|0.0|0.0|0.0"
    "contrast|1.00|contrast_k100|0.0|0.100|0.0100|0|0.0|0.0|0.0"
    "contrast|1.50|contrast_k150|0.0|0.150|0.0150|0|0.0|0.0|0.0"
    "contrast|2.00|contrast_k200|0.0|0.200|0.0200|0|0.0|0.0|0.0"
    "pyramidal|0.000|pyramidal_000|0.0|0.0|0.0|1|0.000|0.0|0.0"
    "pyramidal|0.005|pyramidal_0005|0.0|0.0|0.0|1|0.005|0.0|0.0"
    "pyramidal|0.010|pyramidal_001|0.0|0.0|0.0|1|0.010|0.0|0.0"
    "pyramidal|0.020|pyramidal_002|0.0|0.0|0.0|1|0.020|0.0|0.0"
    "pyramidal|0.040|pyramidal_004|0.0|0.0|0.0|1|0.040|0.0|0.0"
    "pyramidal|0.080|pyramidal_008|0.0|0.0|0.0|1|0.080|0.0|0.0"
    "axon|0.005|axon_0005|0.0|0.0|0.0|0|0.0|0.005|0.0"
    "axon|0.010|axon_001|0.0|0.0|0.0|0|0.0|0.010|0.0"
    "axon|0.020|axon_002|0.0|0.0|0.0|0|0.0|0.020|0.0"
    "axon|0.040|axon_004|0.0|0.0|0.0|0|0.0|0.040|0.0"
    "axon|0.080|axon_008|0.0|0.0|0.0|0|0.0|0.080|0.0"
    "dendrite|0.005|dendrite_0005|0.0|0.0|0.0|0|0.0|0.0|0.005"
    "dendrite|0.010|dendrite_001|0.0|0.0|0.0|0|0.0|0.0|0.010"
    "dendrite|0.020|dendrite_002|0.0|0.0|0.0|0|0.0|0.0|0.020"
    "dendrite|0.040|dendrite_004|0.0|0.0|0.0|0|0.0|0.0|0.040"
    "dendrite|0.080|dendrite_008|0.0|0.0|0.0|0|0.0|0.0|0.080"
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

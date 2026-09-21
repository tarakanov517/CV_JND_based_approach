#!/bin/bash
#SBATCH --job-name=cornet-sweep-fast
#SBATCH --partition=rocky
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --time=1-00:00:00
#SBATCH --array=0-41%5
#SBATCH --output=logs/sweep_fast_%A_%a.out
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

mapfile -t EXPERIMENTS < sweep_configs.txt
SEED=42

IFS='|' read -r KIND LEVEL NAME SIGMA_LATERAL SIGMA_PROP SIGMA_ADD PYRAMID SIGMA_PYRAMID SIGMA_AXON SIGMA_DENDRITE <<< "${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}"

mkdir -p sweep_fast logs "$HF_HOME"

COMMAND=(
    python3 main.py
    --name "$NAME"
    --sweep-kind "$KIND"
    --sweep-level "$LEVEL"
    --seed "$SEED"
    --weights weights.pth
    --output-dir sweep_fast
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

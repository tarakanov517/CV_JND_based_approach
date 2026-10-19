#!/bin/bash
#SBATCH --job-name=cornet-profiles
#SBATCH --partition=rocky
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --time=1-00:00:00
#SBATCH --array=0-65%5
#SBATCH --output=logs/profiles_%A_%a.out
#SBATCH --mail-user=arilmusaev@edu.hse.ru
#SBATCH --mail-type=END,FAIL

set -euo pipefail

PROJECT_DIR="$SLURM_SUBMIT_DIR"
cd "$PROJECT_DIR"

source "$HOME/venvs/cornet/bin/activate"

export PYTHONUNBUFFERED=1
export HF_HOME="$PROJECT_DIR/.cache/huggingface"

SEEDS=(42 43 44)
SEED_INDEX=$((SLURM_ARRAY_TASK_ID % 3))
CONFIG_INDEX=$((SLURM_ARRAY_TASK_ID / 3))
SEED="${SEEDS[$SEED_INDEX]}"

mapfile -t EXPERIMENTS < experiments/profiles/configs.txt
EXPERIMENT="${EXPERIMENTS[$CONFIG_INDEX]}"
IFS='|' read -r NAME FAMILY V1 V2 V4 IT <<< "$EXPERIMENT"

NOISE_KIND="axon"
NOISE_BLOCK="profile"
if [[ "$NAME" == "baseline" ]]; then
    NOISE_KIND="baseline"
    NOISE_BLOCK="none"
fi

mkdir -p results/profiles logs "$HF_HOME"

echo "experiment=$NAME family=$FAMILY seed=$SEED profile=[$V1,$V2,$V4,$IT]"

python3 main_layerwise.py \
    --name "$NAME" \
    --noise-kind "$NOISE_KIND" \
    --noise-block "$NOISE_BLOCK" \
    --profile-family "$FAMILY" \
    --axon-profile "$V1" "$V2" "$V4" "$IT" \
    --seed "$SEED" \
    --weights weights.pth \
    --output-dir results/profiles \
    --dataset "ilee0022/Caltech-256" \
    --batch-size 32 \
    --workers 8 \
    --head-epochs 10 \
    --full-epochs 40 \
    --head-lr 0.001 \
    --full-lr 0.0001 \
    --attack-epsilons 2 4 8 \
    --pgd-steps 10

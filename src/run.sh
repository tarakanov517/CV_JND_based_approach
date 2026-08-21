#!/bin/bash
#SBATCH --job-name=human-noise-seeds
#SBATCH --partition=rocky
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --time=1-00:00:00
#SBATCH --array=0-4
#SBATCH --output=slurm-train-%A_%a.out
#SBATCH --mail-user=arilmusaev@edu.hse.ru
#SBATCH --mail-type=END,FAIL

set -euo pipefail

PROJECT_DIR="$SLURM_SUBMIT_DIR"
SEEDS=(42 43 44 45 46)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
RUN_DIR="$PROJECT_DIR/human_noise_seeds/seed_$SEED"

module load Python/PyTorch_GPU_v2.4

if [[ -f "$HOME/venvs/gaussian/bin/activate" ]]; then
    source "$HOME/venvs/gaussian/bin/activate"
fi

export PYTHONUNBUFFERED=1

python3 -c "import torch; import torchvision; import torchattacks; import datasets"

mkdir -p "$RUN_DIR/models" "$RUN_DIR/data" "$RUN_DIR/experiments"
cd "$RUN_DIR"

EPOCHS=40
LEARNING_RATE=0.01
MOMENTUM=0.9
TRAIN_BATCH_SIZE=32
TEST_BATCH_SIZE=32
DATASET="uoft-cs/cifar10"
TOTAL_EXPERIMENTS=3
CURRENT_EXPERIMENT=0

run_case() {
    local label="$1"
    shift

    CURRENT_EXPERIMENT=$((CURRENT_EXPERIMENT + 1))

    echo
    echo "============================================================"
    echo "Эксперимент $CURRENT_EXPERIMENT/$TOTAL_EXPERIMENTS"
    echo "Название: $label"
    echo "Seed: $SEED"
    echo "Рабочая папка: $RUN_DIR"
    echo "============================================================"

    srun python3 "$PROJECT_DIR/main.py" \
        --epochs "$EPOCHS" \
        --learning_rate "$LEARNING_RATE" \
        --momentum "$MOMENTUM" \
        --train_batch_size "$TRAIN_BATCH_SIZE" \
        --test_batch_size "$TEST_BATCH_SIZE" \
        --dataset_name "$DATASET" \
        --seed "$SEED" \
        --experiment_name "$label" \
        --pyramidal_gamma 1.0 \
        --pyramidal_b 1.0 \
        "$@"
}

run_case "lateral_high" \
    --sigma_lateral 0.10

run_case "dendrite_block1_low" \
    --dendrite_theta1 0.15 \
    --dendrite_sigma1 0.0005

run_case "all_parameter_high" \
    --omega1 0.004 \
    --omega2 0.002 \
    --omega3 0.006 \
    --dendrite_theta1 0.15 \
    --dendrite_sigma1 0.002 \
    --dendrite_theta2 0.15 \
    --dendrite_sigma2 0.001 \
    --dendrite_theta3 0.15 \
    --dendrite_sigma3 0.003

echo
echo "Seed $SEED завершён"
echo "Результаты: $RUN_DIR/experiments/results.csv"

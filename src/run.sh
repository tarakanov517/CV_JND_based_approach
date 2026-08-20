#!/bin/bash
#SBATCH --job-name=gaussian-seeds
#SBATCH --partition=rocky
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --time=1-00:00:00
#SBATCH --array=0-4
#SBATCH --output=slurm-%A_%a.out
#SBATCH --mail-user=arilmusaev@edu.hse.ru
#SBATCH --mail-type=END,FAIL

set -euo pipefail

PROJECT_DIR="$SLURM_SUBMIT_DIR"

SEEDS=(42 43 44 45 46)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"

RUN_DIR="$PROJECT_DIR/check_seeds/seed_$SEED"

module load Python/PyTorch_GPU_v2.4

if [[ -f "$HOME/venvs/gaussian/bin/activate" ]]; then
    source "$HOME/venvs/gaussian/bin/activate"
fi

export PYTHONUNBUFFERED=1

python3 -c "import torch; import torchattacks"

mkdir -p "$RUN_DIR/models"
mkdir -p "$RUN_DIR/data"
mkdir -p "$RUN_DIR/experiments"

cd "$RUN_DIR"
echo "Текущий seed: $SEED"
echo "Рабочая папка: $RUN_DIR"

EPOCHS=40
LEARNING_RATE=0.01
MOMENTUM=0.9
TRAIN_BATCH_SIZE=32
TEST_BATCH_SIZE=32
DATASET="uoft-cs/cifar10"

TOTAL_EXPERIMENTS=3
CURRENT_EXPERIMENT=0

is_completed() {
    local model_name="$1"
    local results_file="$RUN_DIR/experiments/results.csv"

    [[ -f "$results_file" ]] && awk -F',' -v name="$model_name" '
        $1 == name { found = 1 }
        END { exit(found ? 0 : 1) }
    ' "$results_file"
}

run_experiment() {
    local name="$1"
    local sigma1="$2"
    local sigma2="$3"
    local sigma3="$4"
    local omega1="$5"
    local omega2="$6"
    local omega3="$7"
    local model_name="Model_${sigma1}_${sigma2}_${sigma3}_${omega1}_${omega2}_${omega3}"

    CURRENT_EXPERIMENT=$((CURRENT_EXPERIMENT + 1))

    if is_completed "$model_name"; then
        echo "Пропуск завершённого эксперимента: $model_name"
        return
    fi

    echo
    echo "============================================================"
    echo "Эксперимент $CURRENT_EXPERIMENT/$TOTAL_EXPERIMENTS"
    echo "Название: $name"
    echo "Seed: $SEED"
    echo "Sigma: $sigma1 $sigma2 $sigma3"
    echo "Omega: $omega1 $omega2 $omega3"
    echo "============================================================"

    srun python3 "$PROJECT_DIR/main.py" \
        --epochs "$EPOCHS" \
        --learning_rate "$LEARNING_RATE" \
        --momentum "$MOMENTUM" \
        --train_batch_size "$TRAIN_BATCH_SIZE" \
        --test_batch_size "$TEST_BATCH_SIZE" \
        --dataset_name "$DATASET" \
        --sigma1 "$sigma1" \
        --sigma2 "$sigma2" \
        --sigma3 "$sigma3" \
        --omega1 "$omega1" \
        --omega2 "$omega2" \
        --omega3 "$omega3" \
        --noise_schedule fixed \
        --seed "$SEED"
}

run_experiment "activation_block1" \
    0.15 0.0 0.0 \
    0.0 0.0 0.0

run_experiment "parameters_block1" \
    0.0 0.0 0.0 \
    0.002 0.0 0.0

run_experiment "parameters_blocks" \
    0.0 0.0 0.0 \
    0.002 0.001 0.003

echo
echo "Seed $SEED завершён"
echo "Результаты: $RUN_DIR/experiments/results.csv"

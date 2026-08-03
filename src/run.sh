#!/bin/bash
#SBATCH --job-name=gaussian-curriculum-100
#SBATCH --partition=rocky
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --time=1-00:00:00
#SBATCH --mail-user=arilmusaev@edu.hse.ru
#SBATCH --mail-type=END,FAIL

set -euo pipefail

PROJECT_DIR="$SLURM_SUBMIT_DIR"
RUN_DIR="$PROJECT_DIR/curriculum_100"

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

EPOCHS=100
LEARNING_RATE=0.01
MOMENTUM=0.9
TRAIN_BATCH_SIZE=32
TEST_BATCH_SIZE=32
DATASET="uoft-cs/cifar10"
WARMUP_EPOCHS=25
RAMP_EPOCHS=25
TOTAL_EXPERIMENTS=24

EXPERIMENTS=(
    "activation_block1|0.15|0.0|0.0|0.0|0.0|0.0"
    "activation_block2|0.0|0.15|0.0|0.0|0.0|0.0"
    "activation_classifier|0.0|0.0|0.04|0.0|0.0|0.0"
    "parameter_block1|0.0|0.0|0.0|0.002|0.0|0.0"
    "parameter_block2|0.0|0.0|0.0|0.0|0.001|0.0"
    "parameter_classifier|0.0|0.0|0.0|0.0|0.0|0.003"
    "all_activation|0.15|0.15|0.04|0.0|0.0|0.0"
    "all_parameter|0.0|0.0|0.0|0.002|0.001|0.003"
    "block1_combined|0.15|0.0|0.0|0.002|0.0|0.0"
    "block2_combined|0.0|0.15|0.0|0.0|0.001|0.0"
    "classifier_combined|0.0|0.0|0.04|0.0|0.0|0.003"
    "full_combined|0.15|0.15|0.04|0.002|0.001|0.003"
)

is_completed() {
    local expected_name="$1"
    local results_file="$RUN_DIR/experiments/results.csv"

    if [[ ! -f "$results_file" ]]; then
        return 1
    fi

    awk -F',' -v name="$expected_name" '
        $1 == name { found = 1 }
        END { exit(found ? 0 : 1) }
    ' "$results_file"
}

run_experiment() {
    local label="$1"
    local schedule="$2"
    local sigma1="$3"
    local sigma2="$4"
    local sigma3="$5"
    local omega1="$6"
    local omega2="$7"
    local omega3="$8"
    local expected_name

    if [[ "$schedule" == "fixed" ]]; then
        expected_name="Model_${sigma1}_${sigma2}_${sigma3}_${omega1}_${omega2}_${omega3}"
    else
        expected_name="Model_${EPOCHS}_${schedule}_w${WARMUP_EPOCHS}_r${RAMP_EPOCHS}_${sigma1}_${sigma2}_${sigma3}_${omega1}_${omega2}_${omega3}"
    fi

    if is_completed "$expected_name"; then
        echo "Пропуск завершённого эксперимента: $expected_name"
        return
    fi

    echo
    echo "============================================================"
    echo "Experiment $CURRENT_EXPERIMENT/$TOTAL_EXPERIMENTS"
    echo "label=$label schedule=$schedule"
    echo "sigma=($sigma1, $sigma2, $sigma3)"
    echo "omega=($omega1, $omega2, $omega3)"
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
        --noise_schedule "$schedule" \
        --warmup_epochs "$WARMUP_EPOCHS" \
        --ramp_epochs "$RAMP_EPOCHS"
}

CURRENT_EXPERIMENT=0

for experiment in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r label sigma1 sigma2 sigma3 omega1 omega2 omega3 <<< "$experiment"

    for schedule in fixed linear; do
        CURRENT_EXPERIMENT=$((CURRENT_EXPERIMENT + 1))

        run_experiment \
            "$label" \
            "$schedule" \
            "$sigma1" \
            "$sigma2" \
            "$sigma3" \
            "$omega1" \
            "$omega2" \
            "$omega3"
    done
done

echo
echo "============================================================"
echo "Все 24 эксперимента завершены"
echo "Результаты: $RUN_DIR/experiments/results.csv"
echo "============================================================"

#!/bin/bash
#SBATCH --job-name=gaussian-grid
#SBATCH --partition=rocky
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --time=0-04:30:00
#SBATCH --output=slurm-%j.out
#SBATCH --mail-user=arilmusaev@edu.hse.ru
#SBATCH --mail-type=END,FAIL

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

module load Python/PyTorch_GPU_v2.4

export PYTHONUNBUFFERED=1

mkdir -p models
mkdir -p data
mkdir -p experiments

if [[ -f experiments/results.csv ]]; then
    echo "Ошибка: experiments/results.csv уже существует."
    echo "Для нового полного расчёта перемести или удали старый файл."
    exit 1
fi

rm -f data/adv_data_fgsm.pt
rm -f data/adv_data_pgd.pt

EPOCHS=20
LEARNING_RATE=0.01
MOMENTUM=0.9
TRAIN_BATCH_SIZE=256
TEST_BATCH_SIZE=512
DATASET="uoft-cs/cifar10"

ACTIVATION_SIGMAS=(
    0.0025
    0.005
    0.01
    0.015
    0.02
    0.03
    0.04
    0.05
    0.07
    0.1
    0.15
)

PARAMETER_OMEGAS=(
    0.00001
    0.00003
    0.0001
    0.0002
    0.0003
    0.0005
    0.001
    0.002
    0.003
    0.005
    0.01
)

run_experiment() {
    local sigma1="$1"
    local sigma2="$2"
    local sigma3="$3"
    local omega1="$4"
    local omega2="$5"
    local omega3="$6"

    echo
    echo "============================================================"
    echo "sigma1=$sigma1 sigma2=$sigma2 sigma3=$sigma3"
    echo "omega1=$omega1 omega2=$omega2 omega3=$omega3"
    echo "============================================================"

    srun python3 main.py \
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
        --omega3 "$omega3"
}

experiment_number=0
total_experiments=66

for sigma in "${ACTIVATION_SIGMAS[@]}"; do
    experiment_number=$((experiment_number + 1))
    echo "Experiment $experiment_number/$total_experiments"
    run_experiment "$sigma" 0 0 0 0 0

    experiment_number=$((experiment_number + 1))
    echo "Experiment $experiment_number/$total_experiments"
    run_experiment 0 "$sigma" 0 0 0 0

    experiment_number=$((experiment_number + 1))
    echo "Experiment $experiment_number/$total_experiments"
    run_experiment 0 0 "$sigma" 0 0 0
done

for omega in "${PARAMETER_OMEGAS[@]}"; do
    experiment_number=$((experiment_number + 1))
    echo "Experiment $experiment_number/$total_experiments"
    run_experiment 0 0 0 "$omega" 0 0

    experiment_number=$((experiment_number + 1))
    echo "Experiment $experiment_number/$total_experiments"
    run_experiment 0 0 0 0 "$omega" 0

    experiment_number=$((experiment_number + 1))
    echo "Experiment $experiment_number/$total_experiments"
    run_experiment 0 0 0 0 0 "$omega"
done

echo
echo "============================================================"
echo "Все эксперименты завершены"
echo "Результаты: $SLURM_SUBMIT_DIR/experiments/results.csv"
echo "============================================================"
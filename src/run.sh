#!/bin/bash
#SBATCH --job-name=human-noise-grid
#SBATCH --partition=rocky
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --time=1-00:00:00
#SBATCH --mail-user=arilmusaev@edu.hse.ru
#SBATCH --mail-type=END,FAIL

set -euo pipefail

PROJECT_DIR="$SLURM_SUBMIT_DIR"
RUN_DIR="$PROJECT_DIR/human_noises_grid_40"

module load Python/PyTorch_GPU_v2.4

if [[ -f "$HOME/venvs/gaussian/bin/activate" ]]; then
    source "$HOME/venvs/gaussian/bin/activate"
fi

export PYTHONUNBUFFERED=1

python3 -c "import torch; import torchvision; import torchattacks; import datasets"

mkdir -p "$RUN_DIR/models"
mkdir -p "$RUN_DIR/data"
mkdir -p "$RUN_DIR/experiments"

cd "$RUN_DIR"

EPOCHS=40
LEARNING_RATE=0.01
MOMENTUM=0.9
TRAIN_BATCH_SIZE=32
TEST_BATCH_SIZE=32
DATASET="uoft-cs/cifar10"
TOTAL_EXPERIMENTS=43
CURRENT_EXPERIMENT=0

run_case() {
    local label="$1"
    local seed="$2"
    shift 2

    CURRENT_EXPERIMENT=$((CURRENT_EXPERIMENT + 1))

    echo
    echo "============================================================"
    echo "Experiment $CURRENT_EXPERIMENT/$TOTAL_EXPERIMENTS: $label"
    echo "Seed: $seed"
    echo "============================================================"

    srun python3 "$PROJECT_DIR/main.py" \
        --epochs "$EPOCHS" \
        --learning_rate "$LEARNING_RATE" \
        --momentum "$MOMENTUM" \
        --train_batch_size "$TRAIN_BATCH_SIZE" \
        --test_batch_size "$TEST_BATCH_SIZE" \
        --dataset_name "$DATASET" \
        --seed "$seed" \
        --experiment_name "$label" \
        --pyramidal_gamma 1.0 \
        --pyramidal_b 1.0 \
        "$@"
}

run_case "zero_noise_control" 43

run_case "lateral_low" 42 --sigma_lateral 0.02
run_case "lateral_base" 42 --sigma_lateral 0.05
run_case "lateral_high" 42 --sigma_lateral 0.10

run_case "contrast_low" 42 --sigma_prop 0.02 --sigma_add 0.005
run_case "contrast_base" 42 --sigma_prop 0.05 --sigma_add 0.01
run_case "contrast_high" 42 --sigma_prop 0.10 --sigma_add 0.02

run_case "pyramidal_low" 42 --pyramidal_sigma 0.01
run_case "pyramidal_base" 42 --pyramidal_sigma 0.02
run_case "pyramidal_high" 42 --pyramidal_sigma 0.05

run_case "axon_block1_low" 42 --omega1 0.001
run_case "axon_block1_base" 42 --omega1 0.002
run_case "axon_block1_high" 42 --omega1 0.004

run_case "axon_block2_low" 42 --omega2 0.0005
run_case "axon_block2_base" 42 --omega2 0.001
run_case "axon_block2_high" 42 --omega2 0.002

run_case "axon_classifier_low" 42 --omega3 0.0015
run_case "axon_classifier_base" 42 --omega3 0.003
run_case "axon_classifier_high" 42 --omega3 0.006

run_case "dendrite_block1_low" 42 --dendrite_theta1 0.15 --dendrite_sigma1 0.0005
run_case "dendrite_block1_base" 42 --dendrite_theta1 0.15 --dendrite_sigma1 0.001
run_case "dendrite_block1_high" 42 --dendrite_theta1 0.15 --dendrite_sigma1 0.002

run_case "dendrite_block2_low" 42 --dendrite_theta2 0.15 --dendrite_sigma2 0.00025
run_case "dendrite_block2_base" 42 --dendrite_theta2 0.15 --dendrite_sigma2 0.0005
run_case "dendrite_block2_high" 42 --dendrite_theta2 0.15 --dendrite_sigma2 0.001

run_case "dendrite_classifier_low" 42 --dendrite_theta3 0.15 --dendrite_sigma3 0.00075
run_case "dendrite_classifier_base" 42 --dendrite_theta3 0.15 --dendrite_sigma3 0.0015
run_case "dendrite_classifier_high" 42 --dendrite_theta3 0.15 --dendrite_sigma3 0.003

run_case "all_activation_low" 42 \
    --sigma_lateral 0.02 --sigma_prop 0.02 --sigma_add 0.005 --pyramidal_sigma 0.01
run_case "all_activation_base" 42 \
    --sigma_lateral 0.05 --sigma_prop 0.05 --sigma_add 0.01 --pyramidal_sigma 0.02
run_case "all_activation_high" 42 \
    --sigma_lateral 0.10 --sigma_prop 0.10 --sigma_add 0.02 --pyramidal_sigma 0.05

run_case "all_axon_low" 42 --omega1 0.001 --omega2 0.0005 --omega3 0.0015
run_case "all_axon_base" 42 --omega1 0.002 --omega2 0.001 --omega3 0.003
run_case "all_axon_high" 42 --omega1 0.004 --omega2 0.002 --omega3 0.006

run_case "all_dendrite_low" 42 \
    --dendrite_theta1 0.15 --dendrite_sigma1 0.0005 \
    --dendrite_theta2 0.15 --dendrite_sigma2 0.00025 \
    --dendrite_theta3 0.15 --dendrite_sigma3 0.00075
run_case "all_dendrite_base" 42 \
    --dendrite_theta1 0.15 --dendrite_sigma1 0.001 \
    --dendrite_theta2 0.15 --dendrite_sigma2 0.0005 \
    --dendrite_theta3 0.15 --dendrite_sigma3 0.0015
run_case "all_dendrite_high" 42 \
    --dendrite_theta1 0.15 --dendrite_sigma1 0.002 \
    --dendrite_theta2 0.15 --dendrite_sigma2 0.001 \
    --dendrite_theta3 0.15 --dendrite_sigma3 0.003

run_case "all_parameter_low" 42 \
    --omega1 0.001 --omega2 0.0005 --omega3 0.0015 \
    --dendrite_theta1 0.15 --dendrite_sigma1 0.0005 \
    --dendrite_theta2 0.15 --dendrite_sigma2 0.00025 \
    --dendrite_theta3 0.15 --dendrite_sigma3 0.00075
run_case "all_parameter_base" 42 \
    --omega1 0.002 --omega2 0.001 --omega3 0.003 \
    --dendrite_theta1 0.15 --dendrite_sigma1 0.001 \
    --dendrite_theta2 0.15 --dendrite_sigma2 0.0005 \
    --dendrite_theta3 0.15 --dendrite_sigma3 0.0015
run_case "all_parameter_high" 42 \
    --omega1 0.004 --omega2 0.002 --omega3 0.006 \
    --dendrite_theta1 0.15 --dendrite_sigma1 0.002 \
    --dendrite_theta2 0.15 --dendrite_sigma2 0.001 \
    --dendrite_theta3 0.15 --dendrite_sigma3 0.003

run_case "full_combined_low" 42 \
    --sigma_lateral 0.02 --sigma_prop 0.02 --sigma_add 0.005 --pyramidal_sigma 0.01 \
    --omega1 0.001 --omega2 0.0005 --omega3 0.0015 \
    --dendrite_theta1 0.15 --dendrite_sigma1 0.0005 \
    --dendrite_theta2 0.15 --dendrite_sigma2 0.00025 \
    --dendrite_theta3 0.15 --dendrite_sigma3 0.00075
run_case "full_combined_base" 42 \
    --sigma_lateral 0.05 --sigma_prop 0.05 --sigma_add 0.01 --pyramidal_sigma 0.02 \
    --omega1 0.002 --omega2 0.001 --omega3 0.003 \
    --dendrite_theta1 0.15 --dendrite_sigma1 0.001 \
    --dendrite_theta2 0.15 --dendrite_sigma2 0.0005 \
    --dendrite_theta3 0.15 --dendrite_sigma3 0.0015
run_case "full_combined_high" 42 \
    --sigma_lateral 0.10 --sigma_prop 0.10 --sigma_add 0.02 --pyramidal_sigma 0.05 \
    --omega1 0.004 --omega2 0.002 --omega3 0.006 \
    --dendrite_theta1 0.15 --dendrite_sigma1 0.002 \
    --dendrite_theta2 0.15 --dendrite_sigma2 0.001 \
    --dendrite_theta3 0.15 --dendrite_sigma3 0.003

echo
echo "============================================================"
echo "All $TOTAL_EXPERIMENTS experiments completed"
echo "Results: $RUN_DIR/experiments/results.csv"
echo "============================================================"

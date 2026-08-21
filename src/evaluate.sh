#!/bin/bash
#SBATCH --job-name=human-noise-evaluation
#SBATCH --partition=rocky
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-evaluation-%j.out
#SBATCH --mail-user=arilmusaev@edu.hse.ru
#SBATCH --mail-type=END,FAIL

set -euo pipefail

PROJECT_DIR="$SLURM_SUBMIT_DIR"

module load Python/PyTorch_GPU_v2.4

if [[ -f "$HOME/venvs/gaussian/bin/activate" ]]; then
    source "$HOME/venvs/gaussian/bin/activate"
fi

export PYTHONUNBUFFERED=1

python3 -c "import torch; import torchvision; import torchattacks; import datasets"

cd "$PROJECT_DIR"

srun python3 "$PROJECT_DIR/evaluate_seeds.py" \
    --root "$PROJECT_DIR/human_noise_seeds" \
    --dataset "uoft-cs/cifar10" \
    --batch_size 32 \
    --seeds "42,43,44,45,46"

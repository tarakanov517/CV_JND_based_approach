#!/bin/bash
#SBATCH --job-name=human-noises
#SBATCH --gpus=1
#SBATCH --partition=rocky
#SBATCH --cpus-per-task=9
#SBATCH --time=0-4:0
#SBATCH --mail-user=arilmusaev@edu.hse.ru
#SBATCH --mail-type=END,FAIL

module load Python/PyTorch_GPU_v2.4

python3 start.py

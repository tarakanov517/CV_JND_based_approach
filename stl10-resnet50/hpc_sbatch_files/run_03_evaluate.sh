#!/bin/bash
#SBATCH --job-name=eval_models
#SBATCH --output=/home/misavinov/scratch/ws/my_space/stl10-resnet50/logs/eval_models-%j.log
#SBATCH --error=/home/misavinov/scratch/ws/my_space/stl10-resnet50/logs/eval_models-%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=rocky

module purge
module load Python
source activate jnd_env

cd /home/misavinov/scratch/ws/my_space/stl10-resnet50

python scripts/evaluate.py
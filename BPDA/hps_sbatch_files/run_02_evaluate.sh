#!/bin/bash
#SBATCH --job-name=eval_models
#SBATCH --output=/home/misavinov/imsh-vk/jnd-test/BPDA/logs/eval_models-%j.log
#SBATCH --error=/home/misavinov/imsh-vk/jnd-test/BPDA/logs/eval_models-%j.err
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=rocky

module purge
module load Python
source activate jnd_env

cd /scratch/misavinov/BPDA/

python evaluate.py
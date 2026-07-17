#!/bin/bash
#SBATCH --job-name=eval_models
#SBATCH --output=/home/misavinov/imsh-vk/jnd-test/tiny-imagenet-resnet34/logs/eval_models-%j.log
#SBATCH --error=/home/misavinov/imsh-vk/jnd-test/tiny-imagenet-resnet34/logs/eval_models-%j.err
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=rocky

module purge
module load Python
source activate jnd_env

cd /home/misavinov/imsh-vk/jnd-test/tiny-imagenet-resnet34/

python evaluate.py
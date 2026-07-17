#!/bin/bash
#SBATCH --job-name=train_models
#SBATCH --output=/home/misavinov/imsh-vk/jnd-test/tiny-imagenet-resnet34/logs/train_models-%A_task%a.log
#SBATCH --error=/home/misavinov/imsh-vk/jnd-test/tiny-imagenet-resnet34/logs/train_models-%A_task%a.err
#SBATCH --array=0-2
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=rocky

module purge
module load Python
source activate jnd_env

cd /home/misavinov/imsh-vk/jnd-test/tiny-imagenet-resnet34/

LAYERS=('rgb' 'kkk' 'xzk')

CURRENT_LAYER=${LAYERS[$SLURM_ARRAY_TASK_ID]}   

echo "Запускаем обучение для слоя: $CURRENT_LAYER"

python train.py --layer $CURRENT_LAYER
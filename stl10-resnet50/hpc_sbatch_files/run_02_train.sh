#!/bin/bash

#SBATCH --job-name=train_models
#SBATCH --output=/home/misavinov/scratch/ws/my_space/stl10-resnet50/logs/train_models-%A_task%a.log
#SBATCH --error=/home/misavinov/scratch/ws/my_space/stl10-resnet50/logs/train_models-%A_task%a.err
#SBATCH --array=0-8
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=rocky

module purge
module load Python
source activate jnd_env

cd /home/misavinov/scratch/ws/my_space/stl10-resnet50

CONFIGS=(
  "rgb None"
  "kkk None"
  "kkk 4"
  "kkk 8"
  "kkk 16"
  "xzk None"
  "xzk 4"
  "xzk 8"
  "xzk 16"
)

TASK=(${CONFIGS[$SLURM_ARRAY_TASK_ID]})
CURRENT_LAYER=${TASK[0]}
CURRENT_WIDTH=${TASK[1]}

echo "Запускаем обучение для слоя: $CURRENT_LAYER и размера: $CURRENT_WIDTH"

python scripts/train.py --layer "$CURRENT_LAYER" --target_width_cm "$CURRENT_WIDTH"
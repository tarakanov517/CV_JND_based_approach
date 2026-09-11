#!/bin/bash

#SBATCH --job-name=prepare_data
#SBATCH --output=/home/misavinov/scratch/ws/my_space/stl10-resnet50/logs/prep_data-%j.log
#SBATCH --error=/home/misavinov/scratch/ws/my_space/stl10-resnet50/logs/prep_data-%j.err
#SBATCH --time=4:00:00          
#SBATCH --cpus-per-task=8        
#SBATCH --nodes=1                
#SBATCH --partition=rocky

module purge
module load Python
source activate jnd_env

cd /home/misavinov/scratch/ws/my_space/stl10-resnet50

echo "Начинаем генерацию датасетов..."
python scripts/prepare_data.py
echo "Генерация завершена."
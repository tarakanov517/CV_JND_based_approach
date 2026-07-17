#!/bin/bash

#SBATCH --job-name=prep_data
#SBATCH --output=/home/misavinov/imsh-vk/jnd-test/tiny-imagenet-resnet34/logs/prep_data-%j.log
#SBATCH --error=/home/misavinov/imsh-vk/jnd-test/tiny-imagenet-resnet34/logs/prep_data-%j.err
#SBATCH --time=4:00:00          
#SBATCH --cpus-per-task=8        
#SBATCH --nodes=1                
#SBATCH --partition=rocky

module purge
module load Python
source activate jnd_env

cd /home/misavinov/imsh-vk/jnd-test/tiny-imagenet-resnet34/

echo "Начинаем генерацию датасетов..."
python prepare_data.py
echo "Генерация завершена."
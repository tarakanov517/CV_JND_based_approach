#!/bin/bash

#SBATCH --job-name=prep_data
#SBATCH --output=/home/misavinov/imsh-vk/jnd-test/BPDA/logs/prepare-%j.log
#SBATCH --error=/home/misavinov/imsh-vk/jnd-test/BPDA/logs/prepare-%j.err
#SBATCH --time=1:00:00          
#SBATCH --cpus-per-task=8    
#SBATCH --nodes=1                
#SBATCH --partition=rocky

module purge
module load Python
source activate jnd_env

cd /scratch/misavinov/BPDA/

echo "Начинаем генерацию датасетов..."
python prepare_data.py
echo "Генерация завершена."
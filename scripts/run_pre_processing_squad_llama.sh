#!/bin/bash
  
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=9
#SBATCH --gpus=1
#SBATCH --job-name=squad_llama
#SBATCH --ntasks=1
#SBATCH --time=08:00:00
#SBATCH --mem=32000M
#SBATCH --output=output/squad_llama_%A.out


# Run your code
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dl4nlp
python preprocessing.py --model llama3 --dataset squad --batch_size 16 --seed 42
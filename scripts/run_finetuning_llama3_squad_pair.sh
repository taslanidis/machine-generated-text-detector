#!/bin/bash
  
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=9
#SBATCH --gpus=1
#SBATCH --job-name=ft_squad_llama3
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --mem=32000M
#SBATCH --output=output/ft_squad_llama3_%A.out


# Run your code
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dl4nlp
python train.py data/squad --size m --ai llama3 --seed 42
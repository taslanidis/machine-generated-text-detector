#!/bin/bash
  
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=9
#SBATCH --gpus=1
#SBATCH --job-name=squad_mistral
#SBATCH --ntasks=1
#SBATCH --time=08:00:00
#SBATCH --mem=32000M
#SBATCH --output=output/squad_mistral_%A.out


# Run your code
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dl4nlp
python preprocessing.py --model mistral7b --dataset squad --batch_size 32 --seed 42
#!/bin/bash
  
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=9
#SBATCH --gpus=1
#SBATCH --job-name=eval
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --mem=32000M
#SBATCH --output=output/eval%A.out


# Run your code
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dl4nlp
python eval_all.py
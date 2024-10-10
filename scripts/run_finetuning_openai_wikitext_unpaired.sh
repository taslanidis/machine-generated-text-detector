#!/bin/bash
  
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=9
#SBATCH --gpus=1
#SBATCH --job-name=ft_wiki_openai_unp
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --mem=32000M
#SBATCH --output=output/ft_wiki_openai_unpaired_%A.out


# Run your code
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dl4nlp
python train.py data/wikitext --size m --ai openai --seed 42 --train_unpaired
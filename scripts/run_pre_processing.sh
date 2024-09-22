#!/bin/bash
  
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=9
#SBATCH --gpus=1
#SBATCH --job-name=dataset_llama3
#SBATCH --ntasks=1
#SBATCH --time=00:59:00
#SBATCH --mem=32000M
#SBATCH --output=output/dataset_llama3_%A.out


# Run your code
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dl4nlp
python preprocessing.py --model llama3 --dataset followupqg --batch_size 32 --seed 42
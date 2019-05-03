#!/bin/bash
#SBATCH -A T2-CS056-GPU
#SBATCH -p pascal
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00

source ~/pytorch-env/bin/activate
python compare_imputation.py $1
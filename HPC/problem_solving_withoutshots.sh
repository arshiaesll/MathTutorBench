#!/bin/sh
#SBATCH --job-name=mathtutorbench
#SBATCH -N 1    ## requests on 1 node
#SBATCH --gres=gpu:1   # request 2 GPUs
#SBATCH --time=12:00:00
#SBATCH --output /work/aeslami/VLA_results/ConvAI/job%j.out
#SBATCH --error /work/aeslami/VLA_results/ConvAI/job%j.err
#SBATCH -p gpu-A100 


# Load Anaconda module (adjust the path if necessary)
module load python3/anaconda/3.12


# Activate your conda environment, ## user source activate on cluster, not conda activate
source activate /work/aeslami/ConvAI/mathtutorbench/mathtutor_env

cd /work/aeslami/ConvAI/mathtutorbench

# Add this line to pass all the arguments down to main.py
python main.py --tasks problem_solving_withoutshots.yaml --provider local --model_args model=phi3.5vl


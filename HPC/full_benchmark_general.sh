#!/bin/sh
#SBATCH --job-name=MathTutorBenchmark
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/work/aeslami/results/ConvAI/job%j.out
#SBATCH --error=/work/aeslami/results/ConvAI/job%j.err
#SBATCH -p gpu-H200,gpu-A100



module load python3/anaconda/3.12

# Activate your conda environment
source activate /work/${USER}/ConvAI/MathTutorBench/mathtutor_env

cd /work/${USER}/ConvAI/MathTutorBench

for cfg in configs/*.yaml; do
    fname=$(basename "$cfg")
    echo "Benchmarking config: $fname"
    python main.py --tasks "$fname" --provider local "$@"
done

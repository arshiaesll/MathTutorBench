#!/bin/sh
#SBATCH --job-name=mathtutorbench-mistakecorrection
#SBATCH -N 1    ## requests on 1 node
#SBATCH --gres=gpu:1   # request 1 GPU
#SBATCH --time=12:00:00
#SBATCH --output=/work/${USER}/VLA_results/ConvAI/job%j.out
#SBATCH --error=/work/${USER}/VLA_results/ConvAI/job%j.err
#SBATCH -p gpu-H200,gpu-A100 

# Load Anaconda module (adjust the path if necessary)
module load python3/anaconda/3.12

# Activate your conda environment
source activate /work/${USER}/ConvAI/mathtutorbench/mathtutor_env

# Move into your project directory
cd /work/${USER}/ConvAI/mathtutorbench

# Run through configs
for cfg in configs/*.yaml; do
    fname=$(basename "$cfg")
    echo "Benchmarking config: $fname"
    python main.py --tasks "$fname" --provider local --model_args "$@"
done

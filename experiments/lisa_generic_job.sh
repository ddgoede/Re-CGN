#!/bin/bash

#SBATCH --partition=gpu_shared_course
#SBATCH --gres=gpu:1
#SBATCH --job-name=SampleJob
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00:00
#SBATCH --mem=32000M
#SBATCH --output=slurm/output_%A.out

module purge
module load 2021
module load Anaconda3/2021.05

# Activate your environment
source deactivate
source activate gcn-gpu

# Run your code
echo "Running command: python imagenet_eval_contributions.py"
python imagenet_eval_contributions.py
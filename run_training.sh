#!/bin/bash
#SBATCH --job-name=segformer-test
#SBATCH --account=project_2021015
#SBATCH --partition=gpumedium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:gh200:1
#SBATCH --output=/scratch/project_2021015/logs/%x-%j.out

module purge
module load python-pytorch/2.10
export PYTHONUSERBASE=/projappl/project_2021015/my-python-env
DATA_DIR=/scratch/project_2021015/training_data
set -x
srun python3 train_segformer.py --seed 42 --backbone b0 --loss CCE --data-dir="$DATA_DIR"
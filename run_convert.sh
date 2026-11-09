#!/bin/bash
#SBATCH --job-name=segformer-b5-conversion
#SBATCH --account=project_2021015
#SBATCH --partition=gpumedium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=00:10:00                
#SBATCH --output=/scratch/project_2021015/logs/%x-%A_%a.out

module purge
module load python-pytorch/2.10
export PYTHONUSERBASE=/projappl/project_2021015/my-python-env
export PYTHONUNBUFFERED=1
DATA_DIR=/scratch/project_2021015/training_data

set -x
srun python3 convert_hf.py --seed 42 43 44 45 46 --backbone b5 --loss CCE wCCE Focal Dice GenDice Combo Tversky FocalTversky AsymmetricUnifiedFocal LovaszSoftmax --data-dir="$DATA_DIR"
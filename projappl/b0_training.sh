#!/bin/bash
#SBATCH --job-name=segformer-b0
#SBATCH --account=project_2021015
#SBATCH --partition=gpumedium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=02:00:00
#SBATCH --array=0-49%10                   
#SBATCH --output=/scratch/project_2021015/logs/%x-%A_%a.out

SEEDS=(42 43 44 45 46)
LOSSES=(CCE wCCE Focal Dice GenDice Combo Tversky FocalTversky AsymmetricUnifiedFocal LovaszSoftmax)

N_SEEDS=${#SEEDS[@]}
LOSS=${LOSSES[$((SLURM_ARRAY_TASK_ID / N_SEEDS))]}
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID % N_SEEDS))]}

if [[ -z "$LOSS" || -z "$SEED" ]]; then
    echo "Task $SLURM_ARRAY_TASK_ID out of range" >&2
    exit 1
fi
echo "Task $SLURM_ARRAY_TASK_ID: loss=$LOSS seed=$SEED"

module purge
module load python-pytorch/2.10
export PYTHONUSERBASE=/projappl/project_2021015/my-python-env
export PYTHONUNBUFFERED=1
DATA_DIR=/scratch/project_2021015/training_data

set -x
srun python3 train_segformer.py --seed "$SEED" --backbone b0 --loss "$LOSS" --data-dir="$DATA_DIR"
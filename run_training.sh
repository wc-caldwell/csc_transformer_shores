#!/bin/bash
#SBATCH --account=project_2021015
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:gh200:1

module purge
module load python-pytorch/2.10

set -x

tar xf /scratch/project_2021015/cifar-10-python.tar.gz -C $TMPDIR

srun python3 cifar10_cnn.py --data_path=$TMPDIR/cifar-10-batches-py
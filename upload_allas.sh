#!/bin/bash
#SBATCH --job-name=models-to-allas
#SBATCH --account=project_2021015
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:30:00
#SBATCH --array=0-49%10                   
#SBATCH --output=/scratch/project_2021015/logs/%x-%A_%a.out

module purge
module load allas

cd "${1:?Usage: sbatch compress.sh /path/to/folders}" || exit 1
dirs=(*/)
d=${dirs[$SLURM_ARRAY_TASK_ID]%/}
tar -czf "$d.tar.gz" "$d" && tar -tzf "$d.tar.gz" > /dev/null && echo "OK: $d"

a-put "$d.tar.gz"
#!/bin/bash
#SBATCH --job-name=clear_conda
#SBATCH --partition=ada
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

echo "Node: $(hostname)"
nvidia-smi

module purge
module load cuda/12.4
module load conda/3.12

conda clean --all --yes
for env in $(conda env list | grep -v 'base' | grep -v '#' | awk '{print $1}');
    do conda env remove -n $env -y;
done

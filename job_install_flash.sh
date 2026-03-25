#!/bin/bash
#SBATCH --job-name=four_probe
#SBATCH --partition=ada
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

echo "Node: $(hostname)"
nvidia-smi

module purge
module load cuda/12.4
source ~/miniconda3/etc/profile.d/conda.sh
conda activate collm

pip install -U triton
pip install flash-linear-attention

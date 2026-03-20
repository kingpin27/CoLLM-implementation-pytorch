#!/bin/bash
#SBATCH --job-name=bash
#SBATCH --partition=ada
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:2
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

echo "Node: $(hostname)"
nvidia-smi

module purge
module load cuda/12.4  
source ~/miniconda3/etc/profile.d/conda.sh
conda activate collm


# only after first run
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Persist Hugging Face caches across Slurm jobs.
export HF_HOME="/home/anirban/anishc/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export DIFFUSERS_CACHE="$HF_HOME/diffusers"
mkdir -p "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$DIFFUSERS_CACHE"

cd /home/anirban/anishc/CoLLM-implementation-pytorch

start_ts=$(date +%s)
echo "Training started at: $(date '+%Y-%m-%d %H:%M:%S')"

NUM_GPUS=${SLURM_GPUS_ON_NODE:-2}
torchrun --standalone --nproc_per_node=$NUM_GPUS train.py

end_ts=$(date +%s)
elapsed_sec=$((end_ts - start_ts))
echo "Training finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Total runtime: ${elapsed_sec} seconds"
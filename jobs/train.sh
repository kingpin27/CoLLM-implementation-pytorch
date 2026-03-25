#!/bin/bash
#SBATCH --job-name=collm
#SBATCH --partition=ada
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
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

export PROCESSOR_NAME = "Qwen/Qwen3.5-0.8B"
export MODEL_NAME = "Qwen/Qwen3.5-0.8B"
export PROJ_DIM = 512
export NUM_EMBS = 4
export HID_DIM = 1024
export KEEP_LAYERS = 16

export EPOCHS = 1
export BATCH_SIZE = 64
export NUM_WORKERS = 8
export NUM_BATCHES = $($(1024*128)/$BATCH_SIZE)

export PROBE_TEMP = 1
export INFONCE_TEMP = 0.1
export DIVERSITY_WEIGHT = 0.1

cd /home/anirban/anishc/CoLLM-implementation-pytorch

start_ts=$(date +%s)
echo "Traininig started at: $(date '+%Y-%m-%d %H:%M:%S')"

srun python src/train.py

end_ts=$(date +%s)
elapsed_sec=$((end_ts - start_ts))
echo "Training finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Total runtime: ${elapsed_sec} seconds"

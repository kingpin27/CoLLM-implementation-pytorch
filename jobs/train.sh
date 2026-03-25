#!/bin/bash
#SBATCH --job-name=collm
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

# only after first run
# echo "Setting up HF offline..."
# export HF_HUB_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1

# Persist Hugging Face caches across Slurm jobs.
echo "Setting up HF cache dir..."
export HF_HOME="/home/anirban/anishc/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export DIFFUSERS_CACHE="$HF_HOME/diffusers"
mkdir -p "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$DIFFUSERS_CACHE"


echo "Setting up Training hyperparameters env..."
export PROCESSOR_NAME="Qwen/Qwen3.5-0.8B"
export MODEL_NAME="Qwen/Qwen3.5-0.8B"
export PROJ_DIM=512
export NUM_EMBS=4
export HID_DIM=1024
export KEEP_LAYERS=16

export EPOCHS=1
export BATCH_SIZE=64
export NUM_WORKERS=8
export NUM_BATCHES=$(( (1024 * 128) / BATCH_SIZE ))

export PROBE_TEMP=1
export INFONCE_TEMP=0.1
export DIVERSITY_WEIGHT=0.1

# --- Conda env setup ---
echo "Setting up Conda env..."
ENV_NAME="collm_env_anishc"
if conda env list | grep -qE "^${ENV_NAME}\s"; then
    echo "Conda env '${ENV_NAME}' found, activating..."
else
    echo "Conda env '${ENV_NAME}' not found, creating..."
    conda create -y -n "$ENV_NAME" python=3.12
    conda run -n "$ENV_NAME" pip install \
        torch torchvision \
        flash-linear-attention causal-conv1d \
        transformers accelerate \
        diffusers \
        tqdm pillow numpy
    echo "Conda env '${ENV_NAME}' created and packages installed"
fi
conda activate "$ENV_NAME"
# ----------------------

echo "Setting up CWD..."
cd /home/anirban/anishc/CoLLM-implementation-pytorch

start_ts=$(date +%s)
echo "Traininig started at: $(date '+%Y-%m-%d %H:%M:%S')"

srun python src/train.py

end_ts=$(date +%s)
elapsed_sec=$((end_ts - start_ts))
echo "Training finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Total runtime: ${elapsed_sec} seconds"

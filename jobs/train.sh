#!/bin/bash
#SBATCH --job-name=train_collm
#SBATCH --partition=ada
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

echo "Node: $(hostname)"
nvidia-smi

module purge
module load cuda/12.4

# Initialize conda for non-interactive shell
source /home/anirban/anishc/miniconda3/etc/profile.d/conda.sh
conda activate collm5

# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 --no-cache-dir
# pip install tqdm pillow numpy wandb transformers accelerate --no-cache-dir

# only after first run
# echo "Setting up HF offline..."
# export HF_HUB_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1

source ~/.secrets

export WANDB__SERVICE_WAIT=300

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
export NUM_WORKERS=4
export NUM_BATCHES=$(( (1024 * 512) / BATCH_SIZE ))

export PROBE_TEMP=1.0
export INFONCE_TEMP=0.07
export DIVERSITY_WEIGHT=0.1

# echo "resuming previous experiment..."
# export EXPERIMENT_ID=adsasd

echo "Setting up CWD..."
cd /home/anirban/anishc/CoLLM-implementation-pytorch

start_ts=$(date +%s)
echo "Traininig started at: $(date '+%Y-%m-%d %H:%M:%S')"

PYTHON="/home/anirban/anishc/miniconda3/envs/collm5/bin/python"
srun "$PYTHON" src/train.py

end_ts=$(date +%s)
elapsed_sec=$((end_ts - start_ts))
echo "Training finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Total runtime: ${elapsed_sec} seconds"

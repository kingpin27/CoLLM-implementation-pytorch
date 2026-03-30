#!/bin/bash
#SBATCH --job-name=train_collm
#SBATCH --partition=a100
#SBATCH --ntasks=1                  # one torchrun master process per node
#SBATCH --cpus-per-task=16          # 8 workers × 2 GPUs
#SBATCH --mem=64G
#SBATCH --gres=gpu:2                # <-- 2 GPUs on one node
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

echo "Node: $(hostname)"
nvidia-smi

module purge
module load cuda/12.4

source /home/anirban/anishc/miniconda3/etc/profile.d/conda.sh
conda activate collm5

source ~/.secrets

export WANDB__SERVICE_WAIT=300

echo "Setting up HF cache dir..."
export HF_HOME="/home/anirban/anishc/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export DIFFUSERS_CACHE="$HF_HOME/diffusers"
mkdir -p "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$DIFFUSERS_CACHE"

NUM_GPUS=2

echo "Setting up Training hyperparameters env..."
export PROCESSOR_NAME="Qwen/Qwen3.5-0.8B"
export MODEL_NAME="Qwen/Qwen3.5-0.8B"
export PROJ_DIM=512
export NUM_EMBS=4
export HID_DIM=1024
export KEEP_LAYERS=16

export EPOCHS=1
# Per-GPU batch size. Effective global batch = BATCH_SIZE * NUM_GPUS.
# With 4 GPUs and BATCH_SIZE=64 you get a global batch of 256.
export BATCH_SIZE=64
export NUM_WORKERS=8                # per DataLoader (each rank spawns this many)
export NUM_BATCHES=$(( (1024 * 1024) / (BATCH_SIZE * NUM_GPUS) ))
export K_HARD=4096

export PROBE_TEMP=1.0
export INFONCE_TEMP=0.07
export DIVERSITY_WEIGHT=0.1

# echo "resuming previous experiment..."
# export EXPERIMENT_ID=adsasd

echo "Setting up CWD..."
cd /home/anirban/anishc/CoLLM-implementation-pytorch

start_ts=$(date +%s)
echo "Training started at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Launching $NUM_GPUS processes via torchrun..."

PYTHON="/home/anirban/anishc/miniconda3/envs/collm5/bin/python"

# torchrun spawns NUM_GPUS worker processes on this node and sets
# RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT automatically.
torchrun \
    --standalone \
    --nproc_per_node="$NUM_GPUS" \
    --rdzv_backend=c10d \
    src/train.py

end_ts=$(date +%s)
elapsed_sec=$((end_ts - start_ts))
echo "Training finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Total runtime: ${elapsed_sec} seconds"

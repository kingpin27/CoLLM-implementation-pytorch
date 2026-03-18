#!/bin/bash
#SBATCH --job-name=bash
#SBATCH --partition=ada
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
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
conda activate SwiftEdit


# only after first run
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Persist Hugging Face caches across Slurm jobs.
export HF_HOME="/home/anirban/anishc/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export DIFFUSERS_CACHE="$HF_HOME/diffusers"
mkdir -p "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$DIFFUSERS_CACHE"

cd /home/anirban/anishc/extractor2

segment_idx=0
segment_size=128

start_ts=$(date +%s)
echo "Extraction started at for segment index ${segment_idx} and segment size ${segment_size}: $(date '+%Y-%m-%d %H:%M:%S')"

srun python -m torch.distributed.run --nproc_per_node=2 extract.py --segment-index ${segment_idx} --segment-size ${segment_size}

end_ts=$(date +%s)
elapsed_sec=$((end_ts - start_ts))
echo "Extraction finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Total runtime: ${elapsed_sec} seconds"
#!/bin/bash
#SBATCH --job-name=circo_eval
#SBATCH --partition=long
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
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

# Initialize conda for non-interactive shell
source /home/anirban/anishc/miniconda3/etc/profile.d/conda.sh
conda activate collm5

# Persist Hugging Face caches across Slurm jobs.
echo "Setting up HF cache dir..."
export HF_HOME="/home/anirban/anishc/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export DIFFUSERS_CACHE="$HF_HOME/diffusers"
mkdir -p "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$DIFFUSERS_CACHE"


echo "Setting up CWD..."
cd /home/anirban/anishc/CoLLM-implementation-pytorch


start_ts=$(date +%s)
echo "CIRCO eval started at: $(date '+%Y-%m-%d %H:%M:%S')"

srun python src/circo_eval.py \
    --checkpoint ~/CoLLM-implementation-pytorch/collm_4probes_645f4850_20260329_093448.pt \
    --split test \
    --annotations ~/CIRCO/annotations/test.json \
    --coco-img-dir ~/CIRCO/COCO2017_unlabeled/unlabeled2017  \
    --coco-image-info ~/CIRCO/COCO2017_unlabeled/annotations/image_info_unlabeled2017.json \
    --output ~/CoLLM-implementation-pytorch/submission_test_645f4850.json \
    --num-embeddings 4

end_ts=$(date +%s)
elapsed_sec=$((end_ts - start_ts))
echo "CIRCO eval finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Total runtime: ${elapsed_sec} seconds"

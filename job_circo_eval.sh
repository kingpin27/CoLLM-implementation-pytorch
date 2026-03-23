#!/bin/bash
#SBATCH --job-name=bash
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
# export HF_HUB_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1

# Persist Hugging Face caches across Slurm jobs.
export HF_HOME="/home/anirban/anishc/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export DIFFUSERS_CACHE="$HF_HOME/diffusers"
mkdir -p "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$DIFFUSERS_CACHE"

echo "Copying COCO to local scratch..."
mkdir -p $TMPDIR/coco_unlabeled
rsync -a --info=progress2 ~/CIRCO/COCO2017_unlabeled/unlabeled2017/ $TMPDIR/coco_unlabeled/
echo "Done copying."

cd /home/anirban/anishc/CoLLM-implementation-pytorch

start_ts=$(date +%s)
echo "Traininig started at: $(date '+%Y-%m-%d %H:%M:%S')"

srun python circo_eval.py \
    --checkpoint ~/CoLLM-implementation-pytorch/collm_0148243.pt \
    --split val \
    --annotations ~/CIRCO/annotations/val.json \
    --coco-img-dir $TMPDIR/coco_unlabeled  \
    --coco-image-info ~/CIRCO/COCO2017_unlabeled/annotations/image_info_unlabeled2017.json \
    --output ~/CIRCO/submission_val.json

end_ts=$(date +%s)
elapsed_sec=$((end_ts - start_ts))
echo "Training finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Total runtime: ${elapsed_sec} seconds"

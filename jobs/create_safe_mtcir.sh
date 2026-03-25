#!/bin/bash
#SBATCH --job-name=bash
#SBATCH --partition=long
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

echo "Node: $(hostname)"

cd /home/anirban/anishc/CoLLM-implementation-pytorch


python3 scripts/create_safe_mtcir.py -i ./MTCIR/mtcir_expanded_shuffled.jsonl -o ./MTCIR/mtcir_expanded_shuffled_safe.jsonl --seed 42 -l 100

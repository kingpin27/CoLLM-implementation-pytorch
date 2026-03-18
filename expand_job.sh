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

python3 scripts/expand_annotation.py -i ./MTCIR/mtcir.jsonl -o ./MTCIR/mtcir_expanded.jsonl

python3 scripts/shuffle_annotations.py -i ./MTCIR/mtcir_expanded.jsonl -o ./MTCIR/mtcir_expanded_shuffled.jsonl --seed 42

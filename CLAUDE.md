# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CoLLM is a Composed Image Retrieval model that wraps Qwen3.5-0.8B (a vision-language model) to produce CLIP-compatible embeddings. Given a reference image and a modification text (e.g., "same shirt but in blue"), it produces an embedding that matches the target image in CLIP space.

## Commands

### Training
```bash
# Multi-GPU training (2 GPUs) via Accelerate
accelerate launch --num_processes=2 --mixed_precision=bf16 src/train.py

# Single-GPU training
python src/train.py

# Submit SLURM job
sbatch jobs/train.sh
```

### Evaluation
```bash
# Run CIRCO evaluation
python src/circo_eval.py

# Submit SLURM eval job
sbatch jobs/circo_eval.sh
```

### Data Preparation
```bash
# Pre-compute CLIP embeddings for target images
python scripts/generate_clip_emb.py

# Expand multi-modification JSONL annotations to one-per-modification
python scripts/expand_annotation.py

# Filter annotations with long modifications (>100 chars)
python scripts/create_safe_mtcir.py

# Shuffle annotations
python scripts/shuffle_annotations.py
```

### Environment
Training runs in conda env `collm5` with CUDA 12.4. Credentials loaded from `.secrets` (not tracked).

## Architecture

### Model (`src/models.py`)

`CoLLM` wraps the Qwen3.5-0.8B vision-language model:
- **Frozen visual encoder**: processes reference images via Qwen's Conv3d patch embedding + 24 vision transformer blocks
- **Pruned language model**: truncated to `KEEP_LAYERS=16` (from 32) to reduce memory
- **Learnable `cls_probes`**: shape `(K=4, H=1024)` — K probe vectors attending over the LM output sequence
- **`probe_proj`**: `Linear(H=1024, P=512)` — projects pooled features into CLIP embedding space (P=512 matches CLIP-B/32)

**Forward pass**: image+text → Qwen backbone → multi-head attention pooling with each probe attending to the full sequence → project to CLIP space → L2 normalize → return all K embeddings directly (no routing/weighting).

### Training (`src/train.py`)

**Loss**: multiprobe set-based symmetric InfoNCE (`multiprobe_infonce_loss`) between query embeddings (CoLLM output, K embeddings per query) and target embeddings (pre-computed frozen CLIP vectors). Multi-GPU negatives are gathered via `GatherLayer` to maximize batch size of negatives.

**Hard negative mining**: FIFO queue (`NegativeQueue`, size 4096) stores past target embeddings; top-`K_HARD=64` hardest negatives per query are mixed into the InfoNCE loss.

**Output diversity loss**: regularization term (`DIVERSITY_WEIGHT=0.1`) penalizes collapsed probes via gram matrix of both probe weights and output embeddings.

**Validation**: runs CIRCO val every `VAL_INTERVAL=500` batches computing mAP@{5,10,25,50} and R@{1,5,10,25,50}.

Key env vars controlling training (all have defaults in `train.py`):
- `PROJ_DIM=512`, `NUM_EMBS=4`, `HID_DIM=1024`, `KEEP_LAYERS=16`
- `BATCH_SIZE=32` (per GPU), `NUM_BATCHES=16384`
- `INFONCE_TEMP=0.1`, `DIVERSITY_WEIGHT=0.1`
- `QUEUE_SIZE=4096`, `K_HARD=64` (set to 0 to disable hard negatives)
- `EXPERIMENT_ID`: used for checkpoint naming and W&B run grouping

### Data Pipeline (`src/dataset.py`)

`MTCIRDataset` streams from a JSONL annotation file. It pre-indexes byte offsets at init for O(1) random access. Each sample contains:
- Source image (loaded from `./images/`)
- Pre-computed target CLIP embedding (loaded from `./embeddings/<id>.npy`)
- Modification text

Collation via `collm_contrastive_collate_fn` in `utils.py` drops failed samples (missing images/embeddings) rather than erroring.

**Batch construction note** (from README): batches are created offline as JSONL files; the custom sampler returns consecutive samples as batches during training. The `scripts/create_batches.py` script is the skeleton for generating hard/soft negative batches with a probability schedule `P=[0.2, 0.4, 0.6, 0.8, 1.0]` over training.

### Key Model Details

The Qwen3.5-0.8B backbone alternates layer types:
- Even layers (0,2,4,...): `Qwen3.5GatedDeltaNet` (linear attention with depthwise conv)
- Odd layers (1,3,5,...): standard `Qwen3.5Attention`

The model uses `bfloat16` mixed precision and gradient checkpointing on the language model to fit in VRAM.

### Checkpointing

`save_checkpoint()` in `utils.py` saves the full training state (model, optimizer, scheduler, batch index). Resume logic in `train.py` uses `find_latest_checkpoint()` which picks the highest `(epoch, batch)` checkpoint. Checkpoints go to `./checkpoints/` (gitignored).

### Multi-GPU Notes

`GatherLayer` in `utils.py` is a custom autograd function for all-gather that preserves gradients for the local rank's portion. This is critical: standard `dist.all_gather` breaks gradients. The gather is used for both InfoNCE negatives and diversity loss to maximize effective batch size.

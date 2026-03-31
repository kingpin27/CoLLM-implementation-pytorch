# CoLLM

Composed Image Retrieval via a learned multi-probe wrapper around Qwen3.5-0.8B. Given a reference image and a modification text, CoLLM produces a CLIP-compatible embedding that matches the described target image.

## Method

CoLLM adds a small set of learnable components on top of a frozen/pruned Qwen3.5-0.8B vision-language backbone:

- **K=4 cls_probes**: each probe attends over the full LM output sequence via multi-head attention pooling
- **probe_proj**: projects pooled features into CLIP-B/32 embedding space (512-dim)
- **probe_router**: learns soft routing weights over probes, annealed toward hard selection during training

Training uses symmetric InfoNCE against frozen pre-computed CLIP embeddings as targets, with an output diversity loss to prevent probe collapse.

## Data

Training uses the MTCIR dataset (stored as JSONL). Each sample contains a source image, modification text, and a pre-computed target CLIP embedding (`.npy`).

Batches are created offline as JSONL files. The custom sampler returns consecutive samples as batches during training, which allows hard/soft negatives to be arranged at batch-creation time.

## Training

```bash
# Multi-GPU (2x GPU via Accelerate DDP)
accelerate launch --num_processes=2 --mixed_precision=bf16 src/train.py

# Single GPU
python src/train.py
```

Key hyperparameters are controlled via environment variables (see `src/train.py` for defaults). Checkpoints are saved to `./checkpoints/` and training resumes automatically from the latest checkpoint.

Validation on CIRCO (mAP@{5,10,25,50} and R@{1,5,10,25,50}) runs every 500 batches.

## Evaluation

```bash
python src/circo_eval.py
```

Builds a CLIP gallery index from COCO images, encodes CIRCO queries with CoLLM, and computes retrieval metrics.

## Data Preparation

```bash
# Pre-compute target CLIP embeddings
python scripts/generate_clip_emb.py

# Expand multi-modification annotations to one-per-modification
python scripts/expand_annotation.py

# Shuffle annotations
python scripts/shuffle_annotations.py
```

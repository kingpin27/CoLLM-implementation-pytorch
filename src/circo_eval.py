import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

from models import CoLLM  # your model class

# ── device ────────────────────────────────────────────────────────────────────
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.mps.is_available()
    else "cpu"
)


# ── 1. Build the gallery index ─────────────────────────────────────────────────
def build_gallery(coco_img_dir: str, image_info_path: str, batch_size: int = 64):
    import clip

    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()

    with open(image_info_path) as f:
        image_info = json.load(f)["images"]

    # pre-collect valid paths
    valid = []
    for info in image_info:
        path = os.path.join(coco_img_dir, info["file_name"])
        if os.path.exists(path):
            valid.append((info["id"], path))
        else:
            print(f"Missing: {path}")

    coco_ids, embs = [], []

    for batch_start in tqdm(range(0, len(valid), batch_size), desc="Encoding gallery"):
        batch = valid[batch_start : batch_start + batch_size]

        imgs, ids = [], []
        for coco_id, path in batch:
            try:
                img = preprocess(Image.open(path).convert("RGB"))
                imgs.append(img)
                ids.append(coco_id)
            except Exception as e:
                print(f"Skipping {path}: {e}")

        if not imgs:
            continue

        batch_tensor = torch.stack(imgs).to(device)

        with torch.no_grad():
            batch_emb = clip_model.encode_image(batch_tensor).float()
            batch_emb = F.normalize(batch_emb, dim=-1).cpu()

        coco_ids.extend(ids)
        embs.append(batch_emb)

        del batch_tensor, batch_emb, imgs, batch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    gallery_embs = torch.cat(embs, dim=0)  # (N, P)
    return gallery_embs, coco_ids


# ── 2. Load CIRCO queries ──────────────────────────────────────────────────────
def load_queries(annotations_path: str, coco_img_dir: str):
    """
    Returns list of dicts with keys: query_id, image_path, modification_text
    """
    with open(annotations_path) as f:
        annotations = json.load(f)

    queries = []
    for ann in annotations:
        queries.append(
            {
                "query_id": str(ann["id"]),
                "image_path": os.path.join(
                    coco_img_dir, f"{ann['reference_img_id']:012d}.jpg"
                ),
                # CIRCO annotation field for the relative caption:
                "modification_text": ann["relative_caption"],
            }
        )
    return queries


# ── 3. Encode a single query with CoLLM ───────────────────────────────────────
@torch.no_grad()
def encode_queries_batch(
    model, processor, images, texts, gallery_embs, probe_temp: float = 1.0
):
    """
    Mirrors the soft probe selection used during training:
      1. Compute per-probe similarity against every gallery image.
      2. Use those similarities as weights (softmax over K) to form a
         convex combination of the K probe embeddings.
      3. Re-normalise to unit norm and score against the full gallery.

    At eval time we don't have a ground-truth target to derive probe_weights
    from (as in train.py's per_probe_sim = einsum("bkp,bp->bk", ...)), so
    we use each probe's mean similarity over the full gallery as a proxy for
    its relevance — a soft consensus score.

    Args:
        probe_temp: temperature for soft probe selection (should match
                    training PROBE_TEMP; lower = closer to hard argmax).

    Returns: (B, N) similarity matrix — use .topk(50) per row.
    """
    embeddings = model.forward(
        images=images,
        text=texts,
        processor=processor,
    )  # (B, K, P)

    B, K, P = embeddings.shape

    # Normalise all probe embeddings once
    embeddings = F.normalize(embeddings.float(), dim=-1)  # (B, K, P)

    # ── Soft probe selection ──────────────────────────────────────────────
    # Per-probe similarity against every gallery image: (B, K, N)
    # gallery_embs is already unit-normalised → dot product = cosine sim
    probe_gallery_sims = torch.einsum(
        "bkp,np->bkn", embeddings, gallery_embs
    )  # (B, K, N)

    # Summarise each probe's "relevance" as its mean sim over the gallery,
    # then softmax over K to get the convex-combination weights.
    per_probe_score = probe_gallery_sims.mean(dim=-1)  # (B, K)
    probe_weights = torch.softmax(per_probe_score / probe_temp, dim=1)  # (B, K)

    # Weighted combination of probe embeddings → single query vector per item
    best_emb = torch.einsum("bk,bkp->bp", probe_weights, embeddings)  # (B, P)
    best_emb = F.normalize(best_emb, dim=-1)  # re-normalise (mirrors train.py)

    # Final gallery similarities
    best_sims = best_emb @ gallery_embs.T  # (B, N)

    return best_sims  # (B, N) — use .topk(50) per row


# ── 4. Main ────────────────────────────────────────────────────────────────────
def main(args):

    # --- Build gallery ---
    # CIRCO gallery = COCO 2017 unlabeled images
    print("Building gallery index (this takes a while the first time)...")
    gallery_cache = (
        "/home/anirban/anishc/CoLLM-implementation-pytorch/clip_unlabeled2017_cache.pt"
    )
    if os.path.exists(gallery_cache):
        print("  Loading cached gallery...")
        data = torch.load(gallery_cache)
        gallery_embs, coco_ids = data["embs"], data["ids"]
    else:
        gallery_embs, coco_ids = build_gallery(
            coco_img_dir=args.coco_img_dir,
            image_info_path=args.coco_image_info,
        )
        torch.save({"embs": gallery_embs, "ids": coco_ids}, gallery_cache)
        print(f"  Gallery cached to {gallery_cache}")

    gallery_embs = gallery_embs.to(device)  # (N, P)
    print(f"Gallery size: {gallery_embs.shape[0]:,} images")

    # --- Load CIRCO queries ---
    queries = load_queries(args.annotations, args.coco_img_dir)
    print(f"Loaded {len(queries)} queries from {args.annotations}")

    # --- Load model ---
    print(f"Loading model from {args.checkpoint}")
    model = CoLLM(
        model_name=args.model_name,
        projection_dim=args.projection_dim,
        num_embeddings=args.num_embeddings,
        hidden_dim=args.hidden_dim,
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device).eval()

    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    print("Model ready")

    batch_size = 4

    # --- Retrieve top-50 for each query ---
    predictions = {}
    for batch_start in tqdm(
        range(0, len(queries), batch_size), desc="Encoding queries"
    ):
        batch = queries[batch_start : batch_start + batch_size]
        images = [Image.open(q["image_path"]).convert("RGB") for q in batch]
        texts = [q["modification_text"] for q in batch]

        best_sims = encode_queries_batch(
            model, processor, images, texts, gallery_embs, args.probe_temp
        )  # (B, N)

        for i, q in enumerate(batch):
            top50_local = best_sims[i].topk(50).indices
            predictions[q["query_id"]] = [
                coco_ids[j] for j in top50_local.cpu().tolist()
            ]

    # --- Save submission file ---
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(predictions, f)
    print(f"\nSaved submission → {args.output}")
    print("Submit this file to: https://circo.micc.unifi.it/evaluation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to collm_*.pt")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument(
        "--annotations", required=True, help="CIRCO/annotations/test.json or val.json"
    )
    parser.add_argument(
        "--coco-img-dir", required=True, help="COCO2017_unlabeled/unlabeled2017/"
    )
    parser.add_argument(
        "--coco-image-info",
        required=True,
        help="COCO2017_unlabeled/annotations/image_info_unlabeled2017.json",
    )
    parser.add_argument("--output", default="submission.json")
    # must match your training config exactly:
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--projection-dim", type=int, default=512)
    parser.add_argument("--num-embeddings", type=int, default=4)
    parser.add_argument("--probe-temp", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    args = parser.parse_args()
    main(args)

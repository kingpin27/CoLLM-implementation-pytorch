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
                "modification_text": ann["relative_caption"],
            }
        )
    return queries


# ── 3. Encode queries with CoLLM ──────────────────────────────────────────────
@torch.no_grad()
def encode_queries_batch(
    model,
    processor,
    images,
    texts,
    gallery_embs,
    aggregation: str,
):
    """
    Produces per-query similarity scores against the full gallery.

    aggregation choices
    -------------------
    "max_pool"   — takes the max across K probes per gallery image.
                   Matches the training loss (multiprobe_infonce_loss).

    "mean_pool"  — averages K probes then re-normalises.
    """
    embeddings = model.forward(
        images=images,
        text=texts,
        processor=processor,
    )  # (B, K, P)

    embeddings = F.normalize(embeddings.float(), dim=-1)  # (B, K, P)

    if aggregation == "max_pool":
        probe_sims = torch.einsum("bkp,np->bkn", embeddings, gallery_embs)  # (B, K, N)
        sims, _ = probe_sims.max(dim=1)  # (B, N)

    elif aggregation == "mean_pool":
        mean_emb = embeddings.mean(dim=1)  # (B, P)
        mean_emb = F.normalize(mean_emb, dim=-1)
        sims = mean_emb @ gallery_embs.T  # (B, N)

    else:
        raise ValueError(
            f"Unknown aggregation '{aggregation}'. Choose: max_pool | mean_pool"
        )

    return sims  # (B, N)


# ── 4. Main ────────────────────────────────────────────────────────────────────
def main(args):

    # --- Build / load gallery ---
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
    print(f"Model ready  |  aggregation={args.aggregation}")

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
            model=model,
            processor=processor,
            images=images,
            texts=texts,
            gallery_embs=gallery_embs,
            aggregation=args.aggregation,
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
    # Must match your training config exactly:
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--projection-dim", type=int, default=512)
    parser.add_argument("--num-embeddings", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument(
        "--aggregation",
        default="max_pool",
        choices=["max_pool", "mean_pool"],
        help="How to combine K probe embeddings at inference.",
    )
    args = parser.parse_args()
    main(args)

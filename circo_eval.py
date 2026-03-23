import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

from train3 import CoLLM  # your model class

# ── device ────────────────────────────────────────────────────────────────────
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.mps.is_available()
    else "cpu"
)


# ── 1. Build the gallery index ─────────────────────────────────────────────────
def build_gallery(coco_img_dir: str, image_info_path: str):
    """
    Returns:
        gallery_embs  : torch.Tensor  (N, P)  — CLIP L2-normed embeddings
        coco_ids      : list[int]              — COCO image id for each row
    """
    import clip  # pip install git+https://github.com/openai/CLIP.git

    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()

    with open(image_info_path) as f:
        image_info = json.load(f)["images"]

    coco_ids, embs = [], []
    for info in tqdm(image_info, desc="Encoding gallery"):
        img_path = os.path.join(coco_img_dir, info["file_name"])
        try:
            img = (
                preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
            )
            with torch.no_grad():
                emb = clip_model.encode_image(img).float()
                emb = F.normalize(emb, dim=-1)
            coco_ids.append(info["id"])
            embs.append(emb.cpu())
        except Exception as e:
            print(f"Skipping {img_path}: {e}")

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
                "image_path": os.path.join(coco_img_dir, ann["reference_img_id"]),
                # CIRCO annotation field for the relative caption:
                "modification_text": ann["relative_caption"],
            }
        )
    return queries


# ── 3. Encode a single query with CoLLM ───────────────────────────────────────
@torch.no_grad()
def encode_query(model, processor, image: Image.Image, text: str) -> torch.Tensor:
    """Returns a (P,) composed embedding."""
    embeddings = model.forward(
        images=[image],
        text=[text],
        processor=processor,
    )  # (1, K, P)

    # Mean-pool across K probes — safest at inference without a target signal.
    # Alternatively use: embeddings[0].norm(dim=-1).argmax() for hard selection.
    composed = embeddings[0].float().mean(dim=0)  # (P,)
    return F.normalize(composed, dim=-1)


# ── 4. Main ────────────────────────────────────────────────────────────────────
def main(args):
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

    # --- Build gallery ---
    # CIRCO gallery = COCO 2017 unlabeled images
    print("Building gallery index (this takes a while the first time)...")
    gallery_cache = args.checkpoint + ".gallery_cache.pt"
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

    # --- Retrieve top-50 for each query ---
    predictions = {}
    for q in tqdm(queries, desc="Encoding queries"):
        img = Image.open(q["image_path"]).convert("RGB")
        query_emb = encode_query(model, processor, img, q["modification_text"])  # (P,)

        # Cosine similarity against full gallery
        sims = gallery_embs @ query_emb  # (N,)
        top50_local = sims.topk(50).indices  # indices into gallery_embs

        # Map back to COCO image IDs (what the server expects)
        top50_coco_ids = [coco_ids[i] for i in top50_local.cpu().tolist()]
        predictions[q["query_id"]] = top50_coco_ids

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
    parser.add_argument("--projection-dim", type=int, default=768)
    parser.add_argument("--num-embeddings", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    args = parser.parse_args()
    main(args)

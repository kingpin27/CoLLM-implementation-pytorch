import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import CLIPModel, CLIPProcessor

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {device}")

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32", use_fast=False
)
model.eval()

IMAGES_FOLDER_PATH = "./images"
EMBEDDINGS_FOLDER_PATH = "./embeddings"
BATCH_SIZE = 256
NUM_WORKERS = 8


# ── Dataset ────────────────────────────────────────────────────────────────────
class ImageDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ip, op = self.pairs[idx]
        try:
            img = Image.open(str(ip)).convert("RGB")
        except Exception as e:
            print(f"Error loading {ip}: {e}")
            img = Image.new("RGB", (224, 224))
        return img, str(op)


def collate_fn(batch):
    imgs, paths = zip(*batch)
    return list(imgs), list(paths)


# ── Embedding ──────────────────────────────────────────────────────────────────
def process_batch(batch_imgs, batch_out_paths):
    inputs = processor(images=batch_imgs, return_tensors="pt", padding=True)
    pixel_values = inputs["pixel_values"].to(device)
    with torch.no_grad():
        output = model.vision_model(pixel_values=pixel_values)
        embs = model.visual_projection(output.pooler_output)  # (B, 512)
        embs = F.normalize(embs, p=2, dim=-1).cpu().numpy()
    del pixel_values, output, inputs
    for path, emb in zip(batch_out_paths, embs):
        np.save(path, emb)


# ── Collect all pairs ──────────────────────────────────────────────────────────
print("Scanning image folders...")
all_pairs = []
folders = [
    f
    for f in os.listdir(IMAGES_FOLDER_PATH)
    if os.path.isdir(os.path.join(IMAGES_FOLDER_PATH, f))
]
for folder in sorted(folders):
    for filename in os.listdir(os.path.join(IMAGES_FOLDER_PATH, folder)):
        ip_filepath = Path(os.path.join(IMAGES_FOLDER_PATH, folder, filename))
        if ip_filepath.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        op_filepath = Path(
            os.path.join(EMBEDDINGS_FOLDER_PATH, folder, ip_filepath.stem + ".npy")
        )
        if op_filepath.exists():
            continue  # resume-safe: skip already done
        op_filepath.parent.mkdir(parents=True, exist_ok=True)
        all_pairs.append((ip_filepath, op_filepath))

print(f"Images to process: {len(all_pairs):,}")

if len(all_pairs) == 0:
    print("Nothing to do — all embeddings already exist.")
    sys.exit(0)

# ── DataLoader + inference loop ────────────────────────────────────────────────
dataset = ImageDataset(all_pairs)
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    collate_fn=collate_fn,
    prefetch_factor=4,
    pin_memory=True,
    persistent_workers=True,
)

with tqdm(total=len(all_pairs), file=sys.stdout, unit="img") as pbar:
    for batch_imgs, batch_paths in loader:
        process_batch(batch_imgs, batch_paths)
        pbar.update(len(batch_imgs))
        torch.cuda.empty_cache()

print("Done.")

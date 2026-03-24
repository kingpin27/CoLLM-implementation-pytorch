import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
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
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()

IMAGES_FOLDER_PATH = "./images"
EMBEDDINGS_FOLDER_PATH = "./embeddings"
BATCH_SIZE = 512  # Ada 6000 can handle this easily


def process_batch(batch_imgs, batch_out_paths):
    inputs = processor(images=batch_imgs, return_tensors="pt", padding=True)
    pixel_values = inputs["pixel_values"].to(device)
    with torch.no_grad():
        output = model.vision_model(pixel_values=pixel_values)
        embs = model.visual_projection(output.pooler_output)  # (B, 512)
        embs = F.normalize(embs, p=2, dim=-1).cpu().numpy()
    for path, emb in zip(batch_out_paths, embs):
        np.save(path, emb)


# collect all valid (image_path, output_path) pairs first
all_pairs = []
folders = [
    f
    for f in os.listdir(IMAGES_FOLDER_PATH)
    if os.path.isdir(os.path.join(IMAGES_FOLDER_PATH, f))
]
for folder in folders:
    for filename in os.listdir(os.path.join(IMAGES_FOLDER_PATH, folder)):
        ip_filepath = Path(os.path.join(IMAGES_FOLDER_PATH, folder, filename))
        op_filepath = Path(
            os.path.join(EMBEDDINGS_FOLDER_PATH, folder, ip_filepath.stem + ".npy")
        )
        if op_filepath.exists():
            continue  # skip already processed
        op_filepath.parent.mkdir(parents=True, exist_ok=True)
        all_pairs.append((ip_filepath, op_filepath))

print(f"Total images to process: {len(all_pairs):,}")

batch_imgs, batch_paths = [], []

with tqdm(total=len(all_pairs), file=sys.stdout) as pbar:
    for ip_filepath, op_filepath in all_pairs:
        try:
            img = Image.open(str(ip_filepath)).convert("RGB")
        except Exception as e:
            print(f"Skipping {ip_filepath}: {e}")
            continue

        batch_imgs.append(img)
        batch_paths.append(op_filepath)

        if len(batch_imgs) == BATCH_SIZE:
            process_batch(batch_imgs, batch_paths)
            pbar.update(len(batch_imgs))
            batch_imgs, batch_paths = [], []

    # flush remaining
    if batch_imgs:
        process_batch(batch_imgs, batch_paths)
        pbar.update(len(batch_imgs))

print("Done.")

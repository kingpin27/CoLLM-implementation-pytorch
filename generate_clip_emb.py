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

# Load model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("device")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

IMAGES_FOLDER_PATH = "./images"
EMBEDDINGS_FOLDER_PATH = "./embeddings"

folders = [
    f
    for f in os.listdir(IMAGES_FOLDER_PATH)
    if os.path.isdir(os.path.join(IMAGES_FOLDER_PATH, f))
]
pbar = tqdm(
    total=559447,
    file=sys.stdout,
    leave=False,
)
for folder in folders:
    for filename in os.listdir(os.path.join(IMAGES_FOLDER_PATH, folder)):
        ip_filepath = Path(os.path.join(IMAGES_FOLDER_PATH, folder, filename))

        # Save as .npy instead of original extension
        op_filepath = Path(
            os.path.join(EMBEDDINGS_FOLDER_PATH, folder, ip_filepath.stem + ".npy")
        )
        op_filepath.parent.mkdir(parents=True, exist_ok=True)

        # Load image from INPUT path (was incorrectly using op_filepath)
        image = Image.open(str(ip_filepath)).convert("RGB")
        inputs = {
            k: v.to(device)
            for k, v in processor(images=image, return_tensors="pt").items()
        }

        with torch.no_grad():
            image_embeds = model.get_image_features(**inputs)  # (1, 512)
            image_embeds = F.normalize(image_embeds, p=2, dim=-1)

        # Save as .npy (binary, not text)
        np.save(op_filepath, image_embeds[0].cpu().numpy())

        pbar.update(1)

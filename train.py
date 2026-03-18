import torch
from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image


class MTCIRDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.examples = pd.read_json(annotations_file, lines=True)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        sample = self.examples.iloc[idx]

        source_path = os.path.join(self.img_dir, sample["image"])
        target_path = os.path.join(self.img_dir, sample["target_image"])

        source_image = Image.open(source_path).convert("RGB")
        target_image = Image.open(target_path).convert("RGB")

        if self.transform is not None:
            source_image = self.transform(source_image)
        if self.target_transform is not None:
            target_image = self.target_transform(target_image)

        modifications = sample["modifications"]

        return {
            "id": sample["id"],
            "image": source_image,
            "target_image": target_image,
            "modification_text": modifications[0]
        }
    
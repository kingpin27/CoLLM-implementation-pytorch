import json
import os

import numpy as np
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset


class MTCIRDataset(Dataset):
    def __init__(
        self,
        annotations_file,
        img_dir,
        emb_dir,
        LOGGER,
        transform=None,
        target_transform=None,
    ):
        self.img_dir = img_dir
        self.emb_dir = emb_dir
        self.annotations_file = annotations_file
        self.offsets = []
        self.logger = LOGGER

        with open(self.annotations_file, "r", encoding="utf-8") as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                if line.strip():
                    self.offsets.append(offset)

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.offsets)

    def _read_record(self, idx):
        with open(self.annotations_file, "r", encoding="utf-8") as f:
            f.seek(self.offsets[idx])
            line = f.readline()
        return json.loads(line)

    def __getitem__(self, idx):
        record = self._read_record(idx)
        source_path = os.path.join(self.img_dir, record["image"])
        # target_path = os.path.join(self.img_dir, record["target_image"])

        try:
            source_image = Image.open(source_path).convert("RGB")
        except UnidentifiedImageError:
            self.logger.warning(
                "Cannot identify image (empty or corrupt): %s", source_path
            )
            return None
        except (OSError, FileNotFoundError) as e:
            self.logger.warning("Failed to open image %s: %s", source_path, e)
            return None

        emb_path = os.path.join(self.emb_dir, record["target_image"][:-4] + ".npy")
        try:
            target_image_emb = np.load(emb_path)
        except (FileNotFoundError, ValueError) as e:
            self.logger.warning("Failed to load embedding %s: %s", emb_path, e)
            return None

        if self.transform is not None:
            source_image = self.transform(source_image)

        return {
            "id": record["id"],
            "image": source_image,
            "target_image_emb": target_image_emb,
            "modification_text": (
                record["modifications"][0]
                if isinstance(record.get("modifications"), list)
                and len(record["modifications"]) > 0
                else record.get("modifications", "")
            ),
        }

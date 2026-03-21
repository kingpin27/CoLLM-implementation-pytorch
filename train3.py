import os
import json
import torch
import logging
import sys
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
from torch.nn import functional as F
from transformers import AutoModelForMultimodalLM, AutoProcessor
from tqdm.auto import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    stream=sys.stdout,
    force=True,
)
LOGGER = logging.getLogger("train2")
device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

LOGGER.info(f"accelerator type: {device}")

# FIX 1 (OOM): Tell PyTorch's allocator to use expandable segments
# to reduce fragmentation before we even start.
# os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")


def tensor_shape(tensor):
    return tuple(tensor.shape)


class MTCIRDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.annotations_file = annotations_file
        self.offsets = []

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

        source_image = Image.open(source_path).convert("RGB")
        # target_image = Image.open(target_path).convert("RGB")

        target_image = [0,0,0,0] # TODO: embedding using clip stored in some file preprocessed to reduce vram usage

        if self.transform is not None:
            source_image = self.transform(source_image)
        # if self.target_transform is not None:
        #     target_image = self.target_transform(target_image)

        return {
            "id": record["id"],
            "image": source_image,
            "target_image": target_image,
            "modification_text": (
                record["modifications"][0]
                if isinstance(record.get("modifications"), list) and len(record["modifications"]) > 0
                else record.get("modifications", "")
            ),
        }

class CoLLM(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, images, text):
        pass

    def loss(self):
        pass

def collm_contrastive_collate_fn(batch):
    return {
        "id": [item["id"] for item in batch],
        "image": [item["image"] for item in batch],
        "target_image": [item["target_image"] for item in batch],
        "modification_text": [item["modification_text"] for item in batch],
    }


def main():
    processor_name = "Qwen/Qwen3.5-0.8B"
    model_name = "Qwen/Qwen3.5-0.8B"

    batch_size = 4
    num_workers = 4
    # FIX 7 (NaN): Clip gradients to prevent fp16 explosion.
    max_grad_norm = 1.0

    LOGGER.info("Loading processor: %s", processor_name)
    processor = AutoProcessor.from_pretrained(processor_name, trust_remote_code=True)
    LOGGER.info("Processor loaded")

    LOGGER.info("Loading model: %s", model_name)
    model = CoLLM()
    model = model.to(device)
    LOGGER.info("Model loaded and ready")

    LOGGER.info("Creating dataset from %s", './MTCIR/mtcir_expanded_shuffled.jsonl')
    train_dataset = MTCIRDataset('./MTCIR/mtcir_expanded_shuffled.jsonl', './images')
    LOGGER.info("Dataset ready with %d samples", len(train_dataset))
    LOGGER.info("Creating DataLoader batch_size=%d, num_workers=%d", batch_size, num_workers)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        multiprocessing_context="spawn",
        pin_memory=False,
        collate_fn=collm_contrastive_collate_fn,
    )

    LOGGER.info("Initializing optimizer (AdamW, lr=1e-4)")
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    LOGGER.info("Optimizer will update %d parameter tensors", len(trainable_params))
    optimizer = torch.optim.adamw(trainable_params, lr=1e-4)

    # FIX 8 (NaN + OOM): Use a GradScaler so that fp16 underflow doesn't
    # silently zero out gradients and so that backward stays numerically stable.
    scaler = torch.amp.GradScaler(device, enabled=(model.device == "cuda"))

    EPOCHS = 1
    LOGGER.info("Starting training for %d epoch(s)", EPOCHS)

    for epoch in range(EPOCHS):
        LOGGER.info("Running epoch %d/%d", epoch + 1, EPOCHS)
        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{EPOCHS}",
            leave=False,
            file=sys.stdout,
        )

        for batch_idx, batch in enumerate(progress):
            model.train()
            optimizer.zero_grad()

            query_tokens = model.forward(
                batch["image"],
                batch["modification_text"],
                processor=processor,
                return_attention_mask=True,
            )

            loss = 0
            scores = 0

            # NaN guard: skip the update if loss is bad rather than corrupting
            # the model weights with NaN gradients.
            if not torch.isfinite(loss):
                LOGGER.warning("Batch=%d produced non-finite loss (%s), skipping update", batch_idx + 1, loss.item())
                optimizer.zero_grad()
                continue

            scaler.scale(loss).backward()
            # FIX 7 applied here: unscale then clip before the optimizer step.
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            progress.set_postfix(loss=float(loss.item()), step=batch_idx + 1)
            if batch_idx % 10 == 0:
                LOGGER.info(
                    "Epoch=%d batch=%d | loss=%.4f | scores=%s\n",
                    epoch + 1,
                    batch_idx + 1,
                    loss.item(),
                    tensor_shape(scores),
                )
    
    output_path = "collm_model.pt"
    torch.save(model.state_dict(), output_path)
    LOGGER.info("Model saved to %s", output_path)



if __name__ == '__main__':
    main()

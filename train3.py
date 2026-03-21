import json
import logging
import os
import sys

import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForMultimodalLM, AutoProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    stream=sys.stdout,
    force=True,
)
LOGGER = logging.getLogger("train2")
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.mps.is_available()
    else "cpu"
)  # only run this on nvidia hardware

LOGGER.info(f"accelerator type: {device}")

# FIX 1 (OOM): Tell PyTorch's allocator to use expandable segments
# to reduce fragmentation before we even start.
# os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")


def tensor_shape(tensor):
    return tuple(tensor.shape)


class MTCIRDataset(Dataset):
    def __init__(
        self, annotations_file, img_dir, transform=None, target_transform=None
    ):
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

        target_image = [
            0.1 for i in range(512)
        ]  # TODO: embedding using clip stored in some file preprocessed to reduce vram usage

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
                if isinstance(record.get("modifications"), list)
                and len(record["modifications"]) > 0
                else record.get("modifications", "")
            ),
        }


class CoLLM(torch.nn.Module):
    def __init__(
        self,
        model_name="Qwen/Qwen3.5-2B",
        projection_dim=256,
    ):
        super().__init__()
        self.projection_dim = projection_dim
        self.model_dtype = torch.float16 if device == "cuda" else "auto"

        self.model = AutoModelForMultimodalLM.from_pretrained(
            model_name,
            dtype=self.model_dtype,
            trust_remote_code=True,
        ).to(device)

        for p in self.model.model.visual.parameters():
            p.requires_grad = False
        keep_layers = 16
        self.model.model.language_model.layers = self.model.model.language_model.layers[
            :keep_layers
        ]
        self.model.lm_head = None

    @staticmethod
    def make_inputs(processor, image, text):
        if not isinstance(image, (list, tuple)):
            image = [image]
        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, (list, tuple)):
            text = [str(text)]
        if len(image) != len(text):
            raise ValueError("Number of images and texts must match for batching.")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": str(t)},
                ],
            }
            for t in text
        ]
        rendered = []
        for m in messages:
            if not isinstance(m, dict):
                raise TypeError(
                    f"Expected message dict for chat template, got {type(m)}"
                )
            rendered.append(
                processor.apply_chat_template(
                    [m],
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )
        enc = processor(
            text=rendered,
            images=list(image),
            return_tensors="pt",
            padding=True,
        )
        return {k: v.to(device) for k, v in enc.items()}

    def forward(self, images, text, processor):
        inputs = self.make_inputs(processor, images, text)
        outputs = self.model.model(
            **inputs, output_hidden_states=True, return_dict=True
        )
        hidden = outputs.hidden_states[-1]  # (batch, seq_len, 2048)
        return hidden

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
    model = CoLLM(model_name=model_name)
    model = model.to(device)
    LOGGER.info("Model loaded and ready")

    # img = Image.open("./images/00000/cat.webp")
    # txt = "describe this image"

    # hidden = model.forward(img, txt, processor)

    # print(hidden.shape)

    LOGGER.info("Creating dataset from %s", "./MTCIR/mtcir_expanded_shuffled.jsonl")
    train_dataset = MTCIRDataset("./MTCIR/mtcir_expanded_shuffled.jsonl", "./images")
    LOGGER.info("Dataset ready with %d samples", len(train_dataset))
    LOGGER.info(
        "Creating DataLoader batch_size=%d, num_workers=%d", batch_size, num_workers
    )
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
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)

    EPOCHS = 1
    LOGGER.info("Starting training for %d epoch(s)", EPOCHS)

    # for epoch in range(EPOCHS):
    #     LOGGER.info("Running epoch %d/%d", epoch + 1, EPOCHS)
    #     progress = tqdm(
    #         train_loader,
    #         desc=f"Epoch {epoch + 1}/{EPOCHS}",
    #         leave=False,
    #         file=sys.stdout,
    #     )

    #     for batch_idx, batch in enumerate(progress):
    #         model.train()
    #         optimizer.zero_grad()

    #         query_tokens = model.forward(
    #             batch["image"],
    #             batch["modification_text"],
    #             processor=processor,
    #             return_attention_mask=True,
    #         )

    #         loss = 0
    #         scores = 0

    #         # NaN guard: skip the update if loss is bad rather than corrupting
    #         # the model weights with NaN gradients.
    #         if not torch.isfinite(loss):
    #             LOGGER.warning("Batch=%d produced non-finite loss (%s), skipping update", batch_idx + 1, loss.item())
    #             optimizer.zero_grad()
    #             continue

    #         scaler.scale(loss).backward()
    #         # FIX 7 applied here: unscale then clip before the optimizer step.
    #         scaler.unscale_(optimizer)
    #         torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
    #         scaler.step(optimizer)
    #         scaler.update()

    #         progress.set_postfix(loss=float(loss.item()), step=batch_idx + 1)
    #         if batch_idx % 10 == 0:
    #             LOGGER.info(
    #                 "Epoch=%d batch=%d | loss=%.4f | scores=%s\n",
    #                 epoch + 1,
    #                 batch_idx + 1,
    #                 loss.item(),
    #                 tensor_shape(scores),
    #             )

    # output_path = "collm_model.pt"
    # torch.save(model.state_dict(), output_path)
    # LOGGER.info("Model saved to %s", output_path)


if __name__ == "__main__":
    main()

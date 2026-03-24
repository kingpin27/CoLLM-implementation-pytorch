import json
import logging
import os
import sys
from datetime import datetime

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForMultimodalLM,
    AutoProcessor,
    get_linear_schedule_with_warmup,
)

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
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")


def tensor_shape(tensor):
    return tuple(tensor.shape)


class MTCIRDataset(Dataset):
    def __init__(
        self, annotations_file, img_dir, emb_dir, transform=None, target_transform=None
    ):
        self.img_dir = img_dir
        self.emb_dir = emb_dir
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

        target_image_emb = np.load(
            os.path.join(self.emb_dir, record["target_image"][:-4] + ".npy")
        )

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


class CoLLM(torch.nn.Module):
    def __init__(
        self,
        model_name="Qwen/Qwen3.5-0.8B",
        projection_dim=512,
        num_embeddings=4,
        hidden_dim=512,
    ):
        super().__init__()
        self.projection_dim = projection_dim
        self.model_dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self.num_embeddings = num_embeddings

        self.model = AutoModelForMultimodalLM.from_pretrained(
            model_name,
            dtype=self.model_dtype,
            trust_remote_code=True,
            # attn_implementation="flash_attention_2",
        ).to(device)

        for p in self.model.model.visual.parameters():
            p.requires_grad = False
        keep_layers = 20
        self.model.model.language_model.layers = self.model.model.language_model.layers[
            :keep_layers
        ]
        self.model.lm_head = None

        self.cls_probes = torch.nn.Parameter(
            torch.randn(num_embeddings, hidden_dim, dtype=self.model_dtype)
        )
        self.probe_proj = torch.nn.Linear(
            hidden_dim,
            projection_dim,
            dtype=self.model_dtype,
        )

    def make_inputs(self, processor, image, text):
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
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            outputs = self.model.model(
                **inputs, output_hidden_states=True, return_dict=True
            )
        hidden = outputs.hidden_states[-1]  # cast OUT of fp16 immediately
        mask = inputs["attention_mask"].to(torch.bfloat16)  # (B, S)

        scores = torch.matmul(
            self.cls_probes.unsqueeze(0),  # (1, K, H)
            hidden.transpose(1, 2),  # (B, H, S)
        )  # (B, K, S)

        scores = scores.masked_fill(mask.unsqueeze(1) == 0, float("-inf"))
        attn = torch.softmax(scores, dim=-1)  # (B, K, S)
        pooled = torch.matmul(attn, hidden)  # (B, K, H)

        embeddings = self.probe_proj(pooled)  # (B, K, projection_dim)
        embeddings = F.normalize(embeddings, dim=-1)
        return embeddings


def collm_contrastive_collate_fn(batch):
    return {
        "id": [item["id"] for item in batch],
        "image": [item["image"] for item in batch],
        "target_image_emb": [item["target_image_emb"] for item in batch],  # fixed key
        "modification_text": [item["modification_text"] for item in batch],
    }


def log_vram(label=""):
    if device != "cuda":
        return
    allocated = torch.cuda.max_memory_allocated() / 1e9
    reserved = torch.cuda.max_memory_reserved() / 1e9
    LOGGER.info(
        f"VRAM [{label}] peak allocated={allocated:.2f}GB  reserved={reserved:.2f}GB"
    )
    torch.cuda.reset_peak_memory_stats()  # reset so next window is fresh


def param_summary(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    def fmt(n):
        if n >= 1e9:
            return f"{n / 1e9:.2f}B"
        if n >= 1e6:
            return f"{n / 1e6:.2f}M"
        if n >= 1e3:
            return f"{n / 1e3:.2f}K"
        return str(n)

    LOGGER.info(f"Total      : {fmt(total):>10}  ({total:,})")
    LOGGER.info(f"Trainable  : {fmt(trainable):>10}  ({trainable:,})")
    LOGGER.info(f"Frozen     : {fmt(frozen):>10}  ({frozen:,})")


def main():
    processor_name = "Qwen/Qwen3.5-0.8B"
    model_name = "Qwen/Qwen3.5-0.8B"
    projection_dim = 512  # same as CLIP-B = P
    num_embeddings = 4  # num of target proposals = K
    hidden_dim = 1024

    # Temperature for soft probe selection — lower = closer to hard argmax.
    # Can be annealed toward 0 over training for increasingly competitive probes.
    probe_temperature = 1
    # Temperature for InfoNCE contrastive loss.
    # 0.07 (CLIP default) is aggressive early in training; 0.1 is safer to start.
    infonce_temperature = 0.3
    diversity_weight = 0.1

    epochs = 1
    batch_size = 128  # = B
    num_workers = 8
    num_batches = int((1024 * 1024) / batch_size)

    LOGGER.info("Loading processor: %s", processor_name)
    processor = AutoProcessor.from_pretrained(processor_name, trust_remote_code=True)
    LOGGER.info("Processor loaded")

    LOGGER.info("Loading model: %s", model_name)
    model = CoLLM(
        model_name=model_name,
        projection_dim=projection_dim,
        num_embeddings=num_embeddings,
        hidden_dim=hidden_dim,
    )
    model = model.to(device)
    # model = torch.compile(model, mode="reduce-overhead")
    param_summary(model)
    LOGGER.info("Model loaded and ready")

    model.model.model.language_model.gradient_checkpointing_enable()
    model.model.model.language_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    LOGGER.info("Creating dataset from %s", "./MTCIR/mtcir_expanded_shuffled.jsonl")
    train_dataset = MTCIRDataset(
        "./MTCIR/mtcir_expanded_shuffled.jsonl",
        "./images",
        "./embeddings",
    )
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
        shuffle=True,
    )

    LOGGER.info("Initializing optimizer (AdamW, lr=1e-4)")
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    LOGGER.info("Optimizer will update %d parameter tensors", len(trainable_params))
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4, eps=1e-6)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=500,
        num_training_steps=num_batches * 5,  # so only decays lr to 80%
    )

    LOGGER.info("Starting training for %d epoch(s)", epochs)
    LOGGER.info("Training on total exmaples: %d", num_batches * batch_size)

    skipped = 0
    for epoch in range(epochs):
        LOGGER.info("Running epoch %d/%d", epoch + 1, epochs)
        pbar = tqdm(
            total=num_batches,
            file=sys.stdout,
            leave=False,
        )
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= num_batches:
                break
            model.train()
            optimizer.zero_grad()

            # forwars pass
            embeddings = model.forward(
                images=batch["image"],
                text=batch["modification_text"],
                processor=processor,
            )  # (B, K, P)

            # Guard: skip batch if backbone produced inf/nan hidden states.
            # This can happen early in training when fp16 attention overflows.
            if torch.isnan(embeddings).any() or torch.isinf(embeddings).any():
                skipped += 1
                LOGGER.warning(
                    "Epoch=%d batch=%d | bad embeddings (nan/inf), skipping "
                    "[total skipped=%d]",
                    epoch + 1,
                    batch_idx + 1,
                    skipped,
                )
                scheduler.step()
                pbar.update(1)
                continue

            # target embeddings: frozen CLIP vectors, not part of grad graph
            target_emb = (
                torch.stack([torch.from_numpy(e) for e in batch["target_image_emb"]])
                .to(device)
                .float()
            )  # (B, P)
            target_emb = F.normalize(target_emb, dim=-1)  # (B, P)

            # --- soft probe selection (fully differentiable) ---
            # Similarity of every probe against its own target: (B, K)
            per_probe_sim = torch.einsum(
                "bkp,bp->bk", embeddings.float(), target_emb
            )  # (B, K)

            # Soft convex combination weighted by proximity to target.
            # Gradient flows to ALL K probes; sharpness controlled by probe_temperature.
            # As probe_temperature -> 0 this approaches hard argmax.
            probe_weights = torch.softmax(
                per_probe_sim / probe_temperature, dim=1
            )  # (B, K)
            best_emb = torch.einsum(
                "bk,bkp->bp", probe_weights, embeddings.float()
            )  # (B, P) — still unit-norm after softmax combination (approx)
            best_emb = F.normalize(best_emb, dim=-1)  # re-normalise to be exact

            # --- symmetric InfoNCE loss ---
            # Logit matrix: each query's composed embedding vs every target in batch.
            # Diagonal entries are the positives.
            logits = (
                torch.matmul(best_emb, target_emb.T) / infonce_temperature
            )  # (B, B)
            labels = torch.arange(logits.size(0), device=device)  # (B,)

            # Regularise cls_probes directly — they live in a fixed hidden_dim space
            probe_gram = torch.mm(
                F.normalize(model.cls_probes.float(), dim=-1),
                F.normalize(model.cls_probes.float(), dim=-1).T,
            )  # (K, K)
            off_diag = probe_gram.masked_fill(
                torch.eye(num_embeddings, device=device, dtype=torch.bool), 0.0
            )
            diversity_loss = (off_diag**2).sum()

            loss = (
                F.cross_entropy(logits, labels)  # query -> target
                + F.cross_entropy(logits.T, labels)  # target -> query
            ) / 2 + diversity_weight * diversity_loss

            # backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            pbar.update(1)

            if batch_idx % 10 == 0:
                LOGGER.info(
                    "Epoch=%d batch=%d | loss=%.4f | lr=%.2e\n",
                    epoch + 1,
                    batch_idx + 1,
                    loss.item(),
                    scheduler.get_last_lr()[0],
                )
                log_vram(f"epoch={epoch + 1} batch={batch_idx + 1}")
        pbar.close()
        log_vram(f"epoch={epoch + 1} end")

    LOGGER.info("saving model to CoLLM.pt")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"collm_{timestamp}.pt"
    torch.save(model.state_dict(), model_filename)
    LOGGER.info(f"Model saved as: {model_filename}")


if __name__ == "__main__":
    main()

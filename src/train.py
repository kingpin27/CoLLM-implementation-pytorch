import os
import sys
import uuid
from datetime import datetime

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoProcessor
from transformers.optimization import get_linear_schedule_with_warmup

from dataset import MTCIRDataset
from logger import LOGGER
from models import CoLLM
from utils import collm_contrastive_collate_fn, log_vram, param_summary

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


def main():
    experiment_id = os.getenv("EXPERIMENT_ID", uuid.uuid4().hex[:8])

    LOGGER.info("=" * 60)
    LOGGER.info("EXPERIMENT ID: %s", experiment_id)
    LOGGER.info("=" * 60)

    PROCESSOR_NAME = os.getenv("PROCESSOR_NAME", "Qwen/Qwen3.5-0.8B")
    MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3.5-0.8B")
    PROJ_DIM = int(os.getenv("PROJ_DIM", 512))  # same as CLIP-B = P
    NUM_EMBS = int(os.getenv("NUM_EMBS", 4))  # num of target proposals = K
    HID_DIM = int(os.getenv("HID_DIM", 1024))
    KEEP_LAYERS = int(os.getenv("KEEP_LAYERS", 16))
    # Temperature for soft probe selection — lower = closer to hard argmax.
    # Can be annealed toward 0 over training for increasingly competitive probes.
    PROBE_TEMP = float(os.getenv("PROBE_TEMP", 1))
    # Temperature for InfoNCE contrastive loss.
    # 0.07 (CLIP default) is aggressive early in training; 0.1 is safer to start.
    INFONCE_TEMP = float(os.getenv("INFONCE_TEMP", 0.1))
    DIVERSITY_WEIGHT = float(os.getenv("DIVERSITY_WEIGHT", 0.1))
    EPOCHS = int(os.getenv("EPOCHS", 1))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", 64))  # = B
    NUM_WORKERS = int(os.getenv("NUM_WORKERS", 8))
    NUM_BATCHES = int(os.getenv("NUM_BATCHES", (1024 * 128) // BATCH_SIZE))

    LOGGER.info("=" * 60)
    LOGGER.info("HYPERPARAMETERS")
    LOGGER.info("=" * 60)
    LOGGER.info("  %-20s %s", "PROCESSOR_NAME:", PROCESSOR_NAME)
    LOGGER.info("  %-20s %s", "MODEL_NAME:", MODEL_NAME)
    LOGGER.info("  %-20s %s", "PROJ_DIM:", PROJ_DIM)
    LOGGER.info("  %-20s %s", "NUM_EMBS:", NUM_EMBS)
    LOGGER.info("  %-20s %s", "HID_DIM:", HID_DIM)
    LOGGER.info("  %-20s %s", "KEEP_LAYERS:", KEEP_LAYERS)
    LOGGER.info("  %-20s %s", "PROBE_TEMP:", PROBE_TEMP)
    LOGGER.info("  %-20s %s", "INFONCE_TEMP:", INFONCE_TEMP)
    LOGGER.info("  %-20s %s", "DIVERSITY_WEIGHT:", DIVERSITY_WEIGHT)
    LOGGER.info("  %-20s %s", "EPOCHS:", EPOCHS)
    LOGGER.info("  %-20s %s", "BATCH_SIZE:", BATCH_SIZE)
    LOGGER.info("  %-20s %s", "NUM_WORKERS:", NUM_WORKERS)
    LOGGER.info("  %-20s %s", "NUM_BATCHES:", NUM_BATCHES)
    LOGGER.info("=" * 60)

    LOGGER.info("Loading processor: %s", PROCESSOR_NAME)
    processor = AutoProcessor.from_pretrained(PROCESSOR_NAME, trust_remote_code=True)
    LOGGER.info("Processor loaded")

    LOGGER.info("Loading model: %s", MODEL_NAME)
    model = CoLLM(
        model_name=MODEL_NAME,
        projection_dim=PROJ_DIM,
        num_embeddings=NUM_EMBS,
        hidden_dim=HID_DIM,
        keep_layers=KEEP_LAYERS,
    )
    model = model.to(device)
    # model = torch.compile(model, mode="reduce-overhead")
    param_summary(model, LOGGER)
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
        "Creating DataLoader batch_size=%d, num_workers=%d", BATCH_SIZE, NUM_WORKERS
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
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
        num_training_steps=NUM_BATCHES * 5,  # so only decays lr to 80%
    )

    LOGGER.info("Starting training for %d epoch(s)", EPOCHS)
    LOGGER.info("Training on total exmaples: %d", NUM_BATCHES * BATCH_SIZE)

    skipped = 0
    for epoch in range(EPOCHS):
        LOGGER.info("Running epoch %d/%d", epoch + 1, EPOCHS)
        pbar = tqdm(
            total=NUM_BATCHES,
            file=sys.stdout,
            leave=False,
        )
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= NUM_BATCHES:
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
            probe_weights = torch.softmax(per_probe_sim / PROBE_TEMP, dim=1)  # (B, K)
            best_emb = torch.einsum(
                "bk,bkp->bp", probe_weights, embeddings.float()
            )  # (B, P) — still unit-norm after softmax combination (approx)
            best_emb = F.normalize(best_emb, dim=-1)  # re-normalise to be exact

            # --- symmetric InfoNCE loss ---
            # Logit matrix: each query's composed embedding vs every target in batch.
            # Diagonal entries are the positives.
            logits = torch.matmul(best_emb, target_emb.T) / INFONCE_TEMP  # (B, B)
            labels = torch.arange(logits.size(0), device=device)  # (B,)

            # Regularise cls_probes directly — they live in a fixed hidden_dim space
            probe_gram = torch.mm(
                F.normalize(model.cls_probes.float(), dim=-1),
                F.normalize(model.cls_probes.float(), dim=-1).T,
            )  # (K, K)
            off_diag = probe_gram.masked_fill(
                torch.eye(NUM_EMBS, device=device, dtype=torch.bool), 0.0
            )
            diversity_loss = (off_diag**2).sum()

            loss = (
                F.cross_entropy(logits, labels)  # query -> target
                + F.cross_entropy(logits.T, labels)  # target -> query
            ) / 2 + DIVERSITY_WEIGHT * diversity_loss

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
                log_vram(f"epoch={epoch + 1} batch={batch_idx + 1}", device)
        pbar.close()
        log_vram(f"epoch={epoch + 1} end", device)

    LOGGER.info("saving model to CoLLM.pt")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"collm_4probes_{timestamp}.pt"
    torch.save(model.state_dict(), model_filename)
    LOGGER.info(f"Model saved as: {model_filename}")


if __name__ == "__main__":
    main()

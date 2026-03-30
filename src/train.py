import os
import sys
import uuid
from datetime import datetime

import torch
import wandb
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoProcessor
from transformers.optimization import get_linear_schedule_with_warmup

from dataset import MTCIRDataset
from logger import LOGGER
from models import CoLLM
from utils import (
    collm_contrastive_collate_fn,
    find_latest_checkpoint,
    get_probe_temp,
    log_vram,
    param_summary,
    save_checkpoint,
)

CHECKPOINT_INTERVAL = 1000  # save a checkpoint every N batches
CHECKPOINT_DIR = "./checkpoints"

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


class EmbeddingQueue:
    def __init__(self, size, dim, device):
        self.queue = torch.randn(size, dim, device=device)
        self.queue = F.normalize(self.queue, dim=-1)
        self.ptr = 0
        self.size = size
        self.full = False

    @torch.no_grad()
    def enqueue(self, embeddings):
        # embeddings: (B, P)
        B = embeddings.size(0)
        end = (self.ptr + B) % self.size
        if end > self.ptr:
            self.queue[self.ptr : end] = embeddings.detach()
        else:  # wrap-around
            self.queue[self.ptr :] = embeddings.detach()[: self.size - self.ptr]
            self.queue[:end] = embeddings.detach()[self.size - self.ptr :]
        self.ptr = end
        if self.ptr == 0:
            self.full = True

    def get(self):
        # returns only the filled portion early in training
        if self.full:
            return self.queue.clone()
        return self.queue[: self.ptr].clone()


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
    K_HARD = int(os.getenv("K_HARD", 64))

    # ------------------------------------------------------------------
    # Checkpoint resume: look for an existing checkpoint only when
    # EXPERIMENT_ID was explicitly provided by the caller.  A freshly
    # generated UUID never has a matching checkpoint.
    # ------------------------------------------------------------------
    resume_epoch = 0
    resume_batch = 0  # first batch_idx to process in the resumed epoch
    skipped = 0

    resume_ckpt_path = None
    if os.getenv("EXPERIMENT_ID"):  # only attempt resume on explicit IDs
        resume_ckpt_path = find_latest_checkpoint(experiment_id, CHECKPOINT_DIR)

    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="anishchaudhary2706-indian-institute-of-science",
        # Set the wandb project where this run will be logged.
        project="collm",
        resume="allow" if resume_ckpt_path else None,
        # Track hyperparameters and run metadata.
        config={
            # Identifiers
            "experiment_id": experiment_id,
            # Model
            "processor_name": PROCESSOR_NAME,
            "model_name": MODEL_NAME,
            "keep_layers": KEEP_LAYERS,
            # Architecture
            "proj_dim": PROJ_DIM,
            "num_embs": NUM_EMBS,
            "hid_dim": HID_DIM,
            # Training
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "num_workers": NUM_WORKERS,
            "num_batches": NUM_BATCHES,
            # Loss & Temperature
            "probe_temp": PROBE_TEMP,
            "infonce_temp": INFONCE_TEMP,
            "diversity_weight": DIVERSITY_WEIGHT,
            "k_hard": K_HARD,
        },
    )

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
    LOGGER.info("  %-20s %s", "K_HARD:", K_HARD)
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

    model.model.model.language_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    LOGGER.info(
        "Creating dataset from %s", "./MTCIR/mtcir_expanded_shuffled_safe.jsonl"
    )
    train_dataset = MTCIRDataset(
        "./MTCIR/mtcir_expanded_shuffled.jsonl",
        "./images",
        "./embeddings",
        LOGGER=LOGGER,
    )
    LOGGER.info("Dataset ready with %d samples", len(train_dataset))
    LOGGER.info(
        "Creating DataLoader batch_size=%d, num_workers=%d", BATCH_SIZE, NUM_WORKERS
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        multiprocessing_context="fork",
        pin_memory=True,
        prefetch_factor=2,
        collate_fn=collm_contrastive_collate_fn,
        shuffle=False,
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
    # ------------------------------------------------------------------
    # Restore checkpoint state (model weights, optimizer, scheduler, etc.)
    # Must happen *after* model + optimizer + scheduler are constructed.
    # ------------------------------------------------------------------
    skipped = 0
    if resume_ckpt_path:
        LOGGER.info("Resuming from checkpoint: %s", resume_ckpt_path)
        ckpt = torch.load(resume_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        resume_epoch = ckpt["epoch"]
        # resume_batch is the *next* batch to run, i.e. one past the saved one
        resume_batch = ckpt["batch_idx"] + 1
        skipped = ckpt.get("skipped", 0)
        LOGGER.info(
            "Resumed state: epoch=%d, next_batch=%d, skipped_so_far=%d",
            resume_epoch,
            resume_batch,
            skipped,
        )
    else:
        LOGGER.info("Starting fresh training run")

    LOGGER.info("Starting training for %d epoch(s)", EPOCHS)
    LOGGER.info("Training on total exmaples: %d", NUM_BATCHES * BATCH_SIZE)

    for epoch in range(EPOCHS):
        LOGGER.info("Running epoch %d/%d", epoch + 1, EPOCHS)
        # On the resumed epoch we fast-forward the dataloader by skipping
        # batches that were already processed.  On all subsequent epochs we
        # start from batch 0 as normal.
        start_batch = resume_batch if epoch == resume_epoch else 0
        if start_batch > 0:
            LOGGER.info(
                "Fast-forwarding dataloader: skipping first %d batches", start_batch
            )

        pbar = tqdm(
            total=NUM_BATCHES - start_batch,
            file=sys.stdout,
            leave=False,
        )
        loader_iter = iter(train_loader)

        queue = EmbeddingQueue(size=4096, dim=PROJ_DIM, device=device)

        # Consume (but don't train on) the already-processed batches.
        for _ in range(start_batch):
            try:
                next(loader_iter)
            except StopIteration:
                break
        for batch_idx, batch in enumerate(loader_iter):
            if batch is None:
                continue
            if batch_idx >= NUM_BATCHES:
                break
            model.train()
            optimizer.zero_grad()

            # forwars pass
            with torch.autocast("cuda", dtype=torch.bfloat16):
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
                target_emb = torch.stack(
                    [torch.from_numpy(e) for e in batch["target_image_emb"]]
                ).to(device)  # (B, P)
                target_emb = F.normalize(target_emb, dim=-1)  # (B, P)

                # --- soft probe selection (fully differentiable) ---
                # input-conditioned routing (query-side only, no target)
                current_probe_temp = get_probe_temp(
                    batch_idx + start_batch,
                    NUM_BATCHES * EPOCHS,
                    temp_start=PROBE_TEMP,
                    temp_end=0.1,
                )
                probe_logits = model.probe_router(embeddings.mean(dim=1))  # (B, K)
                probe_weights = torch.softmax(
                    probe_logits / current_probe_temp, dim=1
                )  # (B, K)

                best_emb = torch.einsum(
                    "bk,bkp->bp", probe_weights, embeddings
                )  # (B, P) — still unit-norm after softmax combination (approx)
                best_emb = F.normalize(best_emb, dim=-1)  # re-normalise to be exact

                # ----------------------------------------------------------------
                # Enqueue *before* computing loss so current batch can also act
                # as negatives for itself via the in-batch term below.
                # Must happen before `del target_emb`.
                # ----------------------------------------------------------------
                queue.enqueue(target_emb)  # stop-gradient, detach inside

                # ----------------------------------------------------------------
                # Hard negative mining from queue
                # ----------------------------------------------------------------
                queue_negs = queue.get()  # (Q, P)  — detached, no grad
                use_hard_negs = queue_negs.size(0) >= K_HARD

                hard_neg_sim = None
                hard_neg_indices = None
                neg_sim = None

                if use_hard_negs:
                    # Similarity of each query to every queued target.
                    neg_sim = torch.matmul(best_emb, queue_negs.T)  # (B, Q)

                    # Top-K most similar (hardest) queue entries per query.
                    hard_neg_indices = neg_sim.topk(
                        K_HARD, dim=1
                    ).indices  # (B, K_HARD)
                    # hard_negs = queue_negs[hard_neg_indices]  # (B, K_HARD, P)

                    # (B, K_HARD) similarity scores — no extra matmul needed.
                    hard_neg_sim = neg_sim.gather(1, hard_neg_indices)  # (B, K_HARD)

                # ----------------------------------------------------------------
                # Build augmented logit matrix
                #
                # Row layout per query:  [positive | in-batch negatives | hard-queue negatives]
                #   col 0          : dot(query_i, target_i)           ← the positive
                #   cols 1..B      : dot(query_i, target_j), j≠i      ← in-batch negatives
                #   cols B+1..B+K_HARD : hard queue negatives          ← only if queue ready
                # ----------------------------------------------------------------
                # Positive similarity: (B, 1)
                pos_sim = (best_emb * target_emb).sum(dim=-1, keepdim=True)

                # In-batch negatives: mask out the diagonal (those are positives).
                inbatch_sim = torch.matmul(best_emb, target_emb.T)  # (B, B)
                diag_mask = torch.eye(
                    inbatch_sim.size(0), dtype=torch.bool, device=device
                )
                inbatch_neg_sim = inbatch_sim.masked_fill(diag_mask, -1e4)  # (B, B)

                cats = [pos_sim, inbatch_neg_sim]
                if use_hard_negs and hard_neg_sim is not None:
                    cats.append(hard_neg_sim)
                full_logits = torch.cat(cats, dim=1) / INFONCE_TEMP  # (B, 1+B[+K_HARD])

                # Label 0 = the positive is always the first column.
                hard_labels = torch.zeros(
                    full_logits.size(0), dtype=torch.long, device=device
                )
                loss_contrastive = F.cross_entropy(full_logits, hard_labels)

                # Regularise cls_probes directly — they live in a fixed hidden_dim space
                # calulates sum of square of cosine similarity between probes
                probe_gram = torch.mm(
                    F.normalize(model.cls_probes, dim=-1),
                    F.normalize(model.cls_probes, dim=-1).T,
                )  # (K, K)
                off_diag = probe_gram.masked_fill(
                    torch.eye(NUM_EMBS, device=device, dtype=torch.bool), 0.0
                )
                diversity_loss = (off_diag**2).sum()

                # diversity in output embedding space across the batch
                # embeddings: (B, K, P)
                mean_probes = F.normalize(embeddings.mean(dim=0), dim=-1)  # (K, P)
                output_gram = torch.mm(mean_probes, mean_probes.T)  # (K, K)
                output_off_diag = output_gram.masked_fill(
                    torch.eye(NUM_EMBS, device=device, dtype=torch.bool), 0.0
                )
                output_diversity_loss = (output_off_diag**2).sum()

                loss = loss_contrastive + DIVERSITY_WEIGHT * (
                    diversity_loss + output_diversity_loss
                )

                del target_emb, best_emb, embeddings, full_logits
                del inbatch_sim, inbatch_neg_sim, pos_sim
                if use_hard_negs:
                    del hard_neg_sim, hard_neg_indices, neg_sim
                torch.cuda.empty_cache()

            # backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            pbar.update(1)

            run.log(
                {
                    "batch_idx": batch_idx + 1,
                    "loss": loss.item(),
                    "lr": scheduler.get_last_lr()[0],
                    "probe_temp": current_probe_temp,
                }
            )

            # ----------------------------------------------------------
            # Periodic checkpoint (every CHECKPOINT_INTERVAL batches)
            # batch_idx is 0-based, so we trigger at 999, 1999, …
            # ----------------------------------------------------------
            if (batch_idx + 1) % CHECKPOINT_INTERVAL == 0:
                save_checkpoint(
                    experiment_id,
                    epoch,
                    batch_idx,
                    model,
                    optimizer,
                    scheduler,
                    skipped,
                    CHECKPOINT_DIR,
                    LOGGER,
                )

            if batch_idx % 10 == 0:
                LOGGER.info(
                    "Epoch=%d batch=%d | loss=%.4f | lr=%.2e\n",
                    epoch + 1,
                    batch_idx + 1,
                    loss.item(),
                    scheduler.get_last_lr()[0],
                )
                log_vram(f"epoch={epoch + 1} batch={batch_idx + 1}", LOGGER, device)
        pbar.close()
        log_vram(f"epoch={epoch + 1} end", LOGGER, device)

        # Also checkpoint at the end of each epoch so epoch boundaries are
        # always recoverable even if CHECKPOINT_INTERVAL doesn't land there.
        save_checkpoint(
            experiment_id,
            epoch,
            NUM_BATCHES - 1,
            model,
            optimizer,
            scheduler,
            skipped,
            CHECKPOINT_DIR,
            LOGGER,
        )

    LOGGER.info("saving model to CoLLM.pt")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"collm_4probes_{experiment_id}_{timestamp}.pt"
    torch.save(model.state_dict(), model_filename)
    LOGGER.info(f"Model saved as: {model_filename}")

    run.finish()


if __name__ == "__main__":
    main()

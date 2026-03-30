import os
import sys
import uuid
from datetime import datetime

import torch
import torch.distributed as dist
import wandb
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
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

# FIX 1 (OOM): Tell PyTorch's allocator to use expandable segments
# to reduce fragmentation before we even start.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------


def setup_ddp():
    """Initialise the default process group (called once per rank)."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    dist.destroy_process_group()


def is_main_process():
    return dist.get_rank() == 0


def all_gather_tensor(t: torch.Tensor) -> torch.Tensor:
    """Gather a tensor from all ranks and concatenate along dim 0.

    Gradients are NOT propagated through the gathered copies from remote ranks,
    but the local copy retains its grad_fn — which is what we want for the
    in-batch term on the local rank.
    """
    world = dist.get_world_size()
    gathered = [torch.zeros_like(t) for _ in range(world)]
    dist.all_gather(gathered, t.detach())  # detach remote copies
    # Replace local rank's copy with the original (grad-carrying) tensor
    rank = dist.get_rank()
    gathered[rank] = t
    return torch.cat(gathered, dim=0)  # (world*B, ...)


# ---------------------------------------------------------------------------
# Embedding queue (unchanged logic, now fed with all-gathered embeddings)
# ---------------------------------------------------------------------------


class EmbeddingQueue:
    def __init__(self, size, dim, device):
        self.queue = torch.randn(size, dim, device=device)
        self.queue = F.normalize(self.queue, dim=-1)
        self.ptr = 0
        self.size = size
        self.full = False

    @torch.no_grad()
    def enqueue(self, embeddings):
        # embeddings: (B_global, P)  — all-gathered before calling this
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
        if self.full:
            return self.queue.clone()
        return self.queue[: self.ptr].clone()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    # ---- DDP setup --------------------------------------------------------
    local_rank = setup_ddp()
    device = f"cuda:{local_rank}"
    world_size = dist.get_world_size()
    LOGGER.info(f"[rank {dist.get_rank()}] device={device}  world_size={world_size}")

    # ---- Hyperparameters --------------------------------------------------
    experiment_id = os.getenv("EXPERIMENT_ID", uuid.uuid4().hex[:8])

    # Synchronise experiment_id across all ranks so every rank uses the same
    # checkpoint directory even when EXPERIMENT_ID was not set externally.
    id_tensor = torch.zeros(8, dtype=torch.uint8, device=device)
    if is_main_process():
        id_bytes = experiment_id.encode()[:8]
        for i, b in enumerate(id_bytes):
            id_tensor[i] = b
    dist.broadcast(id_tensor, src=0)
    experiment_id = bytes(id_tensor.tolist()).rstrip(b"\x00").decode()

    if is_main_process():
        LOGGER.info("=" * 60)
        LOGGER.info("EXPERIMENT ID: %s", experiment_id)
        LOGGER.info("=" * 60)

    PROCESSOR_NAME = os.getenv("PROCESSOR_NAME", "Qwen/Qwen3.5-0.8B")
    MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3.5-0.8B")
    PROJ_DIM = int(os.getenv("PROJ_DIM", 512))
    NUM_EMBS = int(os.getenv("NUM_EMBS", 4))
    HID_DIM = int(os.getenv("HID_DIM", 1024))
    KEEP_LAYERS = int(os.getenv("KEEP_LAYERS", 16))
    PROBE_TEMP = float(os.getenv("PROBE_TEMP", 1))
    INFONCE_TEMP = float(os.getenv("INFONCE_TEMP", 0.1))
    DIVERSITY_WEIGHT = float(os.getenv("DIVERSITY_WEIGHT", 0.1))
    EPOCHS = int(os.getenv("EPOCHS", 1))
    # BATCH_SIZE is *per-GPU*. The effective global batch = BATCH_SIZE * world_size.
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", 64))
    NUM_WORKERS = int(os.getenv("NUM_WORKERS", 8))
    NUM_BATCHES = int(os.getenv("NUM_BATCHES", (1024 * 128) // BATCH_SIZE))
    K_HARD = int(os.getenv("K_HARD", 64))

    # ---- Checkpoint resume ------------------------------------------------
    resume_epoch = 0
    resume_batch = 0
    skipped = 0
    resume_ckpt_path = None
    if os.getenv("EXPERIMENT_ID"):
        resume_ckpt_path = find_latest_checkpoint(experiment_id, CHECKPOINT_DIR)

    # ---- wandb (main rank only) ------------------------------------------
    if is_main_process():
        run = wandb.init(
            entity="anishchaudhary2706-indian-institute-of-science",
            project="collm",
            resume="allow" if resume_ckpt_path else None,
            config={
                "experiment_id": experiment_id,
                "processor_name": PROCESSOR_NAME,
                "model_name": MODEL_NAME,
                "keep_layers": KEEP_LAYERS,
                "proj_dim": PROJ_DIM,
                "num_embs": NUM_EMBS,
                "hid_dim": HID_DIM,
                "epochs": EPOCHS,
                "batch_size_per_gpu": BATCH_SIZE,
                "global_batch_size": BATCH_SIZE * world_size,
                "world_size": world_size,
                "num_workers": NUM_WORKERS,
                "num_batches": NUM_BATCHES,
                "probe_temp": PROBE_TEMP,
                "infonce_temp": INFONCE_TEMP,
                "diversity_weight": DIVERSITY_WEIGHT,
                "k_hard": K_HARD,
            },
        )
        LOGGER.info("=" * 60)
        LOGGER.info("HYPERPARAMETERS")
        LOGGER.info("=" * 60)
        LOGGER.info("  %-25s %s", "PROCESSOR_NAME:", PROCESSOR_NAME)
        LOGGER.info("  %-25s %s", "MODEL_NAME:", MODEL_NAME)
        LOGGER.info("  %-25s %s", "PROJ_DIM:", PROJ_DIM)
        LOGGER.info("  %-25s %s", "NUM_EMBS:", NUM_EMBS)
        LOGGER.info("  %-25s %s", "HID_DIM:", HID_DIM)
        LOGGER.info("  %-25s %s", "KEEP_LAYERS:", KEEP_LAYERS)
        LOGGER.info("  %-25s %s", "PROBE_TEMP:", PROBE_TEMP)
        LOGGER.info("  %-25s %s", "INFONCE_TEMP:", INFONCE_TEMP)
        LOGGER.info("  %-25s %s", "DIVERSITY_WEIGHT:", DIVERSITY_WEIGHT)
        LOGGER.info("  %-25s %s", "K_HARD:", K_HARD)
        LOGGER.info("  %-25s %s", "EPOCHS:", EPOCHS)
        LOGGER.info("  %-25s %s", "BATCH_SIZE (per GPU):", BATCH_SIZE)
        LOGGER.info("  %-25s %s", "GLOBAL BATCH SIZE:", BATCH_SIZE * world_size)
        LOGGER.info("  %-25s %s", "WORLD_SIZE:", world_size)
        LOGGER.info("  %-25s %s", "NUM_WORKERS:", NUM_WORKERS)
        LOGGER.info("  %-25s %s", "NUM_BATCHES:", NUM_BATCHES)
        LOGGER.info("=" * 60)

    # ---- Processor --------------------------------------------------------
    LOGGER.info("[rank %d] Loading processor: %s", dist.get_rank(), PROCESSOR_NAME)
    processor = AutoProcessor.from_pretrained(PROCESSOR_NAME, trust_remote_code=True)

    # ---- Model ------------------------------------------------------------
    LOGGER.info("[rank %d] Loading model: %s", dist.get_rank(), MODEL_NAME)
    model = CoLLM(
        model_name=MODEL_NAME,
        projection_dim=PROJ_DIM,
        num_embeddings=NUM_EMBS,
        hidden_dim=HID_DIM,
        keep_layers=KEEP_LAYERS,
    )
    model = model.to(device)
    if is_main_process():
        param_summary(model, LOGGER)

    model.model.model.language_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    # Wrap with DDP.  find_unused_parameters=False is faster when all params
    # are used every forward pass (adjust if your architecture has branches).
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # ---- Dataset & distributed sampler -----------------------------------
    LOGGER.info("[rank %d] Creating dataset", dist.get_rank())
    train_dataset = MTCIRDataset(
        "./MTCIR/mtcir_expanded_shuffled.jsonl",
        "./images",
        "./embeddings",
        LOGGER=LOGGER,
    )
    # DistributedSampler ensures each rank sees a non-overlapping shard.
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=dist.get_rank(),
        shuffle=False,
        drop_last=True,
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,  # per-GPU batch size
        sampler=train_sampler,  # replaces shuffle=True/False
        num_workers=NUM_WORKERS,
        multiprocessing_context="fork",
        pin_memory=True,
        prefetch_factor=2,
        collate_fn=collm_contrastive_collate_fn,
    )

    # ---- Optimizer & scheduler -------------------------------------------
    LOGGER.info("[rank %d] Initializing optimizer", dist.get_rank())
    # Access underlying module params through the DDP wrapper.
    trainable_params = [p for p in model.module.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4, eps=1e-6)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=500,
        num_training_steps=NUM_BATCHES * 5,
    )

    # ---- Checkpoint restore -----------------------------------------------
    if resume_ckpt_path:
        LOGGER.info(
            "[rank %d] Resuming from checkpoint: %s", dist.get_rank(), resume_ckpt_path
        )
        ckpt = torch.load(resume_ckpt_path, map_location=device)
        model.module.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        resume_epoch = ckpt["epoch"]
        resume_batch = ckpt["batch_idx"] + 1
        skipped = ckpt.get("skipped", 0)
    else:
        if is_main_process():
            LOGGER.info("Starting fresh training run")

    LOGGER.info(
        "[rank %d] Starting training for %d epoch(s), global_batch=%d",
        dist.get_rank(),
        EPOCHS,
        BATCH_SIZE * world_size,
    )

    # ---- Training loop ----------------------------------------------------
    for epoch in range(EPOCHS):
        # Tell the sampler which epoch we're on (required for reproducibility
        # when shuffle=True; harmless when shuffle=False).
        train_sampler.set_epoch(epoch)

        start_batch = resume_batch if epoch == resume_epoch else 0
        if start_batch > 0 and is_main_process():
            LOGGER.info(
                "Fast-forwarding dataloader: skipping first %d batches", start_batch
            )

        pbar = tqdm(
            total=NUM_BATCHES - start_batch,
            file=sys.stdout,
            leave=False,
            disable=not is_main_process(),  # only rank-0 shows the bar
        )
        loader_iter = iter(train_loader)
        queue = EmbeddingQueue(size=4096, dim=PROJ_DIM, device=device)

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

            with torch.autocast("cuda", dtype=torch.bfloat16):
                embeddings = model(
                    images=batch["image"],
                    text=batch["modification_text"],
                    processor=processor,
                )  # (B_local, K, P)

                if torch.isnan(embeddings).any() or torch.isinf(embeddings).any():
                    skipped += 1
                    LOGGER.warning(
                        "[rank %d] Epoch=%d batch=%d | bad embeddings, skipping [total=%d]",
                        dist.get_rank(),
                        epoch + 1,
                        batch_idx + 1,
                        skipped,
                    )
                    scheduler.step()
                    pbar.update(1)
                    continue

                # ---- target embeddings ------------------------------------
                target_emb_local = torch.stack(
                    [torch.from_numpy(e) for e in batch["target_image_emb"]]
                ).to(device)  # (B_local, P)
                target_emb_local = F.normalize(target_emb_local, dim=-1)

                # All-gather targets across GPUs so every rank has the full
                # global batch of negatives — this is the key DDP change for
                # contrastive learning.
                target_emb_global = all_gather_tensor(target_emb_local)  # (B_global, P)

                # ---- probe routing ----------------------------------------
                current_probe_temp = get_probe_temp(
                    batch_idx + start_batch,
                    NUM_BATCHES * EPOCHS,
                    temp_start=PROBE_TEMP,
                    temp_end=0.1,
                )
                probe_logits = model.module.probe_router(
                    embeddings.mean(dim=1)
                )  # (B_local, K)
                probe_weights = torch.softmax(probe_logits / current_probe_temp, dim=1)
                best_emb_local = torch.einsum("bk,bkp->bp", probe_weights, embeddings)
                best_emb_local = F.normalize(best_emb_local, dim=-1)  # (B_local, P)

                # All-gather query embeddings so the loss matrix is (B_global, B_global)
                best_emb_global = all_gather_tensor(best_emb_local)  # (B_global, P)

                # ---- queue (fed with global targets) ----------------------
                queue.enqueue(target_emb_global)
                queue_negs = queue.get()  # (Q, P) — detached
                use_hard_negs = queue_negs.size(0) >= K_HARD

                hard_neg_sim = None
                hard_neg_indices = None
                neg_sim = None

                if use_hard_negs:
                    neg_sim = torch.matmul(
                        best_emb_global, queue_negs.T
                    )  # (B_global, Q)
                    hard_neg_indices = neg_sim.topk(
                        K_HARD, dim=1
                    ).indices  # (B_global, K_HARD)
                    hard_neg_sim = neg_sim.gather(
                        1, hard_neg_indices
                    )  # (B_global, K_HARD)

                # ---- InfoNCE loss (global batch) --------------------------
                pos_sim = (best_emb_global * target_emb_global).sum(
                    dim=-1, keepdim=True
                )  # (B_global, 1)

                inbatch_sim = torch.matmul(
                    best_emb_global, target_emb_global.T
                )  # (B_global, B_global)
                B_global = inbatch_sim.size(0)
                diag_mask = torch.eye(B_global, dtype=torch.bool, device=device)
                inbatch_neg_sim = inbatch_sim.masked_fill(diag_mask, -1e4)

                cats = [pos_sim, inbatch_neg_sim]
                if use_hard_negs and hard_neg_sim is not None:
                    cats.append(hard_neg_sim)
                full_logits = (
                    torch.cat(cats, dim=1) / INFONCE_TEMP
                )  # (B_global, 1+B_global[+K_HARD])

                hard_labels = torch.zeros(B_global, dtype=torch.long, device=device)
                loss_contrastive = F.cross_entropy(full_logits, hard_labels)

                # ---- diversity loss ---------------------------------------
                probe_gram = torch.mm(
                    F.normalize(model.module.cls_probes, dim=-1),
                    F.normalize(model.module.cls_probes, dim=-1).T,
                )  # (K, K)
                off_diag = probe_gram.masked_fill(
                    torch.eye(NUM_EMBS, device=device, dtype=torch.bool), 0.0
                )
                diversity_loss = (off_diag**2).sum()

                # Output diversity loss — use global embeddings for consistency.
                # embeddings_global: (B_global, K, P)
                embeddings_global = all_gather_tensor(
                    embeddings.view(
                        -1, PROJ_DIM
                    )  # flatten K into batch dim temporarily
                ).view(B_global, NUM_EMBS, PROJ_DIM)

                mean_probes = F.normalize(
                    embeddings_global.mean(dim=0), dim=-1
                )  # (K, P)
                output_gram = torch.mm(mean_probes, mean_probes.T)
                output_off_diag = output_gram.masked_fill(
                    torch.eye(NUM_EMBS, device=device, dtype=torch.bool), 0.0
                )
                output_diversity_loss = (output_off_diag**2).sum()

                loss = loss_contrastive + DIVERSITY_WEIGHT * (
                    diversity_loss + output_diversity_loss
                )

                del target_emb_local, target_emb_global, best_emb_local, best_emb_global
                del embeddings, embeddings_global, full_logits
                del inbatch_sim, inbatch_neg_sim, pos_sim
                if use_hard_negs:
                    del hard_neg_sim, hard_neg_indices, neg_sim
                torch.cuda.empty_cache()

            # ---- backward ------------------------------------------------
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            pbar.update(1)

            # ---- logging (rank 0 only) -----------------------------------
            if is_main_process():
                run.log(
                    {
                        "batch_idx": batch_idx + 1,
                        "loss": loss.item(),
                        "lr": scheduler.get_last_lr()[0],
                        "probe_temp": current_probe_temp,
                    }
                )

            if batch_idx % 10 == 0 and is_main_process():
                LOGGER.info(
                    "Epoch=%d batch=%d | loss=%.4f | lr=%.2e",
                    epoch + 1,
                    batch_idx + 1,
                    loss.item(),
                    scheduler.get_last_lr()[0],
                )
                log_vram(f"epoch={epoch + 1} batch={batch_idx + 1}", LOGGER, device)

            # ---- checkpoint (rank 0 only) --------------------------------
            if (batch_idx + 1) % CHECKPOINT_INTERVAL == 0 and is_main_process():
                save_checkpoint(
                    experiment_id,
                    epoch,
                    batch_idx,
                    model.module,  # unwrap DDP before saving
                    optimizer,
                    scheduler,
                    skipped,
                    CHECKPOINT_DIR,
                    LOGGER,
                )

        pbar.close()
        if is_main_process():
            log_vram(f"epoch={epoch + 1} end", LOGGER, device)
            save_checkpoint(
                experiment_id,
                epoch,
                NUM_BATCHES - 1,
                model.module,
                optimizer,
                scheduler,
                skipped,
                CHECKPOINT_DIR,
                LOGGER,
            )

    # ---- Save final model (rank 0 only) ----------------------------------
    if is_main_process():
        LOGGER.info("Saving model to collm_*.pt")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"collm_4probes_{experiment_id}_{timestamp}.pt"
        torch.save(model.module.state_dict(), model_filename)
        LOGGER.info("Model saved as: %s", model_filename)
        run.finish()

    cleanup_ddp()


if __name__ == "__main__":
    main()

import json
import os
import sys
import uuid
from datetime import datetime

import torch
import torch.distributed as dist
import wandb
from accelerate import Accelerator
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoProcessor
from transformers.optimization import get_linear_schedule_with_warmup

from dataset import MTCIRDataset
from logger import LOGGER
from models import CoLLM
from utils import (
    NegativeQueue,
    collm_contrastive_collate_fn,
    find_latest_checkpoint,
    gather_with_grad,
    log_vram,
    multiprobe_infonce_loss,
    param_summary,
    run_circo_val,
    save_checkpoint,
)

CHECKPOINT_INTERVAL = 1000  # save a checkpoint every N batches
CHECKPOINT_DIR = "./checkpoints"

# Tell PyTorch's allocator to use expandable segments to reduce fragmentation.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")


def main():
    accelerator = Accelerator(mixed_precision="bf16")
    device = accelerator.device

    if accelerator.is_main_process:
        LOGGER.info(f"accelerator: {accelerator.state}")

    experiment_id = os.getenv("EXPERIMENT_ID", uuid.uuid4().hex[:8])

    if accelerator.is_main_process:
        LOGGER.info("=" * 60)
        LOGGER.info("EXPERIMENT ID: %s", experiment_id)
        LOGGER.info("=" * 60)

    PROCESSOR_NAME = os.getenv("PROCESSOR_NAME", "Qwen/Qwen3.5-0.8B")
    MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3.5-0.8B")
    PROJ_DIM = int(os.getenv("PROJ_DIM", 512))  # same as CLIP-B = P
    NUM_EMBS = int(os.getenv("NUM_EMBS", 4))  # num of target proposals = K
    HID_DIM = int(os.getenv("HID_DIM", 1024))
    KEEP_LAYERS = int(os.getenv("KEEP_LAYERS", 16))
    # Temperature for InfoNCE contrastive loss.
    # 0.07 (CLIP default) is aggressive early in training; 0.1 is safer to start.
    INFONCE_TEMP = float(os.getenv("INFONCE_TEMP", 0.1))
    DIVERSITY_WEIGHT = float(os.getenv("DIVERSITY_WEIGHT", 0.1))
    EPOCHS = int(os.getenv("EPOCHS", 1))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))  # per-GPU batch size
    NUM_WORKERS = int(os.getenv("NUM_WORKERS", 8))
    NUM_BATCHES = int(os.getenv("NUM_BATCHES", (1024 * 128) // BATCH_SIZE))

    QUEUE_SIZE = int(os.getenv("QUEUE_SIZE", 4096))  # negatives stored in queue
    K_HARD = int(os.getenv("K_HARD", 64))  # hard negatives per query (0=disabled)

    VAL_INTERVAL = int(os.getenv("VAL_INTERVAL", 500))
    CIRCO_VAL_ANNOTATIONS = os.getenv("CIRCO_VAL_ANNOTATIONS")  # path to val.json
    CIRCO_COCO_IMG_DIR = os.getenv("CIRCO_COCO_IMG_DIR")  # path to unlabeled2017/
    CIRCO_GALLERY_CACHE = os.getenv(
        "CIRCO_GALLERY_CACHE",
        "/home/anirban/anishc/CoLLM-implementation-pytorch/clip_unlabeled2017_cache.pt",
    )

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

    run = None
    if accelerator.is_main_process:
        run = wandb.init(
            entity="anishchaudhary2706-indian-institute-of-science",
            project="collm",
            resume="allow" if resume_ckpt_path else None,
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
                "num_gpus": accelerator.num_processes,
                "num_workers": NUM_WORKERS,
                "num_batches": NUM_BATCHES,
                # Loss & Temperature
                "infonce_temp": INFONCE_TEMP,
                "diversity_weight": DIVERSITY_WEIGHT,
                # Hard sampling
                "queue_size": QUEUE_SIZE,
                "k_hard": K_HARD,
            },
        )

    if accelerator.is_main_process:
        LOGGER.info("=" * 60)
        LOGGER.info("HYPERPARAMETERS")
        LOGGER.info("=" * 60)
        LOGGER.info("  %-20s %s", "PROCESSOR_NAME:", PROCESSOR_NAME)
        LOGGER.info("  %-20s %s", "MODEL_NAME:", MODEL_NAME)
        LOGGER.info("  %-20s %s", "PROJ_DIM:", PROJ_DIM)
        LOGGER.info("  %-20s %s", "NUM_EMBS:", NUM_EMBS)
        LOGGER.info("  %-20s %s", "HID_DIM:", HID_DIM)
        LOGGER.info("  %-20s %s", "KEEP_LAYERS:", KEEP_LAYERS)
        LOGGER.info("  %-20s %s", "INFONCE_TEMP:", INFONCE_TEMP)
        LOGGER.info("  %-20s %s", "DIVERSITY_WEIGHT:", DIVERSITY_WEIGHT)
        LOGGER.info("  %-20s %s", "QUEUE_SIZE:", QUEUE_SIZE)
        LOGGER.info("  %-20s %s", "K_HARD:", K_HARD)
        LOGGER.info("  %-20s %s", "EPOCHS:", EPOCHS)
        LOGGER.info("  %-20s %s", "BATCH_SIZE (per GPU):", BATCH_SIZE)
        LOGGER.info("  %-20s %s", "NUM_GPUS:", accelerator.num_processes)
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
        device=device,
    )
    # model = torch.compile(model, mode="reduce-overhead")
    if accelerator.is_main_process:
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

    # ------------------------------------------------------------------
    # Restore checkpoint state (model weights, optimizer, scheduler, etc.)
    # Must happen *before* accelerator.prepare so we load into the raw model.
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

    # ------------------------------------------------------------------
    # Wrap model, optimizer, dataloader, scheduler with accelerate.
    # This enables DDP across all available GPUs.
    # ------------------------------------------------------------------
    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )
    # raw_model gives direct access to CoLLM attributes (cls_probes, etc.)
    # without going through the DDP wrapper.
    raw_model = accelerator.unwrap_model(model)

    # Negative queue — stores gathered target CLIP embeddings from recent batches.
    # No momentum encoder needed: target_emb is already frozen CLIP (no grad).
    # All ranks enqueue the same all_target_emb, so queues stay in sync.
    neg_queue = NegativeQueue(QUEUE_SIZE, PROJ_DIM, device)

    # ------------------------------------------------------------------
    # CIRCO validation setup — load gallery + val queries once up front.
    # Gated on env vars so training still works without them.
    # ------------------------------------------------------------------
    val_queries = None
    gallery_embs_val = None
    coco_ids_val = None
    if accelerator.is_main_process and CIRCO_VAL_ANNOTATIONS and CIRCO_COCO_IMG_DIR:
        if os.path.exists(CIRCO_GALLERY_CACHE):
            LOGGER.info("Loading CIRCO gallery cache from %s", CIRCO_GALLERY_CACHE)
            data = torch.load(CIRCO_GALLERY_CACHE, map_location=device)
            gallery_embs_val = data["embs"].to(device)
            coco_ids_val = data["ids"]
            LOGGER.info("Gallery loaded: %d images", len(coco_ids_val))
        else:
            LOGGER.warning(
                "CIRCO_GALLERY_CACHE not found at %s — skipping validation",
                CIRCO_GALLERY_CACHE,
            )

        if gallery_embs_val is not None and os.path.exists(CIRCO_VAL_ANNOTATIONS):
            with open(CIRCO_VAL_ANNOTATIONS) as f:
                annotations = json.load(f)
            val_queries = [
                {
                    "query_id": str(ann["id"]),
                    "image_path": os.path.join(
                        CIRCO_COCO_IMG_DIR, f"{ann['reference_img_id']:012d}.jpg"
                    ),
                    "modification_text": ann["relative_caption"],
                    "gt_img_ids": ann["gt_img_ids"],
                }
                for ann in annotations
            ]
            LOGGER.info(
                "CIRCO val: %d queries loaded from %s",
                len(val_queries),
                CIRCO_VAL_ANNOTATIONS,
            )
        elif gallery_embs_val is not None:
            LOGGER.warning(
                "CIRCO_VAL_ANNOTATIONS not found at %s — skipping validation",
                CIRCO_VAL_ANNOTATIONS,
            )

    LOGGER.info("Starting training for %d epoch(s)", EPOCHS)
    LOGGER.info("Training on total examples: %d", NUM_BATCHES * BATCH_SIZE)

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
            disable=not accelerator.is_main_process,
        )
        loader_iter = iter(train_loader)

        # Consume (but don't train on) the already-processed batches.
        for _ in range(start_batch):
            try:
                next(loader_iter)
            except StopIteration:
                break
        for batch_idx, batch in enumerate(loader_iter):
            if batch is None:
                continue

            # Synchronise batch sizes across ranks before any all-gather.
            # collm_contrastive_collate_fn drops failed samples, which can leave
            # ranks with unequal batch sizes — causing NCCL ALLGATHER to deadlock.
            if accelerator.num_processes > 1:
                local_size = torch.tensor(len(batch["id"]), device=device)
                dist.all_reduce(local_size, op=dist.ReduceOp.MIN)
                min_size = int(local_size.item())
                if min_size == 0:
                    continue
                if min_size < len(batch["id"]):
                    batch = {k: v[:min_size] for k, v in batch.items()}

            if batch_idx >= NUM_BATCHES:
                break
            model.train()
            optimizer.zero_grad()

            # forward pass
            with torch.autocast("cuda", dtype=torch.bfloat16):
                embeddings = model.forward(
                    images=batch["image"],
                    text=batch["modification_text"],
                    processor=processor,
                )  # (B, K, P)

                # Guard: skip batch if backbone produced inf/nan hidden states.
                # This can happen early in training when fp16 attention overflows.
                # Synchronize the decision across all ranks: if any rank sees bad
                # embeddings, all ranks must skip together to avoid hanging at the
                # subsequent all_gather collectives (gather_with_grad / accelerator.gather).
                has_bad = torch.isnan(embeddings).any() or torch.isinf(embeddings).any()
                bad_flag = torch.tensor(float(has_bad), device=device)
                dist.all_reduce(bad_flag, op=dist.ReduceOp.MAX)
                if bad_flag.item() > 0:
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

                # --- gather across GPUs ---
                # GatherLayer preserves gradient flow to the local process's slice.
                # target_emb has no grad (frozen CLIP), so plain gather suffices.
                all_embeddings = gather_with_grad(embeddings)  # (B*N, K, P)
                all_target_emb = accelerator.gather(target_emb)  # (B*N, P)

                queue_emb = neg_queue.get() if len(neg_queue) > 0 else None

                # Regularise cls_probes directly — they live in a fixed hidden_dim space
                # calculates sum of square of cosine similarity between probes
                probe_gram = torch.mm(
                    F.normalize(raw_model.cls_probes, dim=-1),
                    F.normalize(raw_model.cls_probes, dim=-1).T,
                )  # (K, K)
                off_diag = probe_gram.masked_fill(
                    torch.eye(NUM_EMBS, device=device, dtype=torch.bool), 0.0
                )
                diversity_loss = (off_diag**2).sum()

                # diversity in output embedding space across the full global batch
                mean_probes = F.normalize(all_embeddings.mean(dim=0), dim=-1)  # (K, P)
                output_gram = torch.mm(mean_probes, mean_probes.T)  # (K, K)
                output_off_diag = output_gram.masked_fill(
                    torch.eye(NUM_EMBS, device=device, dtype=torch.bool), 0.0
                )
                output_diversity_loss = (output_off_diag**2).sum()

                infonce = multiprobe_infonce_loss(
                    all_embeddings, all_target_emb, INFONCE_TEMP, K_HARD, queue_emb
                )
                loss = infonce + DIVERSITY_WEIGHT * (
                    diversity_loss + output_diversity_loss
                )

            # backward pass — del after backward so the autograd graph is released
            accelerator.backward(loss)
            loss_val = loss.item()
            neg_queue.enqueue(all_target_emb)  # grow negative pool for future batches
            del (
                target_emb,
                all_target_emb,
                all_embeddings,
                embeddings,
                loss,
            )
            accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            pbar.update(1)

            if accelerator.is_main_process:
                run.log(
                    {
                        "batch_idx": batch_idx + 1,
                        "loss": loss_val,
                        "lr": scheduler.get_last_lr()[0],
                        "queue_size": len(neg_queue),
                    }
                )

            # ----------------------------------------------------------
            # Periodic CIRCO validation (every VAL_INTERVAL batches)
            # ----------------------------------------------------------
            if (
                (batch_idx + 1) % VAL_INTERVAL == 0
                and accelerator.is_main_process
                and val_queries is not None
            ):
                LOGGER.info("Running CIRCO validation at batch %d...", batch_idx + 1)
                val_metrics = run_circo_val(
                    raw_model,
                    processor,
                    gallery_embs_val,
                    coco_ids_val,
                    val_queries,
                    device,
                )
                run.log({"batch_idx": batch_idx + 1, **val_metrics})
                LOGGER.info(
                    "Val @batch=%d | mAP@5=%.4f mAP@10=%.4f mAP@25=%.4f mAP@50=%.4f"
                    " | R@1=%.4f R@5=%.4f R@10=%.4f R@25=%.4f R@50=%.4f",
                    batch_idx + 1,
                    val_metrics["val/mAP@5"],
                    val_metrics["val/mAP@10"],
                    val_metrics["val/mAP@25"],
                    val_metrics["val/mAP@50"],
                    val_metrics["val/R@1"],
                    val_metrics["val/R@5"],
                    val_metrics["val/R@10"],
                    val_metrics["val/R@25"],
                    val_metrics["val/R@50"],
                )

            # ----------------------------------------------------------
            # Periodic checkpoint (every CHECKPOINT_INTERVAL batches)
            # batch_idx is 0-based, so we trigger at 999, 1999, …
            # ----------------------------------------------------------
            if (batch_idx + 1) % CHECKPOINT_INTERVAL == 0:
                if accelerator.is_main_process:
                    save_checkpoint(
                        experiment_id,
                        epoch,
                        batch_idx,
                        raw_model,
                        optimizer,
                        scheduler,
                        skipped,
                        CHECKPOINT_DIR,
                        LOGGER,
                    )

            if accelerator.is_main_process and batch_idx % 10 == 0:
                LOGGER.info(
                    "Epoch=%d batch=%d | loss=%.4f | lr=%.2e\n",
                    epoch + 1,
                    batch_idx + 1,
                    loss_val,
                    scheduler.get_last_lr()[0],
                )
                log_vram(f"epoch={epoch + 1} batch={batch_idx + 1}", LOGGER, "cuda")
        pbar.close()
        if accelerator.is_main_process:
            log_vram(f"epoch={epoch + 1} end", LOGGER, "cuda")

        # Also checkpoint at the end of each epoch so epoch boundaries are
        # always recoverable even if CHECKPOINT_INTERVAL doesn't land there.
        if accelerator.is_main_process:
            save_checkpoint(
                experiment_id,
                epoch,
                NUM_BATCHES - 1,
                raw_model,
                optimizer,
                scheduler,
                skipped,
                CHECKPOINT_DIR,
                LOGGER,
            )

    if accelerator.is_main_process:
        LOGGER.info("saving model to CoLLM.pt")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"collm_4probes_{experiment_id}_{timestamp}.pt"
        torch.save(raw_model.state_dict(), model_filename)
        LOGGER.info(f"Model saved as: {model_filename}")
        run.finish()


if __name__ == "__main__":
    main()

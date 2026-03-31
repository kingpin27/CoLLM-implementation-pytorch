import glob
import os
import subprocess

import torch
import torch.distributed as dist
from torch.nn import functional as F


class GatherLayer(torch.autograd.Function):
    """All-gather with gradient flow back to the local process.

    forward : concatenates tensors from all ranks → (B*N, ...)
    backward: all-reduces incoming grads, then returns only the local slice
              so gradients flow correctly to each process's own embeddings.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x.contiguous())
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_grads = torch.stack(grads)
        dist.all_reduce(all_grads)
        return all_grads[dist.get_rank()]


def gather_with_grad(x):
    """Gather x from all processes, keeping gradients for the local slice.

    Falls back to a no-op on single-GPU runs where dist is not initialised.
    """
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
        return x
    return torch.cat(GatherLayer.apply(x), dim=0)


class NegativeQueue:
    """FIFO queue of frozen target CLIP embeddings for expanding the InfoNCE negative set.

    Since target embeddings are pre-computed frozen CLIP vectors (no grad), the queue
    requires no momentum encoder — just store and replay them as extra negatives.
    All ranks enqueue the same gathered embeddings, so queues stay in sync automatically.
    """

    def __init__(self, queue_size: int, embed_dim: int, device):
        self.queue_size = queue_size
        self.buffer = torch.zeros(queue_size, embed_dim, device=device)
        self.ptr = 0
        self.filled = 0

    @torch.no_grad()
    def enqueue(self, embeddings: torch.Tensor):
        """Enqueue embeddings (N, D), overwriting oldest entries when full."""
        N = embeddings.shape[0]
        emb = embeddings.detach()
        if N >= self.queue_size:
            self.buffer.copy_(emb[-self.queue_size:])
            self.ptr = 0
            self.filled = self.queue_size
            return
        end = self.ptr + N
        if end <= self.queue_size:
            self.buffer[self.ptr:end].copy_(emb)
        else:
            split = self.queue_size - self.ptr
            self.buffer[self.ptr:].copy_(emb[:split])
            self.buffer[:end - self.queue_size].copy_(emb[split:])
        self.ptr = end % self.queue_size
        self.filled = min(self.filled + N, self.queue_size)

    def get(self) -> torch.Tensor:
        """Return all valid embeddings currently in the queue."""
        if self.filled == self.queue_size:
            return self.buffer.clone()
        return self.buffer[:self.filled].clone()

    def __len__(self) -> int:
        return self.filled


def hard_infonce_loss(
    query_emb: torch.Tensor,
    target_emb: torch.Tensor,
    temp: float,
    k_hard: int,
    queue_emb: torch.Tensor = None,
) -> torch.Tensor:
    """Symmetric InfoNCE with hard negative mining.

    query-to-target direction: mine top-k_hard negatives from batch + queue per query.
    target-to-query direction: standard InfoNCE within current batch only
                               (queue entries have no paired queries).

    Args:
        query_emb:  (B, P) gathered query embeddings — grads flow through these.
        target_emb: (B, P) gathered true-positive target embeddings — no grad.
        temp:       InfoNCE temperature.
        k_hard:     Hard negatives per query. 0 = standard InfoNCE over full pool.
        queue_emb:  (Q, P) queued negatives, or None if queue is empty.
    """
    B = query_emb.shape[0]

    # Build negative pool: [current-batch targets | queue]
    if queue_emb is not None and queue_emb.shape[0] > 0:
        neg_pool = torch.cat([target_emb, queue_emb], dim=0)  # (B+Q, P)
    else:
        neg_pool = target_emb  # (B, P)

    M = neg_pool.shape[0]

    # --- query → target (with hard mining if k_hard > 0 and pool is large enough) ---
    sims = torch.matmul(query_emb, neg_pool.T)  # (B, M)

    if k_hard > 0 and k_hard < M - 1:
        # Mask true positives (diagonal of first B columns) before mining
        tp_mask = torch.zeros(B, M, dtype=torch.bool, device=query_emb.device)
        tp_mask[torch.arange(B), torch.arange(B)] = True
        mining_sims = sims.masked_fill(tp_mask, float("-inf"))

        _, hard_idx = mining_sims.topk(k_hard, dim=1)              # (B, k_hard)
        hard_negs = neg_pool[hard_idx]                               # (B, k_hard, P)

        pos_sim = (query_emb * target_emb).sum(dim=-1, keepdim=True) / temp   # (B, 1)
        hard_sim = torch.bmm(hard_negs, query_emb.unsqueeze(-1)).squeeze(-1) / temp  # (B, k_hard)

        logits_q2t = torch.cat([pos_sim, hard_sim], dim=1)          # (B, 1+k_hard)
        labels_q2t = torch.zeros(B, dtype=torch.long, device=query_emb.device)
    else:
        # Standard InfoNCE over full pool (k_hard=0 or pool too small)
        logits_q2t = sims / temp                                     # (B, M)
        labels_q2t = torch.arange(B, device=query_emb.device)

    loss_q2t = F.cross_entropy(logits_q2t, labels_q2t)

    # --- target → query (current batch only) ---
    logits_t2q = torch.matmul(target_emb, query_emb.T) / temp       # (B, B)
    labels_t2q = torch.arange(B, device=query_emb.device)
    loss_t2q = F.cross_entropy(logits_t2q, labels_t2q)

    return (loss_q2t + loss_t2q) / 2


# Before your training loop
def get_probe_temp(batch_idx, total_batches, temp_start=1.0, temp_end=0.1):
    progress = min(batch_idx / total_batches, 1.0)
    return temp_start * (temp_end / temp_start) ** progress  # exponential decay


def save_checkpoint(
    experiment_id,
    epoch,
    batch_idx,
    model,
    optimizer,
    scheduler,
    skipped,
    CHECKPOINT_DIR,
    LOGGER,
):
    """Save full training state so a run can be resumed exactly."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(
        CHECKPOINT_DIR,
        f"collm_{experiment_id}_epoch{epoch}_batch{batch_idx}.ckpt",
    )
    torch.save(
        {
            "experiment_id": experiment_id,
            "epoch": epoch,
            "batch_idx": batch_idx,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "skipped": skipped,
        },
        path,
    )
    LOGGER.info("Checkpoint saved: %s", path)
    return path


def find_latest_checkpoint(experiment_id, CHECKPOINT_DIR):
    """
    Scan CHECKPOINT_DIR for all checkpoints belonging to experiment_id and
    return the path of the one with the highest (epoch, batch_idx), or None.
    """
    pattern = os.path.join(CHECKPOINT_DIR, f"collm_{experiment_id}_epoch*_batch*.ckpt")
    candidates = glob.glob(pattern)
    if not candidates:
        return None

    def _rank(p):
        # Extract epoch and batch numbers from filename for sorting.
        base = os.path.basename(p)  # collm_<id>_epoch<E>_batch<B>.ckpt
        try:
            parts = base.replace(".ckpt", "").split("_")
            epoch = int(next(p for p in parts if p.startswith("epoch"))[5:])
            batch = int(next(p for p in parts if p.startswith("batch"))[5:])
            return (epoch, batch)
        except Exception:
            return (-1, -1)

    return max(candidates, key=_rank)


def get_git_info() -> dict:
    try:
        commit_hash = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        short_hash = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        branch = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except subprocess.CalledProcessError:
        commit_hash = short_hash = branch = "unknown"
    return {"git_commit": commit_hash, "git_short": short_hash, "git_branch": branch}


def collm_contrastive_collate_fn(batch):
    batch = [item for item in batch if item is not None]  # drop failed samples
    if not batch:
        return None

    return {
        "id": [item["id"] for item in batch],
        "image": [item["image"] for item in batch],
        "target_image_emb": [item["target_image_emb"] for item in batch],
        "modification_text": [item["modification_text"] for item in batch],
    }


def log_vram(label, LOGGER, device):
    if device != "cuda":
        return
    allocated = torch.cuda.max_memory_allocated() / 1e9
    reserved = torch.cuda.max_memory_reserved() / 1e9
    LOGGER.info(
        f"VRAM [{label}] peak allocated={allocated:.2f}GB  reserved={reserved:.2f}GB"
    )
    torch.cuda.reset_peak_memory_stats()  # reset so next window is fresh


def param_summary(model, LOGGER):
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


def tensor_shape(tensor):
    return tuple(tensor.shape)

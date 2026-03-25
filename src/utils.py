import glob
import os
import subprocess

import torch


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

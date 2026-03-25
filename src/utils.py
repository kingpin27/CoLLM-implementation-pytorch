import torch


def collm_contrastive_collate_fn(batch):
    return {
        "id": [item["id"] for item in batch],
        "image": [item["image"] for item in batch],
        "target_image_emb": [item["target_image_emb"] for item in batch],  # fixed key
        "modification_text": [item["modification_text"] for item in batch],
    }


def log_vram(LOGGER, device, label=""):
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

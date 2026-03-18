import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import os
from PIL import Image
from torch.nn import functional as F
from transformers import AutoModelForMultimodalLM , AutoProcessor
from tqdm.auto import tqdm


class MTCIRDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.examples = pd.read_json(annotations_file, lines=True)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        sample = self.examples.iloc[idx]

        source_path = os.path.join(self.img_dir, sample["image"])
        target_path = os.path.join(self.img_dir, sample["target_image"])

        source_image = Image.open(source_path).convert("RGB")
        target_image = Image.open(target_path).convert("RGB")

        if self.transform is not None:
            source_image = self.transform(source_image)
        if self.target_transform is not None:
            target_image = self.target_transform(target_image)

        modifications = sample["modifications"]

        return {
            "id": sample["id"],
            "image": source_image,
            "target_image": target_image,
            "modification_text": modifications[0]
        }

class CoLLM(torch.nn.Module):
    def __init__(
        self,
        model_name="Qwen/Qwen3.5-2B",
        freeze_patterns=None,
        trainable_patterns=None,
        remove_output_projection_layer=True,
        output_projection_name=None,
        disable_last_layers=None,
        llm_block_path=None,
        llm_trainable_last_layers=None,
        projection_dim=None,
        projection_in_features=None,
        projection_dropout=0.0,
        projection_dtype=None,
        projection_device=None,
    ):
        super().__init__()
        self.model_name = model_name  # or "Qwen/Qwen3.5-9B-Instruct"

        # Load tokenizer and model.
        # self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True
        )
        self.projection = None
        self.output_projection_path = None
        self.output_projection_backup = None

        if remove_output_projection_layer:
            self.remove_last_projection_layer(output_projection_name=output_projection_name)

        if projection_dim is not None:
            self.add_projection_layer(
                projection_in_features=projection_in_features,
                projection_out_features=projection_dim,
                dropout=projection_dropout,
                projection_dtype=projection_dtype,
                projection_device=projection_device,
            )

        if disable_last_layers is not None and disable_last_layers > 0:
            self.disable_end_layers(num_layers=disable_last_layers, llm_block_path=llm_block_path)

        if llm_trainable_last_layers is not None and llm_trainable_last_layers > 0:
            self.make_last_llm_layers_trainable(
                num_layers=llm_trainable_last_layers,
                llm_block_path=llm_block_path,
            )

        if freeze_patterns is not None:
            self.freeze_modules(freeze_patterns)

        if trainable_patterns is not None:
            self.set_trainable_modules(trainable_patterns)

    def _is_iterable(self, value):
        return isinstance(value, (list, tuple, set))

    def _normalize_patterns(self, patterns):
        if patterns is None:
            return []
        if isinstance(patterns, str):
            return [patterns]
        return list(patterns)

    def freeze_all(self):
        """Freeze all model parameters."""
        for _, param in self.named_parameters():
            param.requires_grad = False

    def unfreeze_all(self):
        """Unfreeze all model parameters."""
        for _, param in self.named_parameters():
            param.requires_grad = True

    def freeze_modules(self, module_patterns):
        """
        Freeze all parameters whose names contain any pattern.
        Example: ["vision_tower", "vision_model"].
        """
        patterns = self._normalize_patterns(module_patterns)
        for name, param in self.named_parameters():
            if any(pattern in name for pattern in patterns):
                param.requires_grad = False

    def set_trainable_modules(self, module_patterns, strict=True):
        """
        Keep only matching modules trainable. If strict=True, everything else is frozen.
        If strict=False, matching modules are set trainable and others remain unchanged.
        """
        patterns = self._normalize_patterns(module_patterns)
        if strict:
            self.freeze_all()
        for name, param in self.named_parameters():
            if any(pattern in name for pattern in patterns):
                param.requires_grad = True
        return self.get_trainable_param_names()

    def get_trainable_param_names(self):
        return [name for name, param in self.named_parameters() if param.requires_grad]

    def get_trainable_parameters(self):
        return [param for _, param in self.named_parameters() if param.requires_grad]

    def disable_end_layers(self, num_layers, llm_block_path=None):
        """
        Backward-compatible alias for remove_end_layers().
        """
        return self.remove_end_layers(num_layers=num_layers, llm_block_path=llm_block_path)

    def remove_end_layers(self, num_layers, llm_block_path=None):
        """
        Remove the last `num_layers` transformer layers from the language model.
        """
        layer_path, layer_block = self._find_llm_layer_block(llm_block_path)
        if layer_block is None:
            raise ValueError(
                "Could not find transformer layers block. "
                f"Pass a valid llm_block_path (for example: '{llm_block_path}')."
            )

        num_layers = min(num_layers, len(layer_block))
        if num_layers <= 0:
            return []

        keep_count = len(layer_block) - num_layers
        removed_indices = list(range(keep_count, len(layer_block)))
        self._set_submodule(
            layer_path,
            torch.nn.ModuleList(list(layer_block.children())[:keep_count]),
        )
        return removed_indices

    def make_last_llm_layers_trainable(self, num_layers, llm_block_path=None):
        """
        Freeze all parameters, then unfreeze only the last `num_layers` LLM layers.
        """
        self.freeze_all()
        layer_path, layer_block = self._find_llm_layer_block(llm_block_path)
        if layer_block is None:
            raise ValueError(
                "Could not find transformer layers block. "
                f"Pass a valid llm_block_path (for example: '{llm_block_path}')."
            )

        num_layers = max(0, num_layers)
        if num_layers == 0:
            return []

        if num_layers >= len(layer_block):
            for _, param in layer_block.named_parameters():
                param.requires_grad = True
            return [f"{layer_path}.{idx}" for idx in range(len(layer_block))]

        start = len(layer_block) - num_layers
        trainable_layer_indices = []
        for layer_idx in range(start, len(layer_block)):
            for name, param in layer_block[layer_idx].named_parameters():
                param.requires_grad = True
            trainable_layer_indices.append(f"{layer_path}.{layer_idx}")
        return trainable_layer_indices

    def _find_llm_layer_block(self, llm_block_path=None):
        """
        Find a plausible ModuleList/Sequential containing LLM transformer layers.
        """
        candidates = [
            llm_block_path,
            "language_model.model.layers",
            "language_model.model.blocks",
            "language_model.transformer.h",
            "language_model.transformer.layers",
            "language_model.blocks",
            "transformer.h",
            "transformer.layers",
            "transformer.blocks",
            "model.language_model.layers",
            "model.language_model.blocks",
            "model.transformer.h",
            "model.layers",
            "model.blocks",
        ]
        for path in candidates:
            if path is None:
                continue
            try:
                module = self._get_submodule(path)
            except AttributeError:
                continue
            if isinstance(module, (torch.nn.ModuleList, torch.nn.Sequential)) and len(module) > 0:
                return path, module

        # Fallback: pick the largest Layer/Block-like sequential module under model.
        fallback_path = None
        fallback_len = 0
        fallback_module = None
        for name, module in self.model.named_modules():
            if name == "":
                continue
            if not isinstance(module, (torch.nn.ModuleList, torch.nn.Sequential)):
                continue
            if len(module) < 2:
                continue
            if ".layer" not in name and ".layers" not in name and ".h" not in name and ".block" not in name:
                continue
            if len(module) > fallback_len:
                fallback_path = name
                fallback_len = len(module)
                fallback_module = module
        return fallback_path, fallback_module

    def _get_submodule(self, module_path):
        module = self.model
        for key in module_path.split("."):
            module = getattr(module, key)
        return module

    def _set_submodule(self, module_path, module_obj):
        parent = self.model
        keys = module_path.split(".")
        for key in keys[:-1]:
            parent = getattr(parent, key)
        setattr(parent, keys[-1], module_obj)

    def _collect_projection_candidates(self):
        candidates = []
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                candidates.append((name, module))
        return candidates

    def remove_last_projection_layer(self, output_projection_name=None):
        if output_projection_name is not None:
            try:
                module = self._get_submodule(output_projection_name)
            except AttributeError as exc:
                raise ValueError(f"Projection module '{output_projection_name}' not found.") from exc
            if not isinstance(module, torch.nn.Linear):
                raise ValueError(f"'{output_projection_name}' is not a Linear layer.")
            self.output_projection_backup = module
            self.output_projection_path = output_projection_name
            self._set_submodule(output_projection_name, torch.nn.Identity())
            return output_projection_name

        # Qwen variants commonly keep the final language-head at `lm_head`.
        direct_candidates = [
            "lm_head",
            "language_model.lm_head",
            "model.lm_head",
        ]
        for name in direct_candidates:
            try:
                module = self._get_submodule(name)
                if isinstance(module, torch.nn.Linear):
                    self.output_projection_backup = module
                    self.output_projection_path = name
                    self._set_submodule(name, torch.nn.Identity())
                    return name
            except AttributeError:
                continue

        # Fallback: replace the last Linear layer that maps to vocab size, if present.
        vocab_size = getattr(self.model.config, "vocab_size", None)
        projection_like = [
            (name, module)
            for name, module in self._collect_projection_candidates()
            if vocab_size is not None and module.weight.shape[0] == vocab_size
        ]
        if projection_like:
            name, module = projection_like[-1]
            self.output_projection_backup = module
            self.output_projection_path = name
            self._set_submodule(name, torch.nn.Identity())
            return name

        # Last resort: remove the last Linear layer in the model graph.
        linear_layers = self._collect_projection_candidates()
        if linear_layers:
            name, module = linear_layers[-1]
            self.output_projection_backup = module
            self.output_projection_path = name
            self._set_submodule(name, torch.nn.Identity())
            return name

        raise ValueError(
            "Could not identify an output projection layer. "
            "Pass output_projection_name explicitly."
        )

    def restore_last_projection_layer(self):
        if self.output_projection_path is None or self.output_projection_backup is None:
            raise ValueError("No removed projection layer to restore.")
        self._set_submodule(self.output_projection_path, self.output_projection_backup)
        path = self.output_projection_path
        self.output_projection_path = None
        self.output_projection_backup = None
        return path

    def _infer_projection_in_features(self):
        config = getattr(self.model, "config", None)
        if config is None:
            return None
        for key in ["hidden_size", "d_model", "embed_dim"]:
            if hasattr(config, key):
                return getattr(config, key)
        text_cfg = getattr(config, "text_config", None)
        if text_cfg is not None:
            for key in ["hidden_size", "d_model", "embed_dim"]:
                if hasattr(text_cfg, key):
                    return getattr(text_cfg, key)
        vision_cfg = getattr(config, "vision_config", None)
        if vision_cfg is not None:
            for key in ["hidden_size", "d_model", "embed_dim"]:
                if hasattr(vision_cfg, key):
                    return getattr(vision_cfg, key)
        return None

    def _infer_projection_dtype(self):
        ref_param = next(self.model.parameters(), None)
        if ref_param is None:
            return torch.float32
        return ref_param.dtype

    def _infer_projection_device(self):
        ref_param = next(self.model.parameters(), None)
        if ref_param is None:
            return torch.device("cpu")
        return ref_param.device

    def encode(
        self,
        processor_inputs,
        return_attention_mask=True,
        with_projection=True,
    ):
        outputs = self.model(
            **processor_inputs,
            output_hidden_states=True,
            return_dict=True,
        )
        token_embeddings = outputs.hidden_states[-1]
        attention_mask = processor_inputs.get("attention_mask") if return_attention_mask else None

        if with_projection and self.projection is not None:
            token_embeddings = self.projection(token_embeddings)
        return token_embeddings, attention_mask

    @staticmethod
    def late_interaction_score(query_embeddings, query_mask, doc_embeddings, doc_mask):
        query_embeddings = F.normalize(query_embeddings, dim=-1)
        doc_embeddings = F.normalize(doc_embeddings, dim=-1)

        batch_size = query_embeddings.size(0)
        scores = torch.empty((batch_size, batch_size), device=query_embeddings.device, dtype=query_embeddings.dtype)

        for i in range(batch_size):
            q_len = query_embeddings.size(1) if query_mask is None else int(query_mask[i].sum().item())
            q = query_embeddings[i, :q_len]

            for j in range(batch_size):
                d_len = doc_embeddings.size(1) if doc_mask is None else int(doc_mask[j].sum().item())
                d = doc_embeddings[j, :d_len]

                if q_len == 0 or d_len == 0:
                    scores[i, j] = 0.0
                    continue

                sim = q @ d.t()
                scores[i, j] = sim.max(dim=1).values.sum()

        return scores

    def add_projection_layer(
        self,
        projection_in_features=None,
        projection_out_features=None,
        dropout=0.0,
        freeze_projection=True,
        replace_existing=False,
        projection_dtype=None,
        projection_device=None,
    ):
        """
        Attach a linear projection head for extending model outputs.
        projection_out_features is required.
        """
        if projection_out_features is None:
            raise ValueError("projection_out_features is required to add a projection layer.")

        if projection_in_features is None:
            projection_in_features = self._infer_projection_in_features()
            if projection_in_features is None:
                raise ValueError(
                    "Could not infer projection input features. "
                    "Pass projection_in_features explicitly."
                )

        if self.projection is not None and not replace_existing:
            raise ValueError("Projection layer already exists. Set replace_existing=True to recreate it.")

        if projection_dtype is None:
            projection_dtype = self._infer_projection_dtype()
        if projection_device is None:
            projection_device = self._infer_projection_device()

        self.projection = torch.nn.Sequential(
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(
                projection_in_features,
                projection_out_features,
                device=projection_device,
                dtype=projection_dtype,
            ),
        )
        self.projection_dropout = dropout
        self.projection_in_features = projection_in_features
        self.projection_out_features = projection_out_features

        if freeze_projection:
            self.freeze_modules(["projection"])

        return self.projection


def build_contrastive_inputs(processor, images, texts):
    messages = []
    for text in texts:
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": text},
                ],
            }
        )
    rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return processor(
        text=rendered,
        images=images,
        return_tensors="pt",
        padding=True,
    )


def collm_contrastive_collate_fn(batch):
    return {
        "id": [item["id"] for item in batch],
        "image": [item["image"] for item in batch],
        "target_image": [item["target_image"] for item in batch],
        "modification_text": [item["modification_text"] for item in batch],
    }


def save_collm_checkpoint(collm, processor, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    # Save base Qwen model safely (required for reloading by Auto classes).
    collm.model.save_pretrained(output_dir)
    # Save custom projection + training flags from the wrapper.
    torch.save(collm.state_dict(), os.path.join(output_dir, "collm_wrapper.pt"))
    # Save processor for full reproducibility.
    processor.save_pretrained(output_dir)


def train_contrastive(
    collm,
    processor,
    dataloader,
    epochs=1,
    temperature=0.07,
    lr=1e-4,
    device="cpu",
    save_output_dir=None,
    save_every_epoch=False,
    log_every_n_steps=None,
    log_every_n_examples=50000,
):
    model = collm.to(device)
    optimizer = torch.optim.AdamW(model.get_trainable_parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        epoch_steps = 0
        epoch_examples = 0
        interval_steps = log_every_n_steps
        if interval_steps is None:
            interval_steps = max(50, min(1000, len(dataloader) // 200 if len(dataloader) > 0 else 200))
        if interval_steps <= 0:
            interval_steps = 50

        print("loss printing interval:", interval_steps)

        next_example_log = log_every_n_examples
        if next_example_log is None:
            next_example_log = 0
        if next_example_log <= 0:
            next_example_log = int(1e18)

        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False):
            epoch_steps += 1
            batch_size = len(batch["modification_text"])
            epoch_examples += batch_size
            query_texts = [f"Describe the image with modifications- {t}" for t in batch["modification_text"]]
            doc_texts = [f"Describe the image" for _ in batch["modification_text"]]

            query_inputs = build_contrastive_inputs(processor, batch["image"], query_texts)
            doc_inputs = build_contrastive_inputs(processor, batch["target_image"], doc_texts)

            query_inputs = {key: value.to(device) for key, value in query_inputs.items()}
            doc_inputs = {key: value.to(device) for key, value in doc_inputs.items()}

            q_embeddings, q_mask = model.encode(query_inputs)
            d_embeddings, d_mask = model.encode(doc_inputs)

            logits = CoLLM.late_interaction_score(q_embeddings, q_mask, d_embeddings, d_mask) / temperature
            targets = torch.arange(logits.size(0), device=device)
            loss = F.cross_entropy(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if epoch_steps % interval_steps == 0:
                avg_loss = total_loss / epoch_steps
                print(
                    f"[Epoch {epoch + 1}] step {epoch_steps:,} / {len(dataloader) if hasattr(dataloader, '__len__') else '?'} | "
                    f"examples {epoch_examples:,} | avg loss: {avg_loss:.4f}"
                )
            if epoch_examples >= next_example_log:
                avg_loss = total_loss / max(1, epoch_steps)
                print(
                    f"[Epoch {epoch + 1}] processed {epoch_examples:,} examples | step {epoch_steps:,} | avg loss: {avg_loss:.4f}"
                )
                next_example_log += log_every_n_examples

        if len(dataloader) > 0:
            print(f"Epoch {epoch + 1}: contrastive loss = {total_loss / len(dataloader):.4f}")

        if save_output_dir is not None:
            if save_every_epoch:
                epoch_dir = os.path.join(save_output_dir, f"epoch_{epoch+1}")
                save_collm_checkpoint(model, processor, epoch_dir)
            else:
                # Save once after the final epoch in the default output dir.
                if epoch == epochs - 1:
                    save_collm_checkpoint(model, processor, save_output_dir)

    if save_output_dir is not None and not save_every_epoch:
        # Safety net if dataloader is empty.
        save_collm_checkpoint(model, processor, save_output_dir)


if __name__ == "__main__":
    annotations_file = "./MTCIR/mtcir.jsonl"
    image_dir = "./images"
    batch_size = 32
    num_epochs = 1

    device = "mps" if torch.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = MTCIRDataset(annotations_file=annotations_file, img_dir=image_dir)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collm_contrastive_collate_fn,
    )

    processor = AutoProcessor.from_pretrained("Qwen/Qwen3.5-2B", trust_remote_code=True)
    model = CoLLM(
        model_name="Qwen/Qwen3.5-2B",
        remove_output_projection_layer=True,
        projection_dim=768,
        projection_dropout=0.0,
        disable_last_layers=4,
        llm_block_path="model.language_model.layers",
        llm_trainable_last_layers=10,
    )

    train_contrastive(
        collm=model,
        processor=processor,
        dataloader=train_loader,
        epochs=num_epochs,
        temperature=0.07,
        lr=1e-4,
        device=device,
        save_output_dir="./checkpoints",
        save_every_epoch=True,
    )

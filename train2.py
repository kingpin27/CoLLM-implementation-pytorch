import os
import json
import torch
import logging
import sys
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
from torch.nn import functional as F
from transformers import AutoModelForMultimodalLM , AutoProcessor
from tqdm.auto import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    stream=sys.stdout,
    force=True,
)
LOGGER = logging.getLogger("train2")


def tensor_shape(tensor):
    return tuple(tensor.shape)


class MTCIRDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
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
        target_path = os.path.join(self.img_dir, record["target_image"])

        source_image = Image.open(source_path).convert("RGB")
        target_image = Image.open(target_path).convert("RGB")

        if self.transform is not None:
            source_image = self.transform(source_image)
        if self.target_transform is not None:
            target_image = self.target_transform(target_image)

        return {
            "id": record["id"],
            "image": source_image,
            "target_image": target_image,
            "modification_text": (
                record["modifications"][0]
                if isinstance(record.get("modifications"), list) and len(record["modifications"]) > 0
                else record.get("modifications", "")
            ),
        }


class CoLLM(torch.nn.Module):
    def __init__(
        self,
        model_name="Qwen/Qwen3.5-2B",
        projection_dim=256,
        freeze_vision_encoder=True,
    ):
        super().__init__()
        self.model_name = model_name
        self.projection_dim = projection_dim
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.mps.is_available() else "cpu")
        self.model_dtype = torch.float16 if self.device == "cuda" else torch.float32
        self._forward_calls = 0
        self._train_only_projection = True

        LOGGER.info("Initializing CoLLM with model=%s on device=%s", self.model_name, self.device)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            self.model_name,
            dtype=self.model_dtype,
            trust_remote_code=True,
        ).to(self.device)
        if freeze_vision_encoder:
            self.freeze_vision_encoder()
        self._train_only_projection = not any(p.requires_grad for p in self.model.parameters())
        model_dtype = self.model_dtype
        LOGGER.info("Loaded base model successfully. Model dtype=%s", model_dtype)

        hidden_dim = getattr(self.model.config, "hidden_size", 1024)
        self.output_linear_projection = torch.nn.Linear(hidden_dim, self.projection_dim).to(
            self.device,
            dtype=self.model_dtype,
        )
        LOGGER.info("Initialized projection head with in/out dims: %d -> %d", hidden_dim, self.projection_dim)

    def freeze_vision_encoder(self):
        """
        Freeze likely vision-related parameters by module-name heuristics.
        Keeps projection head and text-only language heads trainable.
        """
        vision_keywords = ("vision", "visual", "image", "pixel", "img")
        frozen = 0
        total = 0

        for name, param in self.model.named_parameters():
            total += 1
            if any(keyword in name.lower() for keyword in vision_keywords):
                param.requires_grad = False
                frozen += 1

        LOGGER.info(
            "freeze_vision_encoder enabled: frozen=%d parameter tensors out of %d matched model params",
            frozen,
            total,
        )
    
    @staticmethod
    def make_inputs(processor, image, text, device="cpu"):
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
                raise TypeError(f"Expected message dict for chat template, got {type(m)}")
            rendered.append(processor.apply_chat_template(
                [m],
                tokenize=False,
                add_generation_prompt=False,
            ))
        enc = processor(
            text=rendered,
            images=list(image),
            return_tensors="pt",
            padding=True,
        )
        return {k: v.to(device) for k, v in enc.items()}

    def forward(self, pil_image, text, processor, return_attention_mask=False):
        """
        pil_image/text can be single item or batch.
        Returns projected hidden states [B, T, projection_dim].
        """
        self._forward_calls += 1
        inputs = self.make_inputs(processor, pil_image, text, device=self.device)
        outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
        hidden = outputs.last_hidden_state
        if hidden is None and hasattr(outputs, "hidden_states"):
            hidden = outputs.hidden_states[-1]
        hidden = hidden.to(self.output_linear_projection.weight.dtype)
        projected = self.output_linear_projection(hidden)

        # if self._forward_calls <= 3:
        #     LOGGER.info(
        #         "Forward pass #%d: input_ids=%s, projected=%s",
        #         self._forward_calls,
        #         tensor_shape(inputs["input_ids"]),
        #         tensor_shape(projected),
        #     )

        if return_attention_mask:
            return projected, inputs.get("attention_mask").to(torch.bool) if inputs.get("attention_mask") is not None else None
        return projected

    def late_interaction_similarity(self, query_tokens, doc_tokens, query_mask=None, doc_mask=None, token_chunk_size=64):
        """
        Compute ColBERT-style late interaction scores for a full BxB batch.
        Returns scores shape [B, B] where score[i, j] is q_i against d_j.
        """
        q = F.normalize(query_tokens, dim=-1)
        d = F.normalize(doc_tokens, dim=-1)
        B, Tq, _ = q.shape
        scores = torch.zeros((B, B), device=q.device, dtype=q.dtype)

        for j in range(B):
            doc_j = d[j : j + 1].expand(B, -1, -1)
            qmask = ~query_mask if query_mask is not None else None
            mask_j = ~doc_mask[j : j + 1] if doc_mask is not None else None

            # Compute in chunks to avoid materializing [B, B, Tq, Td]-scale tensors.
            for start in range(0, Tq, token_chunk_size):
                end = min(Tq, start + token_chunk_size)
                q_chunk = q[:, start:end, :]
                token_sim = torch.bmm(q_chunk, doc_j.transpose(1, 2))

                if mask_j is not None:
                    token_sim = token_sim.masked_fill(mask_j[:, None, :], float("-inf"))

                token_max = token_sim.max(dim=2).values

                if qmask is not None:
                    token_max = token_max.masked_fill(qmask[:, start:end], 0.0)

                scores[:, j] += token_max.sum(dim=1)

        return scores

    def infonce_loss(self, query_tokens, doc_tokens, temperature=0.05, query_mask=None, doc_mask=None):
        """
        InfoNCE over in-batch positives:
        each query i is positive with doc i, negatives are all other docs.
        """
        scores = self.late_interaction_similarity(
            query_tokens, doc_tokens, query_mask=query_mask, doc_mask=doc_mask
        )
        labels = torch.arange(query_tokens.size(0), device=query_tokens.device)
        return torch.nn.functional.cross_entropy(scores / temperature, labels), scores

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
    batch_size = 8
    num_workers = 4

    LOGGER.info("Loading processor: %s", processor_name)
    processor = AutoProcessor.from_pretrained(processor_name, trust_remote_code=True)
    LOGGER.info("Processor loaded")
    
    LOGGER.info("Loading model: %s", model_name)
    model = CoLLM(model_name=model_name, freeze_vision_encoder=True)
    LOGGER.info("Model loaded and ready")

    LOGGER.info("Creating dataset from %s", './MTCIR/mtcir_expanded_shuffled.jsonl')
    train_dataset = MTCIRDataset('./MTCIR/mtcir_expanded_shuffled.jsonl', './images')
    LOGGER.info("Dataset ready with %d samples", len(train_dataset))
    LOGGER.info("Creating DataLoader batch_size=%d, num_workers=%d", batch_size, num_workers)
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

    for epoch in range(EPOCHS):
        LOGGER.info("Running epoch %d/%d", epoch + 1, EPOCHS)
        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{EPOCHS}",
            leave=False,
            file=sys.stdout,
        )
        for batch_idx, batch in enumerate(progress):
            model.train()
            optimizer.zero_grad()

            query_tokens, query_mask = model.forward(
                batch["image"],
                batch["modification_text"],
                processor=processor,
                return_attention_mask=True,
            )
            doc_prompts = ["Describe this image" for _ in range(len(batch["modification_text"]))]
            doc_tokens, doc_mask = model.forward(
                batch["target_image"],
                doc_prompts,
                processor=processor,
                return_attention_mask=True,
            )

            loss, scores = model.infonce_loss(
                query_tokens,
                doc_tokens,
                query_mask=query_mask,
                doc_mask=doc_mask,
            )
            loss.backward()
            optimizer.step()
            progress.set_postfix(loss=float(loss.item()), step=batch_idx + 1)
            if batch_idx % 10 == 0:
                LOGGER.info(
                    "Epoch=%d batch=%d | loss=%.4f | scores=%s",
                    epoch + 1,
                    batch_idx + 1,
                    loss.item(),
                    tensor_shape(scores),
                )

if __name__ == '__main__':
    main()

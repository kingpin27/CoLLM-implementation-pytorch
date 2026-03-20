import os
import torch
import logging
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
from PIL import Image
from torch.nn import functional as F
from transformers import AutoModelForMultimodalLM , AutoProcessor
from tqdm.auto import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
LOGGER = logging.getLogger("train2")


def tensor_shape(tensor):
    return tuple(tensor.shape)


class MTCIRDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        examples = pd.read_json(annotations_file, lines=True)
        self.img_dir = img_dir
        self.ids = examples["id"].tolist()
        self.source_images = examples["image"].tolist()
        self.target_images = examples["target_image"].tolist()
        self.modification_texts = [
            sample[0] if isinstance(sample, list) and len(sample) > 0 else sample
            for sample in examples["modifications"].tolist()
        ]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        source_path = os.path.join(self.img_dir, self.source_images[idx])
        target_path = os.path.join(self.img_dir, self.target_images[idx])

        source_image = Image.open(source_path).convert("RGB")
        target_image = Image.open(target_path).convert("RGB")

        if self.transform is not None:
            source_image = self.transform(source_image)
        if self.target_transform is not None:
            target_image = self.target_transform(target_image)

        return {
            "id": self.ids[idx],
            "image": source_image,
            "target_image": target_image,
            "modification_text": self.modification_texts[idx],
        }


class CoLLM(torch.nn.Module):
    def __init__(
        self,
        model_name="Qwen/Qwen3.5-2B",
        projection_dim=256,
    ):
        super().__init__()
        self.model_name = model_name
        self.projection_dim = projection_dim
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.mps.is_available() else "cpu")
        self._forward_calls = 0

        LOGGER.info("Initializing CoLLM with model=%s on device=%s", self.model_name, self.device)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True
        ).to(self.device)
        model_dtype = next(self.model.parameters()).dtype
        LOGGER.info("Loaded base model successfully. Model dtype=%s", model_dtype)

        hidden_dim = getattr(self.model.config, "hidden_size", 1024)
        self.output_linear_projection = torch.nn.Linear(hidden_dim, self.projection_dim).to(
            self.device,
            dtype=model_dtype,
        )
        LOGGER.info("Initialized projection head with in/out dims: %d -> %d", hidden_dim, self.projection_dim)
    
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
        hidden = outputs.hidden_states[-1]
        hidden = hidden.to(self.output_linear_projection.weight.dtype)
        projected = self.output_linear_projection(hidden)

        if self._forward_calls <= 3:
            LOGGER.info(
                "Forward pass #%d: input_ids=%s, projected=%s",
                self._forward_calls,
                tensor_shape(inputs["input_ids"]),
                tensor_shape(projected),
            )

        if return_attention_mask:
            return projected, inputs.get("attention_mask")
        return projected

    def late_interaction_similarity(self, query_tokens, doc_tokens, query_mask=None, doc_mask=None):
        """
        Compute ColBERT-style late interaction scores for a full BxB batch.
        Returns scores shape [B, B] where score[i, j] is q_i against d_j.
        """
        q = F.normalize(query_tokens, dim=-1)
        d = F.normalize(doc_tokens, dim=-1)

        q_exp = q[:, None, :, :]
        d_exp = d[None, :, :, :]

        # [B, B, Tq, Td]
        token_sim = torch.matmul(q_exp, d_exp.transpose(-1, -2))

        if doc_mask is not None:
            dmask = (~doc_mask[None, :, None, :]).to(torch.bool)
            token_sim = token_sim.masked_fill(dmask, float("-inf"))

        token_max = token_sim.max(dim=3).values  # [B, B, Tq]

        if query_mask is not None:
            qmask = (~query_mask[:, None, :]).to(torch.bool)
            token_max = token_max.masked_fill(qmask, 0.0)

        return token_max.sum(dim=2)

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
    batch_size = 32
    num_workers = 4

    LOGGER.info("Loading processor: %s", processor_name)
    processor = AutoProcessor.from_pretrained(processor_name, trust_remote_code=True)
    LOGGER.info("Processor loaded")
    
    LOGGER.info("Loading model: %s", model_name)
    model = CoLLM(model_name=model_name)
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    EPOCHS = 1
    LOGGER.info("Starting training for %d epoch(s)", EPOCHS)

    for epoch in range(EPOCHS):
        LOGGER.info("Running epoch %d/%d", epoch + 1, EPOCHS)
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}", leave=False)
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

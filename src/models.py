import torch
from torch.nn import functional as F
from transformers import AutoModelForMultimodalLM


class CoLLM(torch.nn.Module):
    def __init__(
        self,
        model_name="Qwen/Qwen3.5-0.8B",
        projection_dim=512,
        num_embeddings=4,
        hidden_dim=512,
        keep_layers=16,
        device="cuda",
    ):
        super().__init__()
        self.device = device
        self.projection_dim = projection_dim
        self.model_dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self.num_embeddings = num_embeddings
        self.keep_layers = keep_layers

        self.model = AutoModelForMultimodalLM.from_pretrained(
            model_name,
            dtype=self.model_dtype,
            trust_remote_code=True,
            # attn_implementation="flash_attention_2",
        ).to(device)

        for p in self.model.model.visual.parameters():
            p.requires_grad = False
        self.model.model.language_model.layers = self.model.model.language_model.layers[
            :keep_layers
        ]
        self.model.lm_head = None

        self.cls_probes = torch.nn.Parameter(
            torch.randn(num_embeddings, hidden_dim, dtype=self.model_dtype)
        )
        self.probe_proj = torch.nn.Linear(
            hidden_dim,
            projection_dim,
            dtype=self.model_dtype,
        )

        self.probe_router = torch.nn.Linear(projection_dim, num_embeddings)

    def make_inputs(self, processor, image, text):
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
                raise TypeError(
                    f"Expected message dict for chat template, got {type(m)}"
                )
            rendered.append(
                processor.apply_chat_template(
                    [m],
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )
        enc = processor(
            text=rendered,
            images=list(image),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        return {k: v.to(self.device) for k, v in enc.items()}

    def forward(self, images, text, processor):
        inputs = self.make_inputs(processor, images, text)
        with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
            outputs = self.model.model(
                **inputs, output_hidden_states=False, return_dict=True
            )
        hidden = outputs.last_hidden_state
        mask = inputs["attention_mask"].to(torch.bfloat16)  # (B, S)

        scores = torch.matmul(
            self.cls_probes.unsqueeze(0),  # (1, K, H)
            hidden.transpose(1, 2),  # (B, H, S)
        )  # (B, K, S)

        scores = scores.masked_fill(mask.unsqueeze(1) == 0, float("-inf"))
        attn = torch.softmax(scores, dim=-1)  # (B, K, S)
        pooled = torch.matmul(attn, hidden)  # (B, K, H)

        embeddings = self.probe_proj(pooled)  # (B, K, projection_dim)
        embeddings = F.normalize(embeddings, dim=-1)
        return embeddings

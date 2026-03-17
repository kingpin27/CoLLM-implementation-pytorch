import torch
from transformers import Qwen3VLForConditionalGeneration


class CoLLM(torch.nn.Module):
    def __init__(self):
        super(CoLLM, self).__init()
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-4B-Instruct",
            torch_dtype="auto",
            device_map="mps"
        )
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import os
from PIL import Image
from torch.nn import functional as F
from transformers import AutoModelForMultimodalLM , AutoProcessor
from tqdm.auto import tqdm


processor = AutoProcessor.from_pretrained("Qwen/Qwen3.5-2B", trust_remote_code=True)
model = AutoModelForMultimodalLM.from_pretrained(
        "Qwen/Qwen3.5-2B",
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    )

print(model.dtype)
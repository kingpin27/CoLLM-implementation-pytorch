import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor

from train3 import CoLLM  # import your model class

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load ---
model = CoLLM(
    model_name="Qwen/Qwen3.5-0.8B",
    projection_dim=768,
    num_embeddings=4,
    hidden_dim=1024,
)
model.load_state_dict(torch.load("collm_20250323_120000.pt", map_location=device))
model.to(device)
model.eval()

processor = AutoProcessor.from_pretrained("Qwen/Qwen3.5-0.8B", trust_remote_code=True)

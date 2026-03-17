from transformers import Qwen3VLForConditionalGeneration

model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen3-VL-Embedding-2B",
    torch_dtype="auto",
    device_map="mps"
)

print(model)

# layers = model.model.language_model.model.layers
# print(len(layers))
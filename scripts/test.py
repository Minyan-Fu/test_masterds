import torch
from transformers import AutoTokenizer, AutoModel

model_name = "OpenGVLab/InternVL3_5-8B"

print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())

# 1. 选择设备：有 GPU 就用 GPU，没有就用 CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print("using device:", device)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

print("Loading model...")
model = AutoModel.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype="auto",     # 让它自己选精度
    device_map=None         # 我们手动 .to(device)
)

model = model.to(device).eval()
print("Model loaded!")

# 2. 来一条最简单的 prompt，先看能不能生成
prompt = "You are a helpful assistant. Say hello in one short sentence."

inputs = tokenizer(prompt, return_tensors="pt").to(device)

print("Running generation...")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=64
    )

text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n===== MODEL OUTPUT =====")
print(text)
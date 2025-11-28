import os
import torch
from transformers import AutoTokenizer, AutoModel

# 1. 模型名字（HuggingFace 上的）
model_name = "OpenGVLab/InternVL3_5-8B"

# 2. 本地缓存目录（改成你自己的）
cache_dir = r"/user/minyan.fu/u23252/.project/dir.project/minyan/test/test_masterds/models"  # Windows 路径前面加 r

os.makedirs(cache_dir, exist_ok=True)

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# 3. 第一次会从网上下载到 cache_dir，之后就只用本地文件
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    cache_dir=cache_dir
)

print("Loading model (this may take a while the first time)...")
model = AutoModel.from_pretrained(
    model_name,
    trust_remote_code=True,
    cache_dir=cache_dir,
    torch_dtype="auto",
    device_map=None,          # 先手动 .to(device)
)

model = model.to(device).eval()
print("Model loaded!")

# 4. 简单测一条文本 prompt
prompt = "You are a helpful assistant. Please introduce yourself in one sentence."

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
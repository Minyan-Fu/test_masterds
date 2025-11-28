import os
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import snapshot_download
import torch

model_name = "OpenGVLab/InternVL3_5-2B"
local_dir = r"F:/masterds/test/test_masterds/models/InternVL3_5-2B"

if not os.path.exists(local_dir) or len(os.listdir(local_dir)) < 5:
    snapshot_download(
        repo_id=model_name,
        local_dir=local_dir,
        local_dir_use_symlinks=False
    )
else:
    print("Existing.")

tokenizer = AutoTokenizer.from_pretrained(
    local_dir,
    trust_remote_code=True,
    local_files_only=True,
    fix_mistral_regex=True,
)

model = AutoModel.from_pretrained(
    local_dir,
    trust_remote_code=True,
    local_files_only=True,
    torch_dtype="auto"
).eval()

print("Model loaded.")

from PIL import Image
import numpy as np
# 制作一张黑色的假图片
img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))  

prompt = "<image>\nExplain what prompt engineering is."

inputs = tokenizer(prompt, images=img, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
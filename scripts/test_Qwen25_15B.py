import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download
import torch

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
local_dir = r"F:/masterds/test/test_masterds/models/Qwen2.5-1.5B-Instruct"

if not os.path.exists(local_dir) or len(os.listdir(local_dir)) < 5:
    snapshot_download(
        repo_id=model_name,
        local_dir=local_dir,
        local_dir_use_symlinks=False
    )
    print("Downloaded.")
else:
    print("Existing.")

tokenizer = AutoTokenizer.from_pretrained(
    local_dir,
    trust_remote_code=True,
    local_files_only=True
)

model = AutoModelForCausalLM.from_pretrained(
    local_dir,
    trust_remote_code=True,
    local_files_only=True,
    dtype=torch.float32   
).eval()

print("Model loaded.")

# prompt部分
prompt = "Explain what prompt engineering is with a short example."

inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(**inputs, max_new_tokens=200)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)

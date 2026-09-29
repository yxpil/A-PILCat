# -*- coding: utf-8 -*-
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_XET"] = "1"
from huggingface_hub import snapshot_download
p = snapshot_download("lllyasviel/control_v11p_sd15s2_lineart_anime",
                      local_dir="models/cn_lineart_anime",
                      allow_patterns=["*.json", "*.txt", "*.safetensors"])
print("OK ->", p)

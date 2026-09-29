# -*- coding: utf-8 -*-
"""下载 AnimateDiff + SD1.5（走 hf-mirror 国内镜像，fp16 最小集）。"""
import os, sys, time

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = r"C:\Users\Admin\WorkBuddy\2026-09-28-11-58-13\models\hf"
os.environ["HF_HUB_DISABLE_XET"] = "1"   # hf-mirror 不支持 xet 协议，禁用走普通 HTTP

from huggingface_hub import snapshot_download

t0 = time.time()
print("[1/2] AnimateDiff motion adapter (v1-5-2, ~1.7GB)...", flush=True)
snapshot_download(
    "guoyww/animatediff-motion-adapter-v1-5-2",
    local_dir=r"C:\Users\Admin\WorkBuddy\2026-09-28-11-58-13\models\animatediff",
    allow_patterns=["*.json", "*.safetensors", "*.bin"],
)
print("  done %.0fs" % (time.time() - t0), flush=True)

print("[2/2] SD1.5 base (fp16, ~2.5GB)...", flush=True)
snapshot_download(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    local_dir=r"C:\Users\Admin\WorkBuddy\2026-09-28-11-58-13\models\sd15",
    allow_patterns=[
        "*.json", "*.txt",
        "unet/diffusion_pytorch_model.fp16.safetensors",
        "vae/diffusion_pytorch_model.fp16.safetensors",
        "text_encoder/model.fp16.safetensors",
        "unet/diffusion_pytorch_model.safetensors",
        "vae/diffusion_pytorch_model.safetensors",
        "text_encoder/model.safetensors",
    ],
    ignore_patterns=["*.bin", "*nonema*", "tokenizer/*", "*.ckpt"],
)
print("ALL DONE %.0fs" % (time.time() - t0), flush=True)

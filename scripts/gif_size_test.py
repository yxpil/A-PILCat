# -*- coding: utf-8 -*-
"""表情动画 GIF 分辨率实测：超分原生 1024 / 1280 / 源图原生 ~1900
对比体积、耗时、噪点，用于定档。只读取源图，不改任何产物。"""
import os, sys, io, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from PIL import Image
from polish import sr_polish
from pack_sr import SRC, content_bbox
import torch

torch.cuda.reset_peak_memory_stats()
picks = ["打call.png", "WOW.png", "emo.png"]
picks = [p for p in picks if os.path.exists(os.path.join(SRC, p))]

# (输出尺寸, 超分输入上限 lq_max, 说明)
CASES = [
    (1024, 256, "超分原生 x4"),
    (1280, 320, "1.25x 原生-ish"),
    (1900, 475, "源图原生像素"),
]

for name in picks:
    src = Image.open(os.path.join(SRC, name)).convert("RGBA")
    bb = content_bbox(src)
    src = src.crop(bb) if bb else src
    print(f"\n{name}  (内容 {src.size[0]}x{src.size[1]})")
    for size, lq, note in CASES:
        t0 = time.time()
        im = sr_polish(src, size=size, ow=round(2 * size / 300), lq_max=lq)
        dt = time.time() - t0
        # GIF 体积（无抖动，255 色）
        q = im.convert("RGB").quantize(colors=255, method=Image.FASTOCTREE,
                                       dither=Image.Dither.NONE).convert("RGB")
        buf = io.BytesIO()
        q.save(buf, "GIF", save_all=True, duration=0, loop=0, optimize=True)
        kb = buf.tell() / 1024
        # 相对噪点
        a = np.array(im.convert("RGB")).astype(np.int16)
        from scipy.ndimage import median_filter
        med = median_filter(a, size=3)
        dev = np.abs(a - med).mean(axis=2)
        noise = (dev > 24).mean() * 100
        peak = torch.cuda.max_memory_allocated() / 1024 ** 2
        print(f"  {size:>4}px lq{lq:<4} {kb:7.1f}KB  {dt:5.1f}s  相对噪点{noise:4.1f}%  {note}  峰值显存{peak:.0f}MB")
        torch.cuda.reset_peak_memory_stats()
print("\ndone")

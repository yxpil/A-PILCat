# -*- coding: utf-8 -*-
"""实测单张图的超分+打磨：CUDA 占用 vs CPU 占用 vs 耗时"""
import os, sys, time, threading
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from PIL import Image
from pack_sr import SRC, content_bbox, superres, fit_from_big
from polish import sr_polish
import torch

try:
    import psutil
    HAS = True
except ImportError:
    HAS = False

me = psutil.Process() if HAS else None

def sampler(stop, out):
    while not stop.is_set():
        try:
            p = psutil.Process(os.getpid())
            out.append(sum(t.user + t.system for t in p.threads()))
        except Exception:
            pass
        time.sleep(0.05)

names = [n for n in ("打call.png", "有被笑到.png", "淡定.png", "摸鱼.png")
         if os.path.exists(os.path.join(SRC, n))]
if not names:
    sys.exit(0)

print(f"[cuda] {torch.cuda.is_available()}  device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'}")
for name in names:
    src = Image.open(os.path.join(SRC, name)).convert("RGBA")
    bb = content_bbox(src)
    src = src.crop(bb) if bb else src

    stop = threading.Event(); samples = []
    if HAS:
        th = threading.Thread(target=sampler, args=(stop, samples)); th.start()
    t0 = time.time()
    big = superres(src)                       # GPU
    t_sr = time.time() - t0
    out = sr_polish(src, size=300)            # GPU + CPU 后处理
    t_all = time.time() - t0
    stop.set(); th.join() if HAS else None
    if HAS and len(samples) > 1:
        core = (samples[-1] - samples[0]) / (time.time() - t0)
        print(f"{name:<12} 超分 {t_sr*1000:6.0f}ms | 全流程 {t_all*1000:6.0f}ms | 单线程占用约 {core*100:5.1f}%")
    else:
        print(f"{name:<12} 超分 {t_sr*1000:6.0f}ms | 全流程 {t_all*1000:6.0f}ms")

# -*- coding: utf-8 -*-
"""量化边品质：源图透明区里残留的白边像素占比 + 描边是否真的生成"""
import os, sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pack_sr import SRC, fit_from_big, sr_rgba
from polish import sr_polish, add_outline


def report(name):
    p = os.path.join(SRC, name)
    if not os.path.exists(p):
        return
    src = Image.open(p).convert("RGBA")
    bb = src.split()[3].point(lambda v: 255 if v > 24 else 0).getbbox()
    src = src.crop(bb)
    W = 300
    a_src = np.asarray(
        src.split()[3].resize((W, W), Image.LANCZOS)) > 24     # 源图"本该透明"的区域

    def m(im, tag):
        a = np.asarray(im.split()[3])
        rgb = np.asarray(im.convert("RGB")).astype(np.int16)
        lum = rgb.mean(axis=2)
        zone = a_src & (a < 20)                                 # 仍透明的像素
        band = (a > 20) & (a < 235)                             # 抗锯齿过渡带
        w = int(((rgb[..., 0] > 235) & (rgb[..., 1] > 235) & (rgb[..., 2] > 235))[zone].sum())
        print(f"  {tag:<10} 透明区白点 {w:>4}  过渡带像素 {int(band.sum()):>5} "
              f"亮度均值 {float(lum[band].mean()) if band.any() else 0:5.1f}"
              f"  亮度标准差 {float(lum[band].std()) if band.any() else 0:5.1f}")

    print(name)
    m(fit_from_big(sr_rgba(src), W), "旧版")
    m(sr_polish(src, size=W, outline=False), "打磨无描边")
    m(sr_polish(src, size=W, outline=True, ow=2), "打磨+描边")

    a0 = np.asarray(sr_polish(src, size=W, outline=False).split()[3])
    a1 = np.asarray(sr_polish(src, size=W, outline=True, ow=2).split()[3])
    ring = (a0 == 0) & (a1 > 200)
    print(f"  描边环像素 {ring.sum()}")


for n in ["充电中.png", "打call.png"]:
    report(n)

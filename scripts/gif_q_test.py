# -*- coding: utf-8 -*-
"""GIF 画质方案对比：调色板算法 x 抖动开关 x 色数
指标：体积 / 实际色数 / RGB 高频噪点 / 调色板与原图最大色差
"""
import os, sys, io
import numpy as np
from PIL import Image
from scipy.ndimage import median_filter
from pack_sr import SRC, content_bbox
from polish import sr_polish

CFG = [
    ("255+八叉树+抖动(现行)", 255, Image.FASTOCTREE, Image.Dither.FLOYDSTEINBERG),
    ("255+八叉树+无抖动",     255, Image.FASTOCTREE, Image.Dither.NONE),
    ("255+中位切+抖动",       255, Image.MEDIANCUT,  Image.Dither.FLOYDSTEINBERG),
    ("255+中位切+无抖动",     255, Image.MEDIANCUT,  Image.Dither.NONE),
    ("255+最大覆盖+无抖动",   255, Image.MAXCOVERAGE, Image.Dither.NONE),
]


def noise_of(rgba):
    a = np.array(rgba, dtype=np.uint8)
    al = a[..., 3]
    rgb = a[..., :3].astype(np.int16)
    solid = al > 128
    med = median_filter(rgb, size=3)
    return int((np.abs(rgb - med).mean(axis=2) > 24).__and__(solid).sum()), int(solid.sum())


def palette_err(png, gif):
    """GIF 还原像素 vs 原 PNG 像素的平均色差"""
    if gif.mode != "P":
        return -1.0
    g = np.array(gif.convert("RGB"), dtype=np.int16)
    p = np.array(png.convert("RGB"), dtype=np.int16)
    m = min(g.shape[0], p.shape[0])
    return float(np.abs(g[:m] - p[:m]).mean())


def run(name, colors, method, dither):
    rgba = sr_polish(Image.open(os.path.join(SRC, NAME)).convert("RGBA"), size=300)
    flat = Image.new("RGB", rgba.size, (0, 0, 0))
    flat.paste(rgba, mask=rgba.split()[3])
    a = np.array(rgba.split()[3])
    q = flat.quantize(colors=colors, method=method, dither=dither)
    idx = np.array(q, dtype=np.uint8).copy()
    idx[a <= 128] = colors
    p = Image.fromarray(idx, mode="P")
    pal = list(q.getpalette())[:colors * 3] + [0, 0, 0]
    p.putpalette(pal[:768] + [0] * max(0, 768 - len(pal[:768])))
    buf = io.BytesIO()
    p.save(buf, "GIF", save_all=True, duration=0, loop=0,
           transparency=colors, optimize=True)
    n, area = noise_of(p.convert("RGBA"))
    real = len(set(np.array(p).reshape(-1).tolist()))
    return buf.tell(), n, area, real, palette_err(p, rgba.convert("RGB"))


NAME = sys.argv[1] if len(sys.argv) > 1 else "打call.png"
print(f"=== 样本：{NAME} ===")
for tag, c, m, d in CFG:
    try:
        kb, n, area, real, pe = run(tag, c, m, d)
        print(f"{tag:<22} {kb/1024:6.1f}KB  实际色{real:4d}  "
              f"RGB噪点{n:5d}/{area} ({n*100/max(area,1):4.1f}%)  与原图色差{pe:5.2f}")
    except Exception as e:
        print(f"{tag:<22} 失败: {e}")

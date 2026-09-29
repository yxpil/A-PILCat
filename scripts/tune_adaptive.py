# -*- coding: utf-8 -*-
"""自适应降噪 vs 均匀中值：既要去掉超分高频噪，又要保住发丝/手/小线条

指标：
  noise  : |img - med3| > 24 的像素数（越低越好）
  edge   : Sobel 幅值总和（细节/边缘能量，保留率越高越好）
  detail : 细线像素（Sobel 高值）保留数
"""
import os, sys, io, time, warnings
warnings.filterwarnings("ignore")
from PIL import Image, ImageFilter
import numpy as np
from scipy.ndimage import median_filter, sobel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pack_sr import SRC, content_bbox
from polish import sr_polish, denoise_uniform, denoise_auto

OUTD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bgtest")
os.makedirs(OUTD, exist_ok=True)


def stats(rgba):
    rgb = np.array(rgba.convert("RGB")).astype(np.int16)
    a = np.array(rgba.split()[3]) > 128
    med3 = median_filter(rgb, size=3, mode="nearest")
    noise = (np.abs(rgb - med3).max(axis=2) > 24) & a
    g = np.array(rgba.convert("L")).astype(float)
    sx, sy = sobel(g), sobel(g)
    e = np.hypot(sx, sy)
    fine = e > 40                                   # 细线/发丝
    return int(noise.sum()), int(fine.sum()), float(e.sum())


def run():
    picks = ["充电中.png", "打call.png", "已读乱回.png", "笑死.png"]
    picks = [p for p in picks if os.path.exists(os.path.join(SRC, p))]
    for name in picks:
        src = Image.open(os.path.join(SRC, name)).convert("RGBA")
        bb = content_bbox(src)
        src = src.crop(bb) if bb else src
        print(f"\n### {name}")
        base = None
        rows = []
        for tag, kw in (("raw 未降噪", dict(denoise=0.0)),
                        ("mix50 均匀", dict(denoise=0.5, adaptive=False)),
                        ("mix80 均匀(现行)", dict(denoise=0.8, adaptive=False)),
                        ("自适应 s0.8", dict(denoise=0.8, adaptive=True)),
                        ("自适应 s1.0", dict(denoise=1.0, adaptive=True)),
                        ("自适应 s1.5", dict(denoise=1.5, adaptive=True))):
            t0 = time.time()
            im = sr_polish(src, size=300, outline=False, **kw)
            n, f, e = stats(im)
            if base is None:
                base = (n, f, e)
            ob = io.BytesIO(); im.convert("RGBA").save(ob, "PNG", optimize=True)
            rows.append((tag, n, f, e, ob.tell() / 1024, time.time() - t0))
        print(f"  {'档位':<18}{'噪点':>7}{'细节保留':>9}{'边缘能量':>10}"
              f"{'体积KB':>8}{'耗时':>7}")
        for tag, n, f, e, kb, dt in rows:
            print(f"  {tag:<18}{n:>7}{f/base[1]*100:>8.0f}%{e/base[2]*100:>9.0f}%"
                  f"{kb:>8.0f}{dt:>7.1f}s")


if __name__ == "__main__":
    run()

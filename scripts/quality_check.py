# -*- coding: utf-8 -*-
"""产物质量体检：量化像素毛刺 / 半透明噪点 / 色带
只读取，不改任何产物。"""
import os, sys, io, zipfile, tempfile
import numpy as np
from PIL import Image
from scipy.ndimage import label, median_filter, binary_closing, binary_opening

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pack_sr import OUT

K3 = np.ones((3, 3), bool)


def metrics(im):
    im = im.convert("RGBA")
    a = np.array(im, dtype=np.uint8)
    al = a[..., 3]
    rgb = a[..., :3].astype(np.int16)
    solid = al > 128
    # 1) 半透明噪点：既不是实边也不是干净的透明过渡
    semi = ((al > 25) & (al < 235))
    # 2) 孤立实心点 / 细刺
    from scipy.ndimage import binary_dilation, binary_erosion
    from scipy.signal import convolve2d
    solo = solid & ~binary_closing(solid, structure=K3)         # 8邻域补点后仍是孤立
    cnt = convolve2d(solid.astype(float), K3.astype(float), mode="same")
    spike = solid & (cnt <= 2)                                   # 邻居<=2 个实心 => 细刺
    # 3) RGB 孤立噪点（3x3 中值偏离大）
    med = median_filter(rgb, size=3)
    dev = np.abs(rgb - med).mean(axis=2)
    noise = (dev > 24) & solid
    # 4) 颜色数
    cols = len(np.unique(a.reshape(-1, 4), axis=0))
    return dict(semi=semi, solo=solo, spike=spike, noise=noise,
                cols=cols, solid=solid)


def rep(tag, im):
    m = metrics(im)
    area = m["solid"].sum()
    print(f"{tag:<34} 半透明{m['semi'].sum():>5} "
          f"孤立点{m['solo'].sum():>4} 细刺{m['spike'].sum():>5} "
          f"RGB噪点{m['noise'].sum():>5} 色数{m['cols']:>6} "
          f"(实体{area})")


def main():
    tmp = tempfile.mkdtemp(prefix="qc_")
    dirs = [d for d in sorted(os.listdir(OUT))
            if os.path.isdir(os.path.join(OUT, d)) and d != "Example"]
    for d in dirs:
        pd = os.path.join(OUT, d)
        print(f"--- {d} ---")
        z = os.path.join(pd, "表情缩略图.zip")
        with zipfile.ZipFile(z) as f:
            ns = f.namelist()
            for n in ns[:2] + ns[len(ns) // 2:len(ns) // 2 + 1]:
                im = Image.open(f.open(n))
                rep("缩略图 " + n, im)
        g = os.path.join(pd, "表情动画.zip")
        if os.path.exists(g):
            with zipfile.ZipFile(g) as f:
                im = Image.open(f.open(f.namelist()[0]))
                rep("动画   " + f.namelist()[0] + " [GIF]", im)
        c = os.path.join(pd, "表情封面图.png")
        if os.path.exists(c):
            rep("封面   200x200", Image.open(c))


if __name__ == "__main__":
    main()

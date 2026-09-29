# -*- coding: utf-8 -*-
"""按平台规范逐条校验 上传_重绘：GIF/缩略图/封面 + 命名 + zip 编码。"""
import os, io, re, zipfile, sys
import numpy as np
from PIL import Image

OUT = r"C:/Users/Admin/OneDrive/Desktop/上传_重绘"
dirs = [d for d in sorted(os.listdir(OUT))
        if os.path.isdir(os.path.join(OUT, d)) and not d.startswith("_")]
fail, warn_n = [], 0
for d in dirs:
    p = os.path.join(OUT, d)
    zg = zipfile.ZipFile(os.path.join(p, "表情动画.zip"))
    zp = zipfile.ZipFile(os.path.join(p, "表情缩略图.zip"))
    ng, npn = sorted(zg.namelist()), sorted(zp.namelist())
    # 一一对应
    if [n[:-4] for n in ng] != [n[:-4] for n in npn]:
        fail.append(f"{d}: 缩略图与 GIF 命名不一一对应")
    for n in ng:
        if not re.fullmatch(r"[^0-9\W]{1,4}_\d{2}\.gif", n, re.UNICODE):
            fail.append(f"{d}/{n}: 命名不规范")
        b = zg.read(n)
        if len(b) >= 300 * 1024:
            fail.append(f"{d}/{n}: {len(b)/1024:.0f}KB >= 300KB")
        im = Image.open(io.BytesIO(b))
        if im.size != (300, 300):
            fail.append(f"{d}/{n}: 画布 {im.size}")
        a = np.array(im.convert("RGBA"))
        al = a[..., 3]
        if (al > 128).mean() < 0.05:
            fail.append(f"{d}/{n}: 第一帧疑似空白")
        # 白描边：内容最外圈像素应为近白
        ys, xs = np.where(al > 128)
        edge = [a[ys.min() + 1, (xs.min() + xs.max()) // 2],
                a[ys.max() - 1, (xs.min() + xs.max()) // 2]]
        if min(e[:3].min() for e in edge) < 200:
            fail.append(f"{d}/{n}: 外圈非白描边 {edge[0][:3]}")
        # 透明底
        if (al < 128).mean() < 0.05:
            fail.append(f"{d}/{n}: 无明显透明底")
    for n in npn:
        b = zp.read(n)
        if len(b) >= 200 * 1024:
            fail.append(f"{d}/{n}: {len(b)/1024:.0f}KB >= 200KB")
        im = Image.open(io.BytesIO(b))
        if im.size != (300, 300):
            fail.append(f"{d}/{n}: 画布 {im.size}")
    cov = os.path.join(p, "表情封面图.png")
    if not os.path.exists(cov):
        fail.append(f"{d}: 缺封面")
    else:
        kb = os.path.getsize(cov) / 1024
        ci = Image.open(cov)
        if ci.size != (200, 200) or kb >= 100:
            fail.append(f"{d}: 封面 {ci.size} {kb:.0f}KB")
    # zip UTF-8
    for z in (zg, zp):
        if not all(i.flag_bits & 0x800 for i in z.infolist()):
            fail.append(f"{d}: zip 非 UTF-8 编码")
    print(f"{d}: gif{len(ng)} png{len(npn)} 封面{'OK' if os.path.exists(cov) else '缺'}")

print("\n" + ("ALL OK 全部符合平台规范" if not fail else f"发现 {len(fail)} 处问题:"))
for f in fail[:30]:
    print("  -", f)

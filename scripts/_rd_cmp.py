# -*- coding: utf-8 -*-
"""重绘质量 1:1 细节对比板：原图 vs 768重绘 vs 896重绘+超分。
同一人脸区域等比例裁切 + 3 倍放大，直接看线条/清晰度/文字。
"""
import os, sys
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pack_sr import SRC, content_bbox
from matting_gpu import matte
from head_motion import detect_face
from redraw3 import prep, SEEDS

OUT = "redraw3"
HERE = os.path.dirname(os.path.abspath(__file__))
ZOOM = 3
WIN = 210                      # 展示窗口（源图坐标系）


def white(im):
    b = Image.new("RGB", im.size, (255, 255, 255))
    b.paste(im, mask=im.split()[3] if im.mode == "RGBA" else None)
    return b


for kw in ["打call", "吃东西", "加油"]:
    name = kw + ".png"
    src = Image.open(os.path.join(SRC, name)).convert("RGBA")
    m = matte(src)
    bb = content_bbox(m)
    m = m.crop(bb) if bb else m
    W0, H0 = m.size

    # 脸区域（源图坐标）
    face = detect_face(m)
    if face:
        x, y, w, h = face
        cx, cy = x + w / 2, y + h * 0.75
    else:
        cx, cy = W0 * 0.5, H0 * 0.42
    box0 = (int(cx - WIN / 2), int(cy - WIN / 2), int(cx + WIN / 2), int(cy + WIN / 2))

    cols = []

    def add(label, im, scale_x, scale_y=None):
        """im 在源图坐标系里的比例 -> 该版本上的窗口"""
        sy = scale_y or scale_x
        b = (max(0, int(box0[0] * scale_x)), max(0, int(box0[1] * sy)),
             min(im.width, int(box0[2] * scale_x)), min(im.height, int(box0[3] * sy)))
        c = im.crop(b).resize((WIN * ZOOM, WIN * ZOOM), Image.LANCZOS)
        cols.append((label, c))

    add("原图(源图原生)", white(m), 1.0)
    for v, lb in [("", "重绘768 s11"), ("_w896_up2", "重绘896+超分 s11"),
                  ("_w1024_st0.45_sh30", "重绘1024+锐化 s11")]:
        for sd in SEEDS[:1]:
            p = os.path.join(OUT, f"{kw}_s{sd}{v}.png")
            if not os.path.exists(p):
                continue
            im = Image.open(p).convert("RGB")
            add(f"{lb}", im, im.width / W0, im.height / H0)

    if not cols:
        print("跳过", kw); continue
    W = len(cols) * (WIN * ZOOM + 8) + 8
    sh = Image.new("RGB", (W, WIN * ZOOM + 30), (246, 246, 246))
    dr = ImageDraw.Draw(sh)
    for i, (lab, c) in enumerate(cols):
        sh.paste(c, (8 + i * (WIN * ZOOM + 8), 24))
        dr.text((10 + i * (WIN * ZOOM + 8), 7), lab, fill=(20, 20, 20))
    o = os.path.join(HERE, f"重绘对比_{kw}.png")
    sh.save(o)
    print("saved", o, sh.size, [c[0] for c in cols])

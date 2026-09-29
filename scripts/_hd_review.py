# -*- coding: utf-8 -*-
"""上传_高清 验收拼版：
 1) 动画 GIF 全览（10 组 x 16 个，每格 110px 取 GIF 中间帧）
 2) 静态缩略图抽检（第01包 原比例缩小）
 3) 全部封面 200px 原尺寸
"""
import os, io, zipfile, math
import numpy as np
from PIL import Image, ImageDraw

HD = r"C:\Users\Admin\OneDrive\Desktop\上传_高清"
HERE = os.path.dirname(os.path.abspath(__file__))
BG, INK = (248, 248, 248), (25, 25, 25)


def tile_rgba(im, t):
    bg = Image.new("RGB", (im.width, im.height), (255, 255, 255))
    bg.paste(im, mask=im.split()[3])
    return bg.resize((t, t), Image.LANCZOS)


dirs = [d for d in sorted(os.listdir(HD)) if os.path.isdir(os.path.join(HD, d))]

# ---------- 1) GIF 全览 ----------
TH, COLS, pad = 110, 16, 16
rows = []
for d in dirs:
    with zipfile.ZipFile(os.path.join(HD, d, "表情动画.zip")) as z:
        ims = []
        for n in sorted(z.namelist()):
            g = Image.open(io.BytesIO(z.read(n)))
            g.seek(g.n_frames // 2)
            ims.append(g.convert("RGBA"))
    rows.append((d, ims))
W = COLS * (TH + 4) + 8
H = len(rows) * (TH + pad) + 10
sh = Image.new("RGB", (W, H), BG)
dr = ImageDraw.Draw(sh)
y = 6
for d, ims in rows:
    dr.text((6, y + 2), d, fill=INK)
    for i, im in enumerate(ims):
        sh.paste(tile_rgba(im, TH), (6 + i * (TH + 4), y + pad))
    y += TH + pad
p1 = os.path.join(HERE, "验收HD_动画全览.png")
sh.save(p1)

# ---------- 2) 静态抽检（第01包 全 16 张） ----------
TH2 = 118
with zipfile.ZipFile(os.path.join(HD, dirs[0], "表情缩略图.zip")) as z:
    st = [Image.open(io.BytesIO(z.read(n))).convert("RGBA")
          for n in sorted(z.namelist())]
W2 = COLS * (TH2 + 4) + 8
H2 = (TH2 + pad) + 26
sh2 = Image.new("RGB", (W2, H2), BG)
dr2 = ImageDraw.Draw(sh2)
dr2.text((6, 4), f"{dirs[0]} 静态缩略图（1900x1900 原图等比缩小预览）", fill=INK)
for i, im in enumerate(st):
    sh2.paste(tile_rgba(im, TH2), (6 + i * (TH2 + 4), 22))
p2 = os.path.join(HERE, "验收HD_静态抽检.png")
sh2.save(p2)

# ---------- 3) 封面 ----------
cols = 5
rr = math.ceil(len(dirs) / cols)
sh3 = Image.new("RGB", (cols * 208 + 8, rr * 226 + 8), BG)
dr3 = ImageDraw.Draw(sh3)
for i, d in enumerate(dirs):
    c = Image.open(os.path.join(HD, d, "表情封面图.png")).convert("RGBA")
    x, y = 8 + (i % cols) * 208, 8 + (i // cols) * 226
    sh3.paste(tile_rgba(c, 200), (x, y))
    dr3.text((x, y + 202), d, fill=INK)
p3 = os.path.join(HERE, "验收HD_封面.png")
sh3.save(p3)
print("saved:", p1, p2, p3, sep="\n  ")

# -*- coding: utf-8 -*-
"""最终验收拼版：把所有包的 表情动画.zip 里的 GIF 按包分行拼成大图。"""
import os, io, sys, zipfile
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pack_sr import OUT

TH = 132          # 每格边长
COLS = 16
base = OUT
dirs = [d for d in sorted(os.listdir(base))
        if os.path.isdir(os.path.join(base, d)) and d != "Example"]

rows = []
total = 0
for d in dirs:
    zp = os.path.join(base, d, "表情动画.zip")
    ims = []
    with zipfile.ZipFile(zp) as z:
        for n in sorted(z.namelist()):
            ims.append((n, Image.open(io.BytesIO(z.read(n))).convert("RGBA")))
    rows.append((d, ims))
    total += len(ims)

W = COLS * (TH + 6) + 6
row_h = TH + 24
H = len(rows) * row_h + 10
sheet = Image.new("RGB", (W, H), (250, 250, 250))
dr = ImageDraw.Draw(sheet)
y = 6
for d, ims in rows:
    dr.text((8, y + 2), f"{d} ({len(ims)})", fill=(30, 30, 30))
    for i, (n, im) in enumerate(ims):
        t = im.resize((TH, TH), Image.LANCZOS)
        x = 6 + i * (TH + 6)
        sheet.paste(t, (x, y + 16), t)
    y += row_h

p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "验收_全部GIF.png")
sheet.save(p)
print(f"{total} 张 -> {p}  {sheet.size}")

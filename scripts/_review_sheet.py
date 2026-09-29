# -*- coding: utf-8 -*-
"""验收拼版：GIF 全览 + 封面九宫 + 缩略图抽检，一次看清新版 BOX 管线。"""
import os, io, zipfile
from PIL import Image, ImageDraw

OUT = r"C:/Users/Admin/OneDrive/Desktop/上传"
dirs = [d for d in sorted(os.listdir(OUT))
        if os.path.isdir(os.path.join(OUT, d)) and d != "Example"]
BG = (250, 250, 250)
INK = (25, 25, 25)
GREY = (150, 150, 150)


def tile(im, t, label=None):
    im = im.convert("RGBA")
    bg = Image.new("RGB", (t, t), (255, 255, 255))
    bg.paste(im, mask=im.split()[3])
    return bg.resize((t, t), Image.LANCZOS)


# ---------- 1) GIF 全览 ----------
TH, COLS = 120, 16
pad = 18
rows = []
for d in dirs:
    with zipfile.ZipFile(os.path.join(OUT, d, "表情动画.zip")) as z:
        ims = [Image.open(io.BytesIO(z.read(n))).convert("RGBA") for n in sorted(z.namelist())]
    rows.append((d, ims))
W = COLS * (TH + 4) + 4
H = len(rows) * (TH + pad) + 10
sheet = Image.new("RGB", (W, H), BG)
dr = ImageDraw.Draw(sheet)
y = 6
for d, ims in rows:
    dr.text((6, y + 2), d, fill=INK)
    for i, im in enumerate(ims):
        sheet.paste(tile(im, TH), (4 + i * (TH + 4), y + pad))
    y += TH + pad
p1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "验收_全部GIF_BOX版.png")
sheet.save(p1)

# ---------- 2) 封面 + 缩略图抽检 ----------
TH2 = 108
rowA, rowB = [], []
for d in dirs:
    cov = Image.open(os.path.join(OUT, d, "表情封面图.png")).convert("RGBA")
    rowA.append((d, cov))
with zipfile.ZipFile(os.path.join(OUT, "第03包", "表情缩略图.zip")) as z:
    ths = [Image.open(io.BytesIO(z.read(n))).convert("RGBA") for n in sorted(z.namelist())]
W2 = COLS * (TH2 + 5) + 5
sheet2 = Image.new("RGB", (W2, 2 * (TH2 + pad) + 34), BG)
dr2 = ImageDraw.Draw(sheet2)
y = 8
dr2.text((6, y), "表情封面图（每包一张，200x200）", fill=INK)
y += 20
for d, cov in rowA:
    sheet2.paste(tile(cov, TH2), (4 + len(rowA) * 0, 0)) if False else None
for i, (d, cov) in enumerate(rowA):
    t = tile(cov, TH2)
    sheet2.paste(t, (4 + i * (TH2 + 5), y))
y += TH2 + pad
dr2.text((6, y), "第03包 表情缩略图（300x300 PNG，抽检）", fill=INK)
y += 20
for i, t in enumerate([tile(t, TH2) for t in ths]):
    sheet2.paste(t, (4 + i * (TH2 + 5), y))
p2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "验收_封面与缩略图_BOX版.png")
sheet2.save(p2)
print("GIF 全览", sheet.size, "->", p1)
print("封面/缩略图", sheet2.size, "->", p2)
print("包数", len(dirs))

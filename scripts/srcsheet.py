# -*- coding: utf-8 -*-
"""把素材缩略拼成小图，便于快速筛选"无文字"候选"""
import os, sys
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pack_sr import SRC, collect_items

CELL = 132          # 单格边长（贴纸本体居中，留白裁剪）
COLS = 8
OUTD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "srcpick")
os.makedirs(OUTD, exist_ok=True)


def main():
    items = collect_items()
    n = len(items)
    rows = (n + COLS - 1) // COLS
    W = COLS * CELL
    H = rows * CELL
    sheet = Image.new("RGB", (W, H), (250, 250, 250))
    from PIL import ImageDraw
    dr = ImageDraw.Draw(sheet)
    for i, name in enumerate(items):
        im = Image.open(os.path.join(SRC, name)).convert("RGBA")
        bb = im.split()[3].point(lambda p: 255 if p > 24 else 0).getbbox()
        im = im.crop(bb) if bb else im
        im.thumbnail((CELL - 4, CELL - 4), Image.LANCZOS)
        x = (i % COLS) * CELL + (CELL - im.width) // 2
        y = (i // COLS) * CELL + (CELL - im.height) // 2
        sheet.paste(im, (x, y), im)
        dr.rectangle([x - 2, y - 2, x + im.width + 1, y + im.height + 1],
                     outline=(215, 215, 215))
        dr.text((x, y - 9), f"{i}", fill=(220, 0, 0))
    sheet.save(os.path.join(OUTD, "all.png"))
    # 分片，避免单张过大
    per = 4
    for k in range((rows + per - 1) // per):
        top, bot = k * per, min(rows, (k + 1) * per)
        part = sheet.crop((0, top * CELL, W, bot * CELL))
        part.save(os.path.join(OUTD, f"part{k}.png"))
        print(f"part{k}.png  rows {top}-{bot}  ({len(items[top:bot])} 张)")
    print("总数", n, "-> all.png", sheet.size)


if __name__ == "__main__":
    main()

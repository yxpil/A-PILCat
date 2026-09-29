# -*- coding: utf-8 -*-
"""风格样张：旧版 vs 4 种打磨风格（整图 + 局部 3 倍放大），挑一档再全量重跑"""
import os, io, sys, time
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pack_sr import SRC, fit_from_big, sr_rgba, content_bbox
from polish import sr_polish

STYLES = [
    ("A 现行版",   dict(mode="legacy")),
    ("B 干净无描边", dict(mode="clean")),
    ("C 描边2px", dict(mode="outline2")),
    ("D 描边3px", dict(mode="outline3")),
    ("E 描边2+锐", dict(mode="crisp")),
]

def apply_style(src, mode, W=300):
    if mode == "legacy":
        return fit_from_big(sr_rgba(src), W)
    kw = dict(size=W, outline=True, sharpen=False)
    if mode == "clean":
        return sr_polish(src, size=W, outline=False, sharpen=False)
    if mode == "outline2":
        return sr_polish(src, size=W, outline=True, ow=2, sharpen=False)
    if mode == "outline3":
        return sr_polish(src, size=W, outline=True, ow=3, sharpen=False)
    if mode == "crisp":
        return sr_polish(src, size=W, outline=True, ow=2, sharpen=True)
    raise ValueError(mode)


def three_x(im, box):
    c = im.crop(box)
    return c.resize((c.width * 3, c.height * 3), Image.LANCZOS)


def main():
    picks = ["充电中.png", "打call.png", "已读乱回.png"]
    picks = [p for p in picks if os.path.exists(os.path.join(SRC, p))]
    os.makedirs("cmp_polish", exist_ok=True)
    PAD = 8
    for name in picks:
        src = Image.open(os.path.join(SRC, name)).convert("RGBA")
        bb = content_bbox(src)
        src = src.crop(bb) if bb else src
        t0 = time.time()

        outs, kbs = [], []
        for tag, cfg in STYLES:
            im = apply_style(src, cfg["mode"])
            ob = io.BytesIO(); im.save(ob, "PNG", optimize=True)
            outs.append((tag, im, cfg["mode"]))
            kbs.append(ob.tell() / 1024)
            print(f"  {name} {tag}: {kbs[-1]:.0f}KB")

        # 整图行
        cols = len(outs) * 300 + (len(outs) + 1) * PAD
        row_wide = Image.new("RGB", (cols, 300 + PAD * 2), (246, 246, 248))
        for i, (tag, im, m) in enumerate(outs):
            x = PAD + i * (300 + PAD)
            row_wide.paste(im, (x, PAD))
        dr = ImageDraw.ImageDraw(row_wide, "RGB")
        for i, (tag, im, m) in enumerate(outs):
            x = PAD + i * (300 + PAD)
            dr.text((x + 2, 300 + 2), f"{tag} {kbs[i]:.0f}KB", fill=(40, 40, 40))
        row_wide.save(f"cmp_polish/sheet_{name}.png")

        # 局部放大行（上半部分 3 倍）
        lb = (int(outs[0][1].width * 0.20), int(outs[0][1].height * 0.02),
              int(outs[0][1].width * 0.80), int(outs[0][1].height * 0.50))
        zs = [three_x(im, lb) for _t, im, _m in outs]
        zh = max(z.height for z in zs)
        zw = len(zs) * zs[0].width + (len(zs) + 1) * PAD
        row_zoom = Image.new("RGB", (zw, zh + PAD * 2), (246, 246, 248))
        for i, z in enumerate(zs):
            row_zoom.paste(z.convert("RGB"), (PAD + i * (zs[0].width + PAD), PAD))
        row_zoom.save(f"cmp_polish/sheet_zoom_{name}.png")

        print(f"{name}: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    from PIL import ImageDraw
    main()

# -*- coding: utf-8 -*-
"""降噪档位扫参：在最终像素空间混合 3x3 中值，量化噪点/体积，选一档"""
import os, io, sys, time
import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pack_sr import SRC, content_bbox, fit_from_big, superres, save_png_under
from polish import despill, soften
from quality_check import metrics

OUTD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bgtest")
os.makedirs(OUTD, exist_ok=True)


def polish(src_rgba, size=300, blend=0.6, alpha_clean=False, outline=True, ow=2):
    desp, ext = despill(src_rgba, n=1)
    a_low = src_rgba.split()[3]
    big = superres(desp)
    na = a_low.resize(big.size, Image.LANCZOS) \
               .filter(ImageFilter.GaussianBlur(0.4)) \
               .point(lambda p: 255 if p > 150 else (0 if p < 85 else p))
    out = big.convert("RGBA"); out.putalpha(na)
    out = soften(out)
    out = fit_from_big(out, size)
    if blend > 0:
        med = out.filter(ImageFilter.MedianFilter(3))
        out = Image.blend(out, med, blend)
        if alpha_clean:
            a_im = out.split()[3]
            a = np.array(a_im, dtype=np.int16)
            a = np.where(a < 40, 0, np.where(a >= 215, 255, a))
            o = out.copy()
            o.putalpha(Image.fromarray(a.astype(np.uint8)))
            out = o
    if outline:
        from polish import add_outline
        out = add_outline(out, width=ow)
    return out


def zoom(im, k=4):
    w = int(im.width * 0.56); h = int(im.height * 0.5)
    box = (int(im.width * 0.20), int(im.height * 0.06),
           int(im.width * 0.20) + w, int(im.height * 0.06) + h)
    c = im.crop(box)
    return c.resize((c.width * k, c.height * k), Image.LANCZOS)


def main():
    picks = ["打call.png", "充电中.png", "有被笑到.png", "笑死.png"]
    picks = [p for p in picks if os.path.exists(os.path.join(SRC, p))]
    CFGS = [("A 现行", 0.0, False), ("B mix50", 0.5, False),
            ("C mix65", 0.65, False), ("D mix80", 0.8, False),
            ("E mix65+alpha", 0.65, True)]
    rows = []
    for name in picks:
        src = Image.open(os.path.join(SRC, name)).convert("RGBA")
        bb = content_bbox(src)
        src = src.crop(bb) if bb else src
        ims = []
        for tag, b, ac in CFGS:
            t0 = time.time()
            im = polish(src, 300, b, ac)
            ims.append(im)
            m = metrics(im)
            buf = io.BytesIO(); im.save(buf, "PNG", optimize=True)
            rows.append((name, tag, m["noise"].sum(), m["semi"].sum(),
                         m["cols"], buf.tell() / 1024, time.time() - t0))
        # 对比拼版
        zh = []
        for tag, im in zip([c[0] for c in CFGS], ims):
            z = zoom(im)
            zh.append((tag, z))
        pad = 8
        W = sum(z.width for _, z in zh) + pad * (len(zh) + 1)
        H = max(z.height for _, z in zh)
        board = Image.new("RGB", (W, H + 22), (246, 246, 248))
        x = pad
        from PIL import ImageDraw
        dr = ImageDraw.Draw(board)
        for tag, z in zh:
            board.paste(z, (x, pad)); dr.text((x + 2, H + 4), tag, fill=(30, 30, 30))
            x += z.width + pad
        board.save(os.path.join(OUTD, f"denoise_{name}.png"))
    print(f"{'图':<10}{'档':<14}{'RGB噪点':>8}{'半透明':>8}{'色数':>8}{'体积KB':>8}{'耗时s':>7}")
    for name, tag, nz, sm, cols, kb, t in rows:
        print(f"{name:<10}{tag:<14}{nz:>8}{sm:>8}{cols:>8}{kb:>8.0f}{t:>7.1f}")


if __name__ == "__main__":
    main()

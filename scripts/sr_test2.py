# -*- coding: utf-8 -*-
"""画质对比：硬缩放 vs 超分+去噪 vs 超分
用法: python sr_test2.py 打call.png 比心.png
"""
import os, io, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
from PIL import Image, ImageFilter
from rrdb import load_model

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = r"C:\Users\Admin\OneDrive\Desktop\合格"
OUT = os.path.join(HERE, "cmp_out")
os.makedirs(OUT, exist_ok=True)

net, DEV = load_model()
print("device:", DEV)


def to_t(im):
    a = np.array(im).astype(np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(DEV)


def from_t(t):
    a = t.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    return Image.fromarray(np.clip(a * 255.0, 0, 255).astype(np.uint8), "RGB")


def denoise(rgb, strength=1):
    """轻度保边降噪：中值滤波压噪点，保留线稿锐度"""
    out = rgb
    for _ in range(strength):
        out = out.filter(ImageFilter.MedianFilter(size=3))
    return out


def superres(rgb, lq_max=256):
    """模型要求 LQ 输入尺度，先缩到 lq_max 再 x4"""
    if max(rgb.width, rgb.height) > lq_max:
        s = lq_max / max(rgb.width, rgb.height)
        rgb = rgb.resize((max(1, round(rgb.width * s)), max(1, round(rgb.height * s))),
                         Image.LANCZOS)
    with torch.no_grad():
        return from_t(net(to_t(rgb)))


def process(im, size=300, margin=6, mode="sr_dn", lq_max=256, stroke_px=0):
    bb = im.split()[3].point(lambda p: 255 if p > 24 else 0).getbbox()
    src = im.crop(bb)
    s = min((size - margin * 2) / src.width, (size - margin * 2) / src.height)
    nw, nh = max(1, round(src.width * s)), max(1, round(src.height * s))

    if mode == "hard":
        fixed = src.resize((nw, nh), Image.LANCZOS)
    else:
        a = src.split()[3]
        flat = Image.new("RGB", src.size, (255, 255, 255))
        flat.paste(src, mask=a)
        if mode == "sr_dn":
            flat = denoise_uniform(flat)
        big = superres(flat, lq_max)
        # alpha 同步放大 + 软阈值，去半透明毛边
        na = a.resize(big.size, Image.LANCZOS).filter(ImageFilter.GaussianBlur(0.5)) \
              .point(lambda p: 255 if p > 170 else (0 if p < 90 else p))
        big = big.convert("RGBA"); big.putalpha(na)
        fixed = big.resize((nw * 4, nh * 4), Image.LANCZOS).resize((nw, nh), Image.LANCZOS)

    if stroke_px:
        d = fixed.split()[3].filter(ImageFilter.MaxFilter(stroke_px * 2 + 1))
        st = Image.new("RGBA", fixed.size, (255, 255, 255, 0))
        st.paste(Image.new("RGBA", fixed.size, (255, 255, 255, 255)), mask=d)
        fixed = Image.alpha_composite(st, fixed)

    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    canvas.paste(fixed, ((size - nw) // 2, (size - nh) // 2), fixed)
    return canvas


def save_png(im, path, limit):
    best, used = None, None
    for colors in [None, 256, 192, 128, 96, 64]:
        buf = io.BytesIO()
        out = im if colors is None else im.quantize(
            colors=colors, method=Image.FASTOCTREE,
            dither=Image.Dither.FLOYDSTEINBERG).convert("RGBA")
        out.save(buf, "PNG", optimize=True)
        if best is None or buf.tell() < len(best):
            best, used = buf.getvalue(), (colors or "full")
        if buf.tell() <= limit:
            break
    open(path, "wb").write(best)
    return len(best), used


names = sys.argv[1:] or ["打call.png"]
for n in names:
    im = Image.open(os.path.join(SRC_DIR, n)).convert("RGBA")
    print(f"\n=== {n} {im.size} ===")
    res = {}
    for tag, mode in [("硬缩放", "hard"), ("超分+去噪", "sr_dn"), ("纯超分", "sr")]:
        t = time.time()
        res[tag] = process(im, mode=mode)
        print(f"  {tag}: {time.time()-t:.1f}s")

    bg = Image.new("RGB", (300 * 3, 320), (248, 248, 248))
    for i, (tag, x) in enumerate(res.items()):
        c = Image.new("RGB", x.size, (255, 255, 255))
        c.paste(x, mask=x.split()[3])
        bg.paste(c, (i * 300 + 6, 16))
    bg.save(os.path.join(OUT, f"row_{n[:-4]}.png"))

    for tag, x in res.items():
        b, u = save_png(x, os.path.join(OUT, f"{tag}_{n}"), 200 * 1024)
        print(f"  {tag}: {b/1024:.0f} KB (色数 {u})")

# 局部放大对比（眼睛/脸部，找毛刺）
im = Image.open(os.path.join(SRC_DIR, names[0])).convert("RGBA")
zz = Image.new("RGB", (320 * 3, 320), (248, 248, 248))
for i, (tag, x) in enumerate(res.items()):
    c = Image.new("RGB", x.size, (255, 255, 255))
    c.paste(x, mask=x.split()[3])
    zz.paste(c.crop((65, 60, 225, 220)).resize((320, 320), Image.NEAREST), (i * 320, 0))
zz.save(os.path.join(OUT, "zoom_face.png"))
print("\ndone ->", OUT)

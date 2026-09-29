# -*- coding: utf-8 -*-
"""单张对比测试：硬缩放 vs 超分后缩放"""
import os, time
import numpy as np
import torch
from PIL import Image, ImageFilter
from spandrel import ModelLoader

MODEL = os.path.join(os.path.dirname(__file__), "models", "RealESRGAN_x4plus.pth")
SRC = r"C:\Users\Admin\OneDrive\Desktop\合格\打call.png"
OUT = os.path.dirname(__file__)

model = ModelLoader().load_from_file(MODEL)
print("model load ok, scale:", getattr(model, "scale", None))


def to_tensor(im):
    arr = np.array(im).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def from_tensor(t):
    a = t.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    return Image.fromarray(np.clip(a * 255.0, 0, 255).astype(np.uint8), "RGB")


def sr_upscale(im):
    """RGBA -> 超分 x4（alpha 同步放大并平滑）"""
    a = im.split()[3]
    flat = Image.new("RGB", im.size, (255, 255, 255))
    flat.paste(im, mask=a)
    with torch.no_grad():
        out = from_tensor(model(to_tensor(flat)))
    na = a.resize(out.size, Image.LANCZOS)
    na = na.filter(ImageFilter.GaussianBlur(0.6)).point(
        lambda p: 255 if p > 170 else (0 if p < 90 else p))
    res = out.convert("RGBA")
    res.putalpha(na)
    return res


def fit(im, size, margin=6, use_sr=True):
    bb = im.split()[3].point(lambda p: 255 if p > 24 else 0).getbbox()
    src = im.crop(bb)
    s = min((size - margin * 2) / src.width, (size - margin * 2) / src.height)
    nw, nh = max(1, round(src.width * s)), max(1, round(src.height * s))
    if use_sr:
        big = sr_upscale(src)
        src = big.resize((nw * 4, nh * 4), Image.LANCZOS).resize((nw, nh), Image.LANCZOS)
    else:
        src = src.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    canvas.paste(src, ((size - nw) // 2, (size - nh) // 2), src)
    return canvas


def sheet(path, variants):
    bg = Image.new("RGB", (300 * len(variants), 320), (250, 250, 250))
    for i, x in enumerate(variants):
        c = Image.new("RGB", x.size, (255, 255, 255))
        c.paste(x, mask=x.split()[3])
        bg.paste(c, (i * 300 + 5, 15))
    bg.save(path)


im = Image.open(SRC).convert("RGBA")
print("source:", im.size)

t = time.time(); plain = im.resize((300, 300), Image.LANCZOS)
print("lanczos 1:1 pad:", round(time.time() - t, 2), "s")

t = time.time(); hard = fit(im, 300, use_sr=False)
print("hard fit:", round(time.time() - t, 2), "s")

t = time.time(); sr4 = fit(im, 300, use_sr=True)
print("sr fit (x4):", round(time.time() - t, 2), "s")

sheet(os.path.join(OUT, "cmp_300.png"), [plain, hard, sr4])

# 边缘细节放大对比（脸部区域截图）
crops = []
for x in [plain, hard, sr4]:
    crops.append(x.crop((70, 60, 230, 220)).resize((320, 320), Image.NEAREST))
sheet(os.path.join(OUT, "cmp_zoom.png"), crops)
print("done")

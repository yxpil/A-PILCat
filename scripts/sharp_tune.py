# -*- coding: utf-8 -*-
"""300px 最终锐化参数寻优：既要锐，又不能出光晕/毛刺。"""
import os, sys, numpy as np
from PIL import Image, ImageFilter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pack_sr import SRC, content_bbox
from polish import sr_polish, final_sharp, denoise_auto, add_outline
from down import down_chain
from matting_gpu import matte

MASTER, LQ_MAX = 1900, 475
picks = ["打call.png", "充电中.png", "有被笑到.png", "看戏.png"]
picks = [p for p in picks if os.path.exists(os.path.join(SRC, p))]
os.makedirs("sharp_tune", exist_ok=True)

def lap_var(im):
    """拉普拉斯方差 = 锐度指标，越大越锐利"""
    a = np.array(im.convert("L"), dtype=np.float64)
    k = np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=np.float64)
    from scipy.ndimage import convolve
    return float(convolve(a, k, mode="nearest").var())

def halo_score(im):
    """光晕检测：亮边内侧的过冲。>0 说明出现白色描边状 halo"""
    rgb = np.array(im.convert("RGB"), dtype=np.float64)
    blur = np.array(Image.fromarray(np.clip(rgb,0,255).astype(np.uint8))
                    .filter(ImageFilter.GaussianBlur(2.0)), dtype=np.float64)
    d = rgb - blur
    m = np.array(im.split()[3]) > 200
    return float(np.clip(d[m], 0, None).mean())

print(f"{'参数':<34}{'锐度':>9}{'光晕':>9}")
print("-" * 54)
for name in picks:
    src = matte(Image.open(os.path.join(SRC, name)).convert("RGBA"))
    master = sr_polish(src, size=MASTER, ow=0, lq_max=LQ_MAX, denoise=0.0)
    variants = [
        ("无锐化(基准)",            dict(sharpen=0,   denoise=0.45, ow=2)),
        ("锐40",                   dict(sharpen=40,  denoise=0.45, ow=2)),
        ("锐70",                   dict(sharpen=70,  denoise=0.45, ow=2)),
        ("锐70+降噪0",             dict(sharpen=70,  denoise=0.0,  ow=2)),
        ("锐110",                  dict(sharpen=110, denoise=0.45, ow=2)),
        ("锐70/降噪0/半径1.4",      dict(sharpen=70,  denoise=0.0,  ow=2, radius=1.4)),
    ]
    print(f"\n== {name} ==")
    outs = {}
    for tag, kw in variants:
        r = kw.get("radius", 1.1); d_ = kw["denoise"]; s_ = kw["sharpen"]
        im = down_chain(master, 300)
        if d_ > 0: im = denoise_auto(im, d_)
        if s_ > 0: im = final_sharp(im, s_, radius=r, thresh=2)
        im = add_outline(im, width=kw["ow"])
        outs[tag] = im
        print(f"  {tag:<30}{lap_var(im):9.1f}{halo_score(im):9.2f}")
    # 拼接：基准 / 锐40 / 锐70 / 锐110 各 2x 局部放大
    board = Image.new("RGB", (300*2, 300*len(variants)), (250,250,250))
    for r,(tag,_) in zip(range(len(variants)), variants):
        z = outs[tag].resize((600,600), Image.LANCZOS)
        board.paste(z, (0, r*300))
    board.save(f"sharp_tune/_{name}.png")
    print(f"   -> sharp_tune/_{name}.png (每格 2x 放大)")

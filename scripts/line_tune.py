# -*- coding: utf-8 -*-
"""线稿降采样寻优：对比现行管线 vs line_down 不同 stroke 强度。

客观指标（针对"线条流不流畅"）：
  ridge = 笔触像素相对背景的"站得住"程度 —— 越高，线越实、越少虚虚乎乎的糊边
  consistency = 笔触像素亮度的标准差 —— 越低，线越均匀流畅（亚像素相位抖动越小）
  halo = 光晕 —— 越低越好
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import minimum_filter, maximum_filter, median_filter, uniform_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pack_sr import SRC
from down import down_chain
from line_down2 import down_line
from matting_gpu import matte
from polish import add_outline

OUTD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "line_tune")
os.makedirs(OUTD, exist_ok=True)
NAMES = ["打call.png", "充电中.png", "尴尬.png", "不想动.png"]


def lum(a):
    return a[..., 0] * .2126 + a[..., 1] * .7152 + a[..., 2] * .0722


def ridge(L):
    """笔触强度：L 相对局部背景(5x5均值)的低暗量。越高 = 线越实"""
    bg = uniform_filter(L, 5, mode="nearest")
    d = np.clip(bg - L, 0, None)
    return float(d.mean())


def consistency(L):
    """笔触像素亮度 std：越低 = 线越均匀流畅"""
    bg = uniform_filter(L, 5, mode="nearest")
    d = np.clip(bg - L, 0, None)
    mask = d > np.percentile(d, 60)
    if mask.sum() < 50:
        return 0.0
    return float(L[mask].std())


def halo(im):
    """光晕：USM 过头的直接证据，越低越好"""
    a = np.asarray(im.convert("RGB")).astype(np.float32)
    g = a @ np.array([.2126, .7152, .0722], dtype=np.float32)
    b = median_filter(g, 3)
    return float(np.abs(g - b).mean())


def report(tag, im):
    L = lum(np.asarray(im).astype(np.float32) / 255.0)
    return dict(ridge=ridge(L), consist=consistency(L), halo=halo(im))


def main():
    from pack_final import SRC as _S
    res = {}
    for nm in NAMES:
        src = matte(Image.open(os.path.join(SRC, nm)).convert("RGBA"))
        variants = {}
        t0 = time.time()
        variants["现行(多级LANCZOS)"] = down_chain(src, 300)
        from pack_sr import fit_from_big
        variants["BOX面积平均"] = fit_from_big(src, 300)
        for f in (0.45, 0.62, 0.8):
            variants[f"线稿floor={f}"] = down_line(src, 300, floor=f)
        print(f"{nm} ({time.time()-t0:.1f}s)", flush=True)
        res[nm] = {}
        for tag, im in variants.items():
            r = report(tag, im)
            res[nm][tag] = (im, r)
            print(f"   {tag:<18} ridge={r['ridge']:.4f} consist={r['consist']:.4f} halo={r['halo']:.2f}",
                  flush=True)

    # 平均
    print("\n=== 平均 ===")
    agg = {}
    for nm, d in res.items():
        for tag, (im, r) in d.items():
            a = agg.setdefault(tag, [0, 0, 0, 0])
            a[0] += r["ridge"]; a[1] += r["consist"]; a[2] += r["halo"]; a[3] += 1
    for tag, a in agg.items():
        print(f"{tag:<18} ridge={a[0]/a[3]:.4f}  consist={a[1]/a[3]:.4f}  halo={a[2]/a[3]:.2f}")

    # 拼 4x 放大对比板（先合成白底再放大，透明区不再显示成黑噪点）
    tags = list(agg)
    Z, CROP = 4, 90
    tw, th = CROP * Z, CROP * Z
    gw, gh = tw + 20, th + 20
    sheet = Image.new("RGB", (gw * len(tags) + 8, (gh + 20) * len(NAMES) + 8), (240, 240, 240))
    from PIL import ImageDraw
    dr = ImageDraw.Draw(sheet)
    for ci, tag in enumerate(tags):
        dr.text((8 + ci * gw, 3), tag, fill=(20, 20, 20))
    for ri, nm in enumerate(NAMES):
        for ci, tag in enumerate(tags):
            im = res[nm][tag][0]
            flat = Image.new("RGB", im.size, (250, 250, 250))
            flat.paste(im, mask=im.split()[3])
            cx, cy = flat.size[0] // 2, flat.size[1] // 2
            c = flat.crop((cx - CROP // 2, cy - CROP // 2, cx + CROP // 2, cy + CROP // 2))
            sheet.paste(c.resize((tw, th), Image.NEAREST), (8 + ci * gw + 20, 8 + ri * gh + 20))
        dr.text((8, 8 + ri * gh + gh), nm, fill=(60, 60, 60))
    sheet.save(os.path.join(OUTD, "线稿对比板.png"))
    print("\n板:", os.path.join(OUTD, "线稿对比板.png"), sheet.size)


if __name__ == "__main__":
    main()

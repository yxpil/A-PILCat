# -*- coding: utf-8 -*-
"""超分后的"原生感"打磨：去溢色边 + 干净 alpha + 官方描边 + 弱锐化 + 降塑料噪
与旧版 (pack_sr.sr_rgba) 对比效果。
"""
import os, io, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
from PIL import Image, ImageFilter, ImageChops
from scipy.ndimage import distance_transform_edt, median_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pack_sr import SRC, sr_rgba, superres, fit_from_big, save_png_under, content_bbox


# ---------- 1) 去溢色：把半透明/透明区的 RGB 用邻近实心色填满 ----------
def despill(im, n=1):
    """边缘外扩 n 圈并填成邻近实心色：超分时 alpha 会内缩，不留 skirt 就会渗白底"""
    a = np.array(im.split()[3]) > 0
    rgb = np.array(im.convert("RGB")).astype(np.int16)
    filled = rgb.copy()
    near = rgb
    if (~a).any():
        idx = distance_transform_edt(~a, return_indices=True)[1]   # (2,H,W) 索引栈
        near = rgb[idx[0].astype(np.int32), idx[1].astype(np.int32)]  # 最近实心像素色
        filled[~a] = near[~a]
    prev_im = Image.fromarray((a * 255).astype(np.uint8))
    cur = a
    for _ in range(n):
        cur_im = prev_im.filter(ImageFilter.MaxFilter(3))
        cur = np.array(cur_im) > 0
        filled[cur & ~a] = near[cur & ~a]           # 只填新扩出来的那一圈
        prev_im = cur_im
    return Image.fromarray(filled.astype(np.uint8), "RGB"), cur   # cur=外扩遮罩


# ---------- 2) 官方描边 ----------
def add_outline(rgba, color=(46, 46, 54), width=1):
    a = rgba.split()[3].point(lambda p: 255 if p > 110 else 0)
    d = a.filter(ImageFilter.MaxFilter(2 * max(width, 1) + 1))   # 外扩 width 像素
    ring = ImageChops.subtract(d, a)
    if ring.getbbox() is None:
        return rgba
    o = Image.new("RGBA", rgba.size, color + (255,))
    o.putalpha(ring)
    return Image.alpha_composite(o, rgba)        # 描边在下，主体在上


# ---------- 3) 降"塑料感"：极轻模糊混合，压超分噪点 ----------
def soften(rgba, amount=0.35):
    r, g, b, a = rgba.split()
    sm = Image.merge("RGB", (r, g, b)).filter(ImageFilter.GaussianBlur(0.6))
    base = Image.merge("RGB", (r, g, b))
    m = Image.eval(base, lambda p: int(255 * (1 - amount)))
    mixed = Image.blend(base, sm, amount)
    out = mixed.convert("RGBA"); out.putalpha(a)
    return out


# ---------- 3.5) 压超分噪点 + alpha 清理 ----------
def denoise_uniform(rgba, blend=0.8):
    """3x3 中值混合：Real-ESRGAN 的高频噪在最终像素空间压平，发丝等连续细节保留"""
    med = rgba.filter(ImageFilter.MedianFilter(3))
    out = Image.blend(rgba, med, blend)
    # alpha 清理：孤立/细弱半透明归零，接近实心的归实，留一圈抗锯齿
    a_im = out.split()[3]
    a = np.array(a_im, dtype=np.int16)
    a = np.where(a < 40, 0, np.where(a >= 215, 255, a))
    out = out.copy(); out.putalpha(Image.fromarray(a.astype(np.uint8)))
    return out


# ---------- 3.6) 自适应降噪：只压孤立噪点，放过连续细节 ----------
def denoise_auto(rgba, strength=1.0):
    """多尺度判噪：|img-med3| 大但 |med3-med5| 小 => 孤立噪点；
    两者都大 => 发丝/线条等真实高频细节，降权保护。"""
    rgb = np.array(rgba.convert("RGB")).astype(np.int16)
    med3 = np.array(median_filter(rgb, size=3, mode="nearest"))
    med5 = np.array(median_filter(rgb, size=5, mode="nearest"))
    d1 = np.abs(rgb - med3).max(axis=2)
    d2 = np.abs(med3 - med5).max(axis=2)
    w = np.clip((d1 - d2) * 0.45, 0, 1) * strength       # 只在噪点上生效
    w3 = w[..., None]
    mixed = rgb * (1 - w3) + med3 * w3

    a = np.array(rgba.split()[3], dtype=np.int16)
    am3 = np.array(median_filter(a, size=3, mode="nearest"))
    wa = np.clip((np.abs(a - am3) - 6) * 0.5, 0, 1) * strength
    a = np.where(a < 40, 0, np.where(a >= 215, 255, np.where(wa > 0.5, am3, a)))
    out = Image.fromarray(np.clip(mixed, 0, 255).astype(np.uint8), "RGB")
    o = out.convert("RGBA"); o.putalpha(Image.fromarray(a.astype(np.uint8)))
    return o


# ---------- 3.7) 最终像素空间锐化：补回降采样丢掉的高频 ----------
def final_sharp(rgba, amount=70, radius=1.1, thresh=2, gain=1.0):
    """降采样到目标尺寸后做 USM。amount 越大越"硬"，过大会出白边光晕。"""
    r, g, b, a = rgba.split()
    rgb = Image.merge("RGB", (r, g, b))
    sharp = rgb.filter(ImageFilter.UnsharpMask(radius, amount, thresh))
    if gain != 1.0:
        sharp = Image.blend(rgb, sharp, gain)
    out = sharp.convert("RGBA"); out.putalpha(a)
    return out


# ---------- 4) 统一入口 ----------
def sr_polish(src_rgba, size=300, outline=True, ow=2, sharpen=False, lq_max=256,
              denoise=0.8, adaptive=True, final_sharpen=0.0):
    desp, ext = despill(src_rgba, n=1)
    a_low = src_rgba.split()[3]
    big = superres(desp, lq_max)                  # x4
    na = a_low.resize(big.size, Image.LANCZOS) \
               .filter(ImageFilter.GaussianBlur(0.4)) \
               .point(lambda p: 255 if p > 150 else (0 if p < 85 else p))
    out = big.convert("RGBA"); out.putalpha(na)

    out = soften(out)
    if sharpen:
        rgb = out.convert("RGB").filter(ImageFilter.UnsharpMask(1.4, 60, 2))
        out = rgb.convert("RGBA"); out.putalpha(out.split()[3])
    out = fit_from_big(out, size)                 # 先缩到目标画布
    if denoise > 0:                               # 降噪在最终像素空间做
        out = denoise_auto(out, denoise) if adaptive else denoise_uniform(out, denoise)
    if final_sharpen > 0:                         # 降采样后补高频，300px 才不发糊
        out = final_sharp(out, amount=final_sharpen, radius=1.1, thresh=2, gain=1.0)
    if outline:
        out = add_outline(out, width=ow)          # 描边在最终像素空间加，才不会被缩掉
    return out


if __name__ == "__main__":
    os.makedirs("cmp_polish", exist_ok=True)
    picks = ["充电中.png", "有被笑到.png", "打call.png", "已读乱回.png"]
    picks = [p for p in picks if os.path.exists(os.path.join(SRC, p))]
    for i, name in enumerate(picks, 1):
        src = Image.open(os.path.join(SRC, name)).convert("RGBA")
        bb = content_bbox(src)
        src = src.crop(bb) if bb else src
        t0 = time.time()

        old = fit_from_big(sr_rgba(src), 300)
        new = fit_from_big(sr_polish(src, size=300, outline=False), 300)
        lit = sr_polish(src, size=300, outline=True, ow=2)   # 带描边

        # 局部放大（人脸/边缘 3 倍对比）
        def zoom(im, box):
            c = im.crop(box)
            w, h = c.size
            return c.resize((w * 3, h * 3), Image.LANCZOS)

        bw = max(new.width, old.width)
        top = Image.new("RGBA", (bw * 2 + 20, 300), (250, 250, 250, 255))
        top.paste(old, (0, 0)); top.paste(new, (bw + 20, 0))
        top.save(f"cmp_polish/{i}_old_vs_new_{name}.png")

        # 局部三连：无描边 / 带描边
        lb = (int(new.width * 0.22), int(new.height * 0.05),
              int(new.width * 0.78), int(new.height * 0.55))
        z1 = zoom(lit, lb); z2 = zoom(new, lb)
        zw = max(z1.width, z2.width) + 20
        board = Image.new("RGBA", (zw, z1.height * 2 + 6), (250, 250, 250, 255))
        board.paste(z1, (0, 0)); board.paste(z2, (0, z1.height + 6))
        board.save(f"cmp_polish/{i}_zoom_{name}.png")

        kb = {}
        for tag, im in [("plain", new), ("outline", lit)]:
            ob = io.BytesIO(); im.save(ob, "PNG", optimize=True); kb[tag] = ob.tell()/1024
        ob = io.BytesIO(); old.save(ob, "PNG", optimize=True)
        print(f"{name}: {time.time()-t0:.1f}s  旧 {ob.tell()/1024:.0f}KB | "
              f"打磨无描边 {kb['plain']:.0f}KB | 打磨+描边 {kb['outline']:.0f}KB")
    print("done -> cmp_polish/")

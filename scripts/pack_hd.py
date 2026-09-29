# -*- coding: utf-8 -*-
"""高清原分辨率版三件套 —— 与 `上传`(300x300 平台版) 并存的 ~1900px 版。

源: C:\\Users\\Admin\\OneDrive\\Desktop\\合格   (155 张 ~1900px 原生贴纸)
出: C:\\Users\\Admin\\OneDrive\\Desktop\\上传_高清\\第0N包\
     表情缩略图.zip   PNG  ~源图分辨率(长边<=1900, 信息零损失, 256色量化)
     表情动画.zip     GIF  ~源图分辨率 10 帧语义动效
     表情封面图.png   200x200  (沿用平台封面规格)

形变通道（抽样 A/B/C 对比，见 _hd_ab.py）:
  A  LQ 通道    抠图+形变在 475px 算 -> RealESRGAN x4 回 ~1900   （用户建议的省显存路线）
  B  半分辨率    抠图原生，形变在 950px 算 -> LANCZOS 2x 回 ~1900
  C  原生        抠图+形变都在源图分辨率算                        （信息最真，最慢）

用法:
  python pack_hd.py                # 全量 10 包
  python pack_hd.py --channel B    # 指定通道
  python pack_hd.py --only 0       # 只跑第 1 包（调试）
  python pack_hd.py --from 2       # 从第 3 包开始
"""
import os, io, re, sys, time, hashlib, zipfile, warnings, tempfile
warnings.filterwarnings("ignore")
import numpy as np
import torch
from PIL import Image, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

TMPDIR = tempfile.mkdtemp(prefix="qqhd_")
SRC = r"C:\Users\Admin\OneDrive\Desktop\合格"
OUT = r"C:\Users\Admin\OneDrive\Desktop\上传_高清"

# ---------- 规格 ----------
HD_MAX = 1900      # 统一正方形画布（与 300px 平台版同构，内容 fit 留边）
HD_MARGIN = 24     # 画布四周留白（内容最长边 ~1852，等效 300px 空间 4px 边距）
LQ_MAX = 475       # A 通道低分辨率上限
B_DIV = 2          # B 通道形变分辨率 = 原生 / B_DIV
OUTLINE = 11       # 1900px 空间里等效 300px 的 2px（视觉等比）
SHARPEN = 26
FRAMES = 10
GIF_MS = 90
GIF_LIMIT_KB = 1400
PNG_LIMIT_KB = 900
COVER = 200
CHANNEL = "C"     # A / B / C

# ---------- 复用现有部件 ----------
from matting_gpu import matte
from pack_sr import (content_bbox, save_png_under, robust_write,
                     strip_top_text, keyword_of, fix_keyword, collect_items)
from polish import despill, denoise_auto, final_sharp, add_outline
from simple_down import alpha_fix
from motion2 import frame_seq as seq2
from motion_map import motion2_of
from pack_motion import frames_to_gif


# ---------- 基础 ----------
def prep(name):
    """读源图 -> GPU 抠图 -> 裁掉全透明边。返回原生 RGBA。"""
    src = Image.open(os.path.join(SRC, name)).convert("RGBA")
    m = matte(src)
    bb = content_bbox(m)
    return m.crop(bb) if bb else m


def fit_native(rgba):
    """trim 后等比 fit 到统一 HD_MAX 正方形画布居中（与 300px 平台版同构图）。
    内容最长边 ~1852，源图原生 ~1900 -> 仅 2.5% 缩放，信息基本零损失。"""
    w, h = rgba.size
    s = min((HD_MAX - 2 * HD_MARGIN) / w, (HD_MAX - 2 * HD_MARGIN) / h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    sm = rgba.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGBA", (HD_MAX, HD_MAX), (0, 0, 0, 0))
    canvas.paste(sm, ((HD_MAX - nw) // 2, (HD_MAX - nh) // 2), sm)
    return canvas


def to_hd(base):
    """原生 -> 高清成品：alpha 补齐 + 去溢色 + 自适应降噪 + 锐化 + 描边。"""
    im = alpha_fix(base)
    rgb, _ = despill(im, n=1)
    out = rgb.convert("RGBA")
    out.putalpha(im.split()[3])
    out = denoise_auto(out, 0.2)
    out = final_sharp(out, SHARPEN, radius=6.0, thresh=2)
    return add_outline(out, width=OUTLINE)


def cover_from(hd, size=COVER, frac=0.92):
    """裁内容 bbox -> fit 到 size 画布占 frac 居中（主流程同款 margin 比例，
    保证封面不切头不贴边）。"""
    bb = content_bbox(hd)
    src = hd.crop(bb) if bb else hd
    w, h = src.size
    s = min(size * frac / w, size * frac / h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    sm = src.resize((nw, nh), Image.LANCZOS)
    cv = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    cv.paste(sm, ((size - nw) // 2, (size - nh) // 2), sm)
    return cv


def _work_res(base, channel):
    """形变工作分辨率（base 已是 HD_MAX 画布）"""
    if channel == "A":
        target = LQ_MAX
    elif channel == "B":
        target = HD_MAX // B_DIV
    else:
        return base
    w, h = base.size
    s = target / max(w, h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    return base.resize((nw, nh), Image.LANCZOS)


def _sr4(rgba):
    """LQ 帧 -> RealESRGAN x4 回高分辨率，alpha 同步放大（走 pack_sr/rrdb 通道）。"""
    from pack_sr import superres
    a = rgba.split()[3]
    flat = Image.new("RGB", rgba.size, (255, 255, 255))
    flat.paste(rgba, mask=a)
    big = superres(flat, LQ_MAX)
    na = a.resize(big.size, Image.LANCZOS) \
           .filter(ImageFilter.GaussianBlur(0.5)) \
           .point(lambda p: 255 if p > 170 else (0 if p < 90 else p))
    big = big.convert("RGBA")
    big.putalpha(na)
    return big


def frames_hd(base, kw, channel=CHANNEL, n=FRAMES):
    """高分辨率动作帧。通道决定在哪一级分辨率做形变、怎么回原生。"""
    kind, mouth = motion2_of(kw)
    work = _work_res(base, channel)
    seq = seq2(work, kind, n=n, amp=1.0, open_mode=mouth, mouth_amp=1.0)
    if channel == "A":
        return [_sr4(f) for f in seq]                      # LQ 形变 -> x4 超分
    if channel == "B":
        return [f.resize((f.width * B_DIV, f.height * B_DIV),
                         Image.LANCZOS) for f in seq]      # 半分辨形变 -> 2x
    return seq                                             # 原生形变


# ---------- 单张 ----------
_TMPN = [0]


def one(name, kw, channel=CHANNEL):
    _TMPN[0] += 1
    k = _TMPN[0]
    base = fit_native(prep(name))
    hd = to_hd(base)

    tp = os.path.join(TMPDIR, f"h{k}.png")
    n, u = save_png_under(hd, tp, PNG_LIMIT_KB)
    pngs.append((tp, f"{kw}_{k:02d}.png", n))

    gp = os.path.join(TMPDIR, f"h{k}.gif")
    fr = frames_hd(base, kw, channel)
    ng, nf = frames_to_gif(fr, gp, duration=GIF_MS,
                           limit_kb=GIF_LIMIT_KB, colors=64)
    gifs.append((gp, f"{kw}_{k:02d}.gif", ng))

    kind, mouth = motion2_of(kw)
    print(f"   {kw}_{k:02d} {hd.width}x{hd.height} png{n/1024:.0f}/gif{ng/1024:.0f}"
          f"({kind}) ", end="", flush=True)
    return hd, base


# ---------- 分组（与 上传 目录 1:1 对齐） ----------
def build_packs():
    items = collect_items()
    packs = []
    for name in items:
        kw = fix_keyword(keyword_of(name))
        for p in packs:
            if len(p) < 16 and kw not in [x[1] for x in p]:
                p.append((name, kw))
                break
        else:
            packs.append([(name, kw)])
    return packs


def cover_fallback(kw):
    """封面兜底：源图打call 去顶部文字带 -> 裁上部35% -> 高清->200。"""
    base = Image.open(os.path.join(SRC, "打call.png")).convert("RGBA")
    cand = strip_top_text(base)
    if cand is not None:
        base = cand
        tag = "打call 去文字带"
    else:
        bb = content_bbox(base)
        crop = base.crop((bb[0], bb[1] + int((bb[3] - bb[1]) * 0.35), bb[2], bb[3]))
        base = crop if content_bbox(crop) else base
        tag = "打call 裁上部35%"
    cb = content_bbox(base)
    hd = to_hd(base.crop(cb) if cb else base)
    print(f"   封面: {tag}", end="", flush=True)
    return cover_from(hd)


def main():
    global CHANNEL, pngs, gifs
    args = sys.argv[1:]
    if "--channel" in args:
        CHANNEL = args[args.index("--channel") + 1]
    skip = int(args[args.index("--from") + 1]) if "--from" in args else 0
    only = int(args[args.index("--only") + 1]) if "--only" in args else None

    packs = build_packs()
    print(f"通道={CHANNEL}  源图={HD_MAX}上限  "
          f"缩略图/动画 长边~{HD_MAX}  封面={COVER}  帧={FRAMES}")
    print(f"共 {sum(len(p) for p in packs)} 张 -> {len(packs)} 组")

    for pi, pack in enumerate(packs):
        if pi < skip or (only is not None and pi != only):
            continue
        tag = f"第{pi+1:02d}包" if len(pack) == 16 else f"剩余{len(pack)}张"
        pdir = os.path.join(OUT, tag)
        os.makedirs(pdir, exist_ok=True)
        pngs, gifs = [], []
        _TMPN[0] = 0
        print(f"\n== {tag}: {len(pack)} 个 ==", flush=True)
        t0 = time.time()
        cov = None
        for i, (name, kw) in enumerate(pack, 1):
            hd, base = one(name, kw, CHANNEL)
            # 封面候选：第一张能裁掉顶部文字带的图（与主流程同规矩：封面无文字不切头）
            if cov is None and name != "打call.png":
                cand = strip_top_text(base)
                if cand is not None:
                    cb = content_bbox(cand)
                    if cb:
                        cov = cover_from(to_hd(cand.crop(cb)))
                        print(f"[封面候选={kw}]", end="", flush=True)
        if cov is None:
            cov = cover_fallback(kw)
        cp = os.path.join(pdir, "表情封面图.png")
        n, u = save_png_under(cov, cp, 100 * 1024)
        print(f"\n   封面 {n/1024:.1f}KB({u})", flush=True)

        buf1, buf2 = io.BytesIO(), io.BytesIO()
        with zipfile.ZipFile(buf1, "w", zipfile.ZIP_DEFLATED) as z:
            for tmp, fn, _ in pngs:
                z.writestr(fn, open(tmp, "rb").read())
        with zipfile.ZipFile(buf2, "w", zipfile.ZIP_DEFLATED) as z:
            for tmp, fn, _ in gifs:
                z.writestr(fn, open(tmp, "rb").read())
        robust_write(os.path.join(pdir, "表情缩略图.zip"), buf1.getvalue())
        robust_write(os.path.join(pdir, "表情动画.zip"), buf2.getvalue())
        print(f"   缩略图.zip {len(pngs)}个 {len(buf1.getvalue())//1024}KB | "
              f"动画.zip {len(gifs)}个 {len(buf2.getvalue())//1024}KB | "
              f"耗时 {time.time()-t0:.0f}s", flush=True)

    print("\nALL DONE")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""把「每贴纸 3 张重绘图」组成一个 GIF，并按平台三件套打包。

每张贴纸：
  * 3 张重绘（seed 11/22/33，同一画布像素对齐）-> 逐个抠图 -> 400 统一降采样 -> 3 帧 GIF
  * 缩略图 = 第 1 张重绘的 300x300 PNG
  * 每包 1 张 200x200 封面（无文字）

输出 <OUT>/第XX包/：
  表情动画.zip(16 个 3 帧 GIF, 300x300, <=250KB)
  表情缩略图.zip(16 张 300x300 PNG, <200KB)
  表情封面图.png(200x200, <100KB)

用法:
  python pack3.py                # 全量
  python pack3.py --only 0       # 只跑第 1 包
"""
import os, sys, io, time, zipfile, shutil, warnings
warnings.filterwarnings("ignore")
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pack_sr import (SRC, content_bbox, collect_items, keyword_of, fix_keyword,
                     save_png_under, robust_write, strip_top_text, TMPDIR)
from polish import final_sharp, add_outline, denoise_auto
from simple_down import box_down, alpha_fix
from matting_gpu import matte

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "redraw3")
OUT = r"C:\Users\Admin\OneDrive\Desktop\上传_重绘"
CAND = os.path.join(OUT, "_重绘候选")      # 原始候选图归档处
TAG = "_w1024_st0.45_sh30"
SEEDS = [11, 22, 33]
SIZE = 300
COVER = 200
FRAME_MS = 300
GIF_LIMIT_KB = 250
PNG_LIMIT_KB = 200
COVER_LIMIT_KB = 100
OUTLINE = 3                    # 平台规范：3 像素白色描边（防边缘锯齿）
OUTLINE_COLOR = (255, 255, 255)
SHARP = 32

# 平台命名红线：关键词 ≤4 字符、无数字标点。超长的映射为合规近义词（仅影响文件名）
KW_ALIAS = {"打call": "应援", "DNA动了": "心动了"}


def kw_name(kw):
    return KW_ALIAS.get(kw, kw)


def paths_of(kw, name):
    """取该贴纸 3 个 seed 的重绘文件（关键字重名的用带后缀 stem）"""
    stem = os.path.splitext(name)[0]
    out = []
    for sd in SEEDS:
        for key in (stem, kw):
            p = os.path.join(RAW, f"{key}_s{sd}{TAG}.png")
            if os.path.exists(p):
                out.append(p)
                break
        else:
            out.append(None)
    return out


def finish(rgba, size=SIZE):
    """降到 size 画布：BOX 面积平均 + 补 alpha + 轻降噪 + 锐化 + 描边。
    （不做 trim——三帧必须共用同一画布坐标系，否则会抖）"""
    im = alpha_fix(box_down(rgba, size))
    im = denoise_auto(im, 0.2)
    im = final_sharp(im, SHARP, radius=1.1, thresh=2)
    return add_outline(im, color=OUTLINE_COLOR, width=OUTLINE)


def gif3(frames, path, duration=FRAME_MS, limit_kb=GIF_LIMIT_KB):
    """3 帧透明 GIF，全帧统一调色板；超限只降色不丢帧。"""
    w, h = frames[0].size

    def enc(ncol):
        flats = []
        for f in frames:
            bg = Image.new("RGB", (w, h), (255, 255, 255))   # 白描边贴白底，防灰边
            bg.paste(f, mask=f.split()[3])
            flats.append(bg)
        strip = Image.new("RGB", (w, h * len(flats)))
        for i, fl in enumerate(flats):
            strip.paste(fl, (0, i * h))
        q = strip.quantize(colors=ncol, method=Image.FASTOCTREE,
                           dither=Image.Dither.NONE)
        pal = (q.getpalette() + [0] * 768)[:768]
        pal[765:768] = [0, 0, 0]
        ps = []
        for i in range(len(flats)):
            idx = np.asarray(q.crop((0, i * h, w, (i + 1) * h)),
                             dtype=np.uint8).copy()
            idx[np.asarray(frames[i].split()[3]) <= 128] = ncol
            p = Image.fromarray(idx, mode="P")
            p.putpalette(pal)
            ps.append(p)
        buf = io.BytesIO()
        ps[0].save(buf, "GIF", save_all=True, append_images=ps[1:],
                   duration=duration, loop=0, transparency=ncol,
                   disposal=2, optimize=True)
        return buf.getvalue()

    data = None
    for ncol in (255, 224, 192, 160, 128, 112, 96, 80, 64):
        data = enc(ncol)
        if len(data) <= limit_kb * 1024:
            break
    robust_write(path, data)
    return len(data), len(frames)


def _fit_cover(im, size=COVER):
    """内容 bbox -> fit size 留边 -> 锐化 + 描边 -> 居中画布"""
    bb = content_bbox(im)
    if bb:
        im = im.crop(bb)
    w, h = im.size
    s = (size - 8) / max(w, h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    small = alpha_fix(im.resize((nw, nh), Image.BOX))
    small = final_sharp(small, SHARP, radius=1.1, thresh=2)
    small = add_outline(small, width=2)   # 封面维持已验收的深色细描边
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    canvas.paste(small, ((size - nw) // 2, (size - nh) // 2), small)
    return canvas


def cover_all():
    """全项目共用一张封面：源图 打call 去顶部文字带（平台包已验收的同款来源）。
    裁不掉时再逐包成员的重绘图试 strip_top_text。"""
    from pack_sr import DCALL_FALLBACK
    base = Image.open(DCALL_FALLBACK).convert("RGBA")
    cand = strip_top_text(base)
    if cand is not None:
        return _fit_cover(cand), "打call裁顶部文字带"
    bb = content_bbox(base)
    crop = base.crop((bb[0], bb[1] + int((bb[3] - bb[1]) * 0.35), bb[2], bb[3]))
    if content_bbox(crop):
        return _fit_cover(crop), "打call裁上部35%兜底"
    return _fit_cover(base), "打call原图兜底"


def main():
    args = sys.argv[1:]
    only = int(args[args.index("--only") + 1]) if "--only" in args else None
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

    os.makedirs(OUT, exist_ok=True)
    missing = []
    cover, cover_how = cover_all()
    print(f"封面来源: {cover_how}", flush=True)
    print(f"共 {len(items)} 张 -> {len(packs)} 组；每组 16 个 -> 16 个 3 帧 GIF", flush=True)

    for pi, pack in enumerate(packs):
        if only is not None and pi != only:
            continue
        tag = f"第{pi+1:02d}包" if len(pack) == 16 else f"剩余{len(pack)}张"
        pdir = os.path.join(OUT, tag)
        cdir = os.path.join(CAND, tag)
        os.makedirs(pdir, exist_ok=True)
        os.makedirs(cdir, exist_ok=True)
        # 把上一版散落的候选图挪进归档目录
        for f in os.listdir(pdir):
            if f.endswith(".png") and "_s" in f:
                try:
                    shutil.move(os.path.join(pdir, f), os.path.join(cdir, f))
                except Exception:
                    pass

        print(f"\n== {tag}: {len(pack)} 个 ==", flush=True)
        t0 = time.time()
        pngs, gifs = [], []
        for i, (name, kw) in enumerate(pack, 1):
            ps = paths_of(kw, name)
            if any(x is None for x in ps):
                missing.append(f"{kw}({name})")
                continue
            frames = []
            for p in ps:
                shutil.copy2(p, os.path.join(cdir, f"{kw}_{i:02d}_" + os.path.basename(p)))
                rgba = matte(Image.open(p).convert("RGB"))
                frames.append(finish(rgba))
            # 缩略图 = 第 1 帧
            tp = os.path.join(TMPDIR, f"g{pi}_{i}.png")
            n, u = save_png_under(frames[0], tp, PNG_LIMIT_KB * 1024)
            pngs.append((tp, f"{kw_name(kw)}_{i:02d}.png", n))
            gp = os.path.join(TMPDIR, f"g{pi}_{i}.gif")
            ng, nf = gif3(frames, gp)
            gifs.append((gp, f"{kw_name(kw)}_{i:02d}.gif", ng))
            print(f"   {kw_name(kw)}_{i:02d} png{n/1024:.0f}/gif{ng/1024:.0f}({nf}帧) ", end="", flush=True)
        print(f"\n   耗时 {time.time()-t0:.0f}s", flush=True)

        # 封面：全项目共用（源图打call 去文字，无文字硬规矩）
        cp = os.path.join(pdir, "表情封面图.png")
        n, u = save_png_under(cover, cp, COVER_LIMIT_KB * 1024)
        print(f"   表情封面图.png {n/1024:.1f}KB({u})", flush=True)

        b1, b2 = io.BytesIO(), io.BytesIO()
        with zipfile.ZipFile(b1, "w", zipfile.ZIP_DEFLATED) as z:
            for tmp, fn, _ in pngs:
                z.writestr(fn, open(tmp, "rb").read())
        with zipfile.ZipFile(b2, "w", zipfile.ZIP_DEFLATED) as z:
            for tmp, fn, _ in gifs:
                z.writestr(fn, open(tmp, "rb").read())
        robust_write(os.path.join(pdir, "表情缩略图.zip"), b1.getvalue())
        robust_write(os.path.join(pdir, "表情动画.zip"), b2.getvalue())
        print(f"   缩略图.zip {len(pngs)}个 {len(b1.getvalue())//1024}KB | "
              f"动画.zip {len(gifs)}个 {len(b2.getvalue())//1024}KB", flush=True)

    print(f"\nALL DONE -> {OUT}")
    if missing:
        print(f"缺失 {len(missing)}: {missing}")


if __name__ == "__main__":
    main()

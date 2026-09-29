# -*- coding: utf-8 -*-
"""补齐 上传/Example 目录：用 BOX 定稿管线重出 16 张（缩略图 zip + 动画 zip），
并补上缺失的 表情封面图.png。与 pack_final.py 完全同源同参数。"""
import os, io, re, sys, zipfile, warnings, time
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pack_sr import SRC, OUT, DCALL, DCALL_FALLBACK, TMPDIR, \
    content_bbox, sr_rgba, save_png_under, robust_write, strip_top_text, \
    keyword_of, fix_keyword, collect_items
from polish import sr_polish
from simple_down import box_down, alpha_fix
from pack_final import to_final, GIF_SIZE, MASTER, LQ_MAX, FINAL_SHARPEN, OUTLINE
from matting_gpu import matte
from pack_motion import frames_to_gif
from make_gif_v2 import make_frames_v2

EX = os.path.join(OUT, "Example")
PNG_LIMIT = 200 * 1024
GIF_LIMIT_KB = 250


def kw_to_file():
    """kw -> 源图文件名（md5 去重后取第一个）。"""
    seen, files = {}, {}
    for name in sorted(os.listdir(SRC)):
        if not name.lower().endswith(".png"):
            continue
        h = __import__("hashlib").md5(open(os.path.join(SRC, name), "rb").read()).hexdigest()
        if h in seen:
            continue
        seen[h] = name
        k = fix_keyword(keyword_of(name))
        files.setdefault(k, name)
    return files


def build(kw, fname, pi, i, pngs, gifs):
    src = matte(Image.open(os.path.join(SRC, fname)).convert("RGBA"))
    master = sr_polish(src, size=MASTER, ow=0, lq_max=LQ_MAX, denoise=0.0)
    g300 = to_final(master, GIF_SIZE, FINAL_SHARPEN, denoise=0.2, ow=OUTLINE)

    tp = os.path.join(TMPDIR, f"ex_{pi}_{i}.png")
    n, u = save_png_under(g300, tp, PNG_LIMIT)
    pngs.append((tp, f"{kw}_{i:02d}.png", n))

    gp = os.path.join(TMPDIR, f"ex_{pi}_{i}.gif")
    fr, kind, mouth = make_frames_v2(g300, kw)
    ng, nf = frames_to_gif(fr, gp, limit_kb=GIF_LIMIT_KB)
    gifs.append((gp, f"{kw}_{i:02d}.gif", ng))
    print(f"   {kw}_{i:02d} png{n/1024:.0f}/gif{ng/1024:.0f}({kind}{'+'+mouth if mouth else ''},{nf}帧)",
          flush=True)
    return g300


def main():
    files = kw_to_file()
    kws = open("_ex_kws.txt", encoding="utf-8").read().split("\n")
    kws = [k.strip() for k in kws if k.strip()]
    print(f"Example: {len(kws)} 张  [BOX 定稿管线]", flush=True)

    pngs, gifs = [], []
    cover_src = None
    t0 = time.time()
    for i, kw in enumerate(kws, 1):
        f = files.get(kw)
        if f is None:
            print(f"   {kw}: 源图缺失，跳过", flush=True)
            continue
        # 封面候选：优先能裁掉顶部文字带的贴纸（无文字、不切头）
        raw = Image.open(os.path.join(SRC, f)).convert("RGBA")
        if cover_src is None:
            cand = strip_top_text(raw)
            if cand is not None:
                cb = content_bbox(cand)
                cover_src = to_final(sr_polish(matte(cand).copy(), size=MASTER, ow=0,
                                               lq_max=LQ_MAX, denoise=0.0), 200, ow=OUTLINE) \
                    if cb else None
        build(kw, f, 0, i, pngs, gifs)
    print(f"   16 张耗时 {time.time()-t0:.0f}s", flush=True)

    # 封面兜底：打call 无文字版 > 裁掉文字带 > 原图裁上部
    if cover_src is None:
        base = Image.open(DCALL).convert("RGBA") if os.path.exists(DCALL) \
            else Image.open(DCALL_FALLBACK).convert("RGBA")
        cand = strip_top_text(base)
        if cand is None:
            bb = content_bbox(base)
            crop = base.crop((bb[0], bb[1] + int((bb[3] - bb[1]) * 0.35), bb[2], bb[3]))
            cand = crop if content_bbox(crop) else None
        if cand is not None:
            base = cand
        master = sr_polish(matte(base), size=MASTER, ow=0, lq_max=LQ_MAX, denoise=0.0)
        cover_src = to_final(master, 200, ow=OUTLINE)

    cp = os.path.join(EX, "表情封面图.png")
    n, u = save_png_under(cover_src, cp, 100 * 1024)
    print(f"   表情封面图.png {n/1024:.1f}KB({u})", flush=True)

    buf1, buf2 = io.BytesIO(), io.BytesIO()
    with zipfile.ZipFile(buf1, "w", zipfile.ZIP_DEFLATED) as z:
        for tmp, fin, _ in pngs:
            z.writestr(fin, open(tmp, "rb").read())
    with zipfile.ZipFile(buf2, "w", zipfile.ZIP_DEFLATED) as z:
        for tmp, fin, _ in gifs:
            z.writestr(fin, open(tmp, "rb").read())
    robust_write(os.path.join(EX, "表情缩略图.zip"), buf1.getvalue())
    robust_write(os.path.join(EX, "表情动画.zip"), buf2.getvalue())
    print(f"   缩略图.zip {len(pngs)}个 {len(buf1.getvalue())//1024}KB | "
          f"动画.zip {len(gifs)}个 {len(buf2.getvalue())//1024}KB", flush=True)
    print("EXAMPLE DONE")


if __name__ == "__main__":
    from PIL import Image
    main()

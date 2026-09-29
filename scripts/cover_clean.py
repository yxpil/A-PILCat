# -*- coding: utf-8 -*-
"""封面专用：去背 + 连通块级删字（只删人像头顶上方的独立文字块，人像/装饰完整保留）"""
import os, sys
import numpy as np
from PIL import Image
from scipy.ndimage import label

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pack_sr import SRC, content_bbox
from bgtest import unwhite

OUTD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bgtest")
os.makedirs(OUTD, exist_ok=True)


def remove_top_text(im, keep_thr=0.03):
    """删掉位于人像主体 bbox 上方的独立连通块（标题字）。
    返回 (处理图, 删除的块数, 删除块总面积占比)"""
    a = np.array(im)
    solid = a[..., 3] > 30
    lab, n = label(solid)
    if n <= 1:
        return im, 0, 0.0
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    main = int(np.argmax(sizes))
    ys, xs = np.where(lab == main)
    mt, mh = ys.min(), ys.max()
    tol = keep_thr * (mh - mt)
    out = a.copy()
    removed_px = 0
    n_rm = 0
    for c in range(1, n + 1):
        if c == main or sizes[c] < 40:
            continue
        cy, cx = np.where(lab == c)
        if cy.max() <= mt + tol:                    # 整块在主体头顶以上
            out[cy, cx, 3] = 0
            removed_px += len(cy)
            n_rm += 1
    return Image.fromarray(out, "RGBA"), n_rm, removed_px / (a.shape[0] * a.shape[1])


def clean_cover_src(src):
    """完整封面源处理：去背 -> 删顶部标题字 -> 紧 bbox"""
    cut = unwhite(src)
    cut, n_rm, frac = remove_top_text(cut)
    bb = content_bbox(cut)
    if not bb:
        return None, 0
    return cut.crop(bb), n_rm


def main():
    for name in ["充电中.png", "笑死.png", "比心.png", "闪闪.png"]:
        p = os.path.join(SRC, name)
        if not os.path.exists(p):
            cand = [f for f in os.listdir(SRC) if f.startswith(name[:2])]
            if not cand:
                print("缺", name); continue
            name = cand[0]
            p = os.path.join(SRC, name)
        src = Image.open(p).convert("RGBA")
        out, n_rm = clean_cover_src(src)
        if out is None:
            print(name, "空"); continue
        out.save(os.path.join(OUTD, f"clean_{name}"))
        print(f"{name}: 删顶部块 {n_rm} 个, 最终 {out.size}")


if __name__ == "__main__":
    main()

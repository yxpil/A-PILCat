# -*- coding: utf-8 -*-
"""诊断：每张源图顶部是否有一条可安全裁掉的独立文字带
输出：bbox占比 / 顶部独立带高度占比 / 裁掉后剩余内容占比 / 切头风险
"""
import os, sys, hashlib
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pack_sr import SRC, content_bbox, keyword_of, fix_keyword, collect_items


def profile(im):
    al = np.array(im.convert("RGBA").split()[3]) > 60
    H, W = al.shape
    rows = al.sum(axis=1)
    thr = max(3, int(0.0025 * W))
    y = 0
    while y < H and rows[y] < thr:
        y += 1
    if y >= H:
        return None
    # 从第一个内容行往下找最深的 gap
    band_top, band_bot, gap, best, yy = y, y, 0, None, y
    while yy < H:
        if rows[yy] < thr:
            gap += 1
            if gap >= max(6, int(0.010 * H)):
                best = (band_top, yy - gap)
                break
        else:
            gap = 0
            band_bot = yy
        yy += 1
    if best is None:
        best = (band_top, band_bot)
    return best, H, W, rows


def analyse(path):
    im = Image.open(path).convert("RGBA")
    r = profile(im)
    if r is None:
        return dict(tag="空", pct=0)
    (bt, bb), H, W, rows = r
    band_h = (bb - bt) / H
    below = rows[bb + 1:].sum() / max(1, rows.sum())
    bb_full = content_bbox(im)
    if bb_full:
        pct = ((bb_full[2] - bb_full[0]) * (bb_full[3] - bb_full[1])) / (W * H)
    else:
        pct = 0
    return dict(band_h=band_h, below=below, pct=pct, top=int(bb_full[1]) if bb_full else -1)


def main():
    items = collect_items()
    packs = []
    for name in items:
        kw = fix_keyword(keyword_of(name))
        for p in packs:
            if len(p) < 16 and kw not in [x[1] for x in p]:
                p.append((name, kw)); break
        else:
            packs.append([(name, kw)])
    targets = {1: "第02包", 4: "第05包", 5: "第06包", 6: "第07包",
               8: "第09包", 9: "剩余10张"}
    for pi, tag in sorted(targets.items()):
        print(f"\n===== {tag} =====")
        for name, kw in packs[pi]:
            r = analyse(os.path.join(SRC, name))
            if "band_h" not in r:
                print(f"  {kw:<8} 空图")
                continue
            risk = ""
            if r["band_h"] > 0.20:
                risk = "  <== 文字带过厚/与人像连片，切头风险高"
            print(f"  {kw:<8} 内容占画布{r['pct']*100:5.1f}%  "
                  f"顶部带高{r['band_h']*100:5.1f}%  下方余量{r['below']*100:5.1f}%"
                  f"{risk}")


if __name__ == "__main__":
    main()

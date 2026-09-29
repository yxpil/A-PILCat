# -*- coding: utf-8 -*-
"""封面重做：去背 + 连通块删字 + 紧bbox。
每包生成 4x4 候选拼版（处理后的成员图）供目检，
并默认用第一个"成功删字或无字"的成员出 200x200 封面预览。"""
import os, sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pack_sr import SRC, OUT, content_bbox, keyword_of, fix_keyword, collect_items
from cover_clean import clean_cover_src, remove_top_text
from bgtest import unwhite

OUTD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bgtest")
os.makedirs(OUTD, exist_ok=True)

CELL = 110


def preview_pack(pack, tag):
    """返回拼版图 + 每成员的处理元数据"""
    cols = 4
    rows = (len(pack) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * CELL, rows * CELL), (240, 240, 242))
    meta = []
    for i, (name, kw) in enumerate(pack):
        src = Image.open(os.path.join(SRC, name)).convert("RGBA")
        cut = unwhite(src)
        cut2, n_rm, frac = remove_top_text(cut)
        bb = content_bbox(cut2)
        tight = cut2.crop(bb) if bb else cut2
        t = tight.copy()
        t.thumbnail((CELL - 6, CELL - 6), Image.LANCZOS)
        x = (i % cols) * CELL + (CELL - t.width) // 2
        y = (i // cols) * CELL + (CELL - t.height) // 2
        sheet.paste(t, (x, y), t)
        meta.append(dict(name=name, kw=kw, n_rm=n_rm, frac=frac, tight=tight))
    sheet.save(os.path.join(OUTD, f"cand_{tag}.png"))
    return meta


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
    tags = [f"第{i+1:02d}包" for i in range(len(packs))]
    if len(packs[-1]) < 16:
        tags[-1] = f"剩余{len(packs[-1])}张"
    for pi, pack in enumerate(packs):
        tag = tags[pi]
        meta = preview_pack(pack, tag)
        info = "  ".join(f"{m['kw']}:{m['n_rm']}" for m in meta)
        print(f"{tag}: {len(pack)}个  删块数[{info}]")


if __name__ == "__main__":
    main()

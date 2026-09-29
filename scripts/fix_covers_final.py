# -*- coding: utf-8 -*-
"""最终封面生成：目检选定的无文字成员 -> 去背+删字+紧bbox -> 超分打磨 -> 200x200"""
import os, sys
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pack_sr import SRC, OUT, save_png_under, fit_from_big, collect_items, \
    keyword_of, fix_keyword
from cover_clean import clean_cover_src
from pack_final import fin

# 每包 -> 选定成员关键词（collect_items 里唯一）
PICKS = {
    0: "佛系", 1: "发呆中", 2: "在吗", 3: "好家伙", 4: "想开了",
    5: "打滚", 6: "比心", 7: "然后呢", 8: "躺平", 9: "隐身",
}


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
    for pi, kw in PICKS.items():
        pack = packs[pi]
        tag = f"第{pi+1:02d}包" if len(pack) == 16 else f"剩余{len(pack)}张"
        match = [n for n, k in pack if k == kw]
        if not match:
            print(f"{tag}: 找不到 {kw}"); continue
        name = match[0]
        src = Image.open(os.path.join(SRC, name)).convert("RGBA")
        tight, n_rm = clean_cover_src(src)
        if tight is None:
            print(f"{tag}: {name} 处理为空"); continue
        cover = fin(tight, 200)
        cp = os.path.join(OUT, tag, "表情封面图.png")
        n, u = save_png_under(fit_from_big(cover, 200), cp, 100 * 1024)
        print(f"{tag}: 封面 <- {name} (删块{n_rm})  {n/1024:.1f}KB({u})")


if __name__ == "__main__":
    main()

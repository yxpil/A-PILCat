# -*- coding: utf-8 -*-
"""给兜底封面包各换一张各自的封面（各包成员不同 -> 封面互不相同）：
   优先 strip_top_text 成功者；否则裁掉上部 35% 文字区后兜底。
   与 pack_final 同一条打磨管线，保证风格一致。"""
import os, hashlib, sys
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pack_sr import SRC, OUT, content_bbox, keyword_of, fix_keyword, collect_items
from pack_final import fin


def main():
    items = collect_items()
    if "打call.png" in items:
        items.remove("打call.png")
    items.insert(0, "打call.png")
    packs = []
    for name in items:
        kw = fix_keyword(keyword_of(name))
        for p in packs:
            if len(p) < 16 and kw not in [x[1] for x in p]:
                p.append((name, kw))
                break
        else:
            packs.append([(name, kw)])

    targets = {1: "第02包", 4: "第05包", 5: "第06包", 6: "第07包",
               8: "第09包", 9: "剩余10张"}
    for pi, tag in sorted(targets.items()):
        pack = packs[pi]
        cover_src = None
        chosen = None
        for name, _kw in pack:
            src = Image.open(os.path.join(SRC, name)).convert("RGBA")
            cand = None
            try:
                from pack_sr import strip_top_text
                cand = strip_top_text(src)
            except Exception:
                pass
            if cand is not None:
                cb = content_bbox(cand)
                if cb:
                    cover_src = fin(cand.crop(cb), 200)
                    chosen = f"{name} (裁文字带)"
                    break
        if cover_src is None:
            name = pack[0][0]
            base = Image.open(os.path.join(SRC, name)).convert("RGBA")
            bb = content_bbox(base)
            crop = base.crop((bb[0], bb[1] + int((bb[3]-bb[1])*0.35), bb[2], bb[3]))
            base = crop if content_bbox(crop) else base
            cb = content_bbox(base)
            cover_src = fin(base.crop(cb), 200) if cb else fin(base, 200)
            chosen = f"{name} (裁上部35%兜底)"

        from pack_sr import save_png_under
        cp = os.path.join(OUT, tag, "表情封面图.png")
        n, u = save_png_under(cover_src, cp, 100 * 1024)
        print(f"{tag}: 封面 <- {chosen}  {n/1024:.1f}KB({u})")


if __name__ == "__main__":
    main()

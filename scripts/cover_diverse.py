# -*- coding: utf-8 -*-
"""封面多样性重选：全部成员 -> 去背+删字 -> dhash -> 最远点采样贪心
保证 10 个包的封面构图两两差异最大化。
默认只出拼版目检；--write 时写入上传目录。"""
import os, sys, time
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pack_sr import SRC, OUT, save_png_under, fit_from_big, collect_items, \
    keyword_of, fix_keyword
from cover_clean import clean_cover_src

WRITE = "--write" in sys.argv
TMPD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bgtest")
os.makedirs(TMPD, exist_ok=True)

# 目检发现文字与人像连通删不掉的成员 -> 排除（包索引 -> 关键词列表）
EXCLUDE = {4: ["干杯"], 5: ["拒绝"], 2: ["哈哈哈"], 5: ["摸鱼"]}

# 人工目检 10 张 cand_拼版后记录的"无文字"成员索引（0-based，行优先 4 列）
ALLOW_IDX = {
    0: [2, 4, 5, 8, 11, 12],              # 01包: 抬头/趴/沙漏/摸下巴/佛系/捂嘴
    1: [1, 4, 5, 6, 9, 11, 13, 14, 15],   # 02包: 潜水/趴桌加班/看月亮等
    2: [0, 2, 3, 4, 5, 7, 8, 10, 11, 12],  # 03包: 加油火焰/吃西瓜/抱头蹲等
    3: [2, 3, 5, 6, 7, 8, 10, 11, 12, 14, 15],
    4: [0, 5, 6, 9, 12, 13, 14, 15],      # 05包: 低头抱/捂脸云/想开了等
    5: [2, 5, 9, 10, 12, 13, 14],         # 06包: 捧脸/抱猫/飘等
    6: [3, 5, 9, 10, 11, 12, 15],         # 07包: 比心/蹲坐/溜了/然后呢等
    7: [0, 1, 2, 3, 5, 10, 11, 12, 14],   # 08包: 看报纸/叹气等
    8: [0, 6, 10, 11, 12, 13, 14],        # 09包: 趴桌/抱礼物/躺平等
    9: [0, 4, 5, 6],                      # 剩余: 捂脸趴/闪人/蹲坐/隐身
}


def dhash(im, s=8):
    g = im.convert("L").resize((s + 1, s), Image.LANCZOS)
    a = np.asarray(g, int)
    return (a[:, 1:] > a[:, :-1]).flatten()


def main():
    t0 = time.time()
    items = collect_items()
    packs = []
    for name in items:
        kw = fix_keyword(keyword_of(name))
        for p in packs:
            if len(p) < 16 and kw not in [x[1] for x in p]:
                p.append((name, kw)); break
        else:
            packs.append([(name, kw)])

    # 阶段1：全部成员预处理 + dhash
    cands = []          # (pack_idx, name, kw, hash, clean_img)
    for pi, pack in enumerate(packs):
        allow = ALLOW_IDX.get(pi)
        for idx, (name, kw) in enumerate(pack):
            if allow is not None and idx not in allow:
                continue
            if kw in EXCLUDE.get(pi, []):
                continue
            p = os.path.join(SRC, name)
            src = Image.open(p).convert("RGBA")
            tight, n_rm = clean_cover_src(src)
            if tight is None or tight.width < 80 or tight.height < 80:
                continue
            # dhash 用统一 200x200 视角，公平比较构图
            h = dhash(fit_from_big(tight, 200))
            cands.append((pi, name, kw, h, tight))
    print(f"候选 {len(cands)} 个  预处理 {time.time()-t0:.0f}s")

    # 阶段2：最远点采样（包间互斥）
    picked = {}         # pack_idx -> cand
    sel_hashes = []
    pack_cands = {}
    for c in cands:
        pack_cands.setdefault(c[0], []).append(c)
    remaining = set(pack_cands)
    # 第一张：选"与其他所有候选最小距离"最大的那张所在的包
    def min_dist_to(h, hs):
        return min((h != x).sum() for x in hs) if hs else 999
    while remaining:
        best = None
        for pi in remaining:
            for c in pack_cands[pi]:
                md = min_dist_to(c[3], sel_hashes)
                key = (md, -abs(np.asarray(c[4].convert("L")).mean() - 128))
                if best is None or key > best[0]:
                    best = (key, pi, c)
        _, pi, c = best
        picked[pi] = c
        sel_hashes.append(c[3])
        remaining.discard(pi)
        print(f"  第{pi+1:02d}包 <- {c[2]}  min-dist={min_dist_to(c[3], sel_hashes[:-1]) if len(sel_hashes)>1 else '-'}")

    # 阶段3：拼版
    TH = 200
    sheet = Image.new("RGB", (5 * TH + 60, 2 * TH + 90), (244, 244, 246))
    dr = ImageDraw.Draw(sheet)
    for i, pi in enumerate(sorted(picked)):
        _, name, kw, h, tight = picked[pi]
        cov = fit_from_big(tight, TH)
        x, y = (i % 5) * (TH + 10) + 10, (i // 5) * (TH + 40) + 10
        sheet.paste(cov, (x, y), cov)
        tag = f"P{pi+1:02d}" if len(packs[pi]) == 16 else f"P尾"
        dr.text((x + 2, y + TH + 4), f"{tag} {kw}", fill=(30, 30, 30))
    sheet.save(os.path.join(TMPD, "diverse_sheet.png"))

    # 封面间两两距离
    hs = [picked[pi][3] for pi in sorted(picked)]
    dmin = 999
    for i in range(len(hs)):
        for j in range(i + 1, len(hs)):
            dmin = min(dmin, (hs[i] != hs[j]).sum())
    print(f"封面两两最小 dhash 距离: {dmin}  (旧版最低 4)")
    sheet.save(os.path.join(TMPD, "diverse_sheet.png"))

    if not WRITE:
        print("(仅拼版预览，未写入。确认后加 --write)")
        return
    # 阶段4：写入
    from pack_final import fin
    for pi, c in sorted(picked.items()):
        _, name, kw, h, tight = c
        pack = packs[pi]
        tag = f"第{pi+1:02d}包" if len(pack) == 16 else f"剩余{len(pack)}张"
        cov = fin(tight, 200)
        cp = os.path.join(OUT, tag, "表情封面图.png")
        n, u = save_png_under(fit_from_big(cov, 200), cp, 100 * 1024)
        print(f"写入 {tag}: {kw}  {n/1024:.1f}KB({u})")


if __name__ == "__main__":
    main()

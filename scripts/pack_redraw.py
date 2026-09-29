# -*- coding: utf-8 -*-
"""把 redraw3.py 产出的「每贴纸 3 张重绘图」按原 QQ 表情包分组打包。

分组规则与 pack_final/pack_sr 完全一致（同 collect_items 顺序，16 个不同关键字一包），
所以「第01包」里的关键字与平台包里的一一对应，便于对照。

输出:
  <OUT>/第01包/打call_01_s11.png / _s22.png / _s33.png  ...
  <OUT>/第01包/_总览.png            # 16 关键字 × 3 张 的缩略接触表
  <OUT>/第01包/重绘.zip             # 该包 48 张 PNG 打包

用法:
  python pack_redraw.py                 # 全量
  python pack_redraw.py --only 0        # 只跑第 1 包
  python pack_redraw.py --seed 11       # 只挑某个 seed 作为基准（默认 3 张全放）
"""
import os, sys, io, time, zipfile, shutil, warnings
warnings.filterwarnings("ignore")
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pack_sr import content_bbox, collect_items, keyword_of, fix_keyword, robust_write, TMPDIR

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "redraw3")
OUT = r"C:\Users\Admin\OneDrive\Desktop\上传_重绘"
SEEDS = [11, 22, 33]


def find_raw(kw, seed):
    """在 redraw3 里找 <kw>_s<seed>*.png（忽略参数后缀，取文件名最短的稳定版）"""
    if not os.path.isdir(RAW):
        return None
    cands = [f for f in os.listdir(RAW)
             if f.startswith(f"{kw}_s{seed}") and f.endswith(".png") and not f.startswith("_")]
    if not cands:
        return None
    cands.sort(key=len)
    return os.path.join(RAW, cands[0])


def contact(files, path, title=""):
    """一包接触表：行=关键字，列=3 个 seed"""
    T, PAD = 172, 6
    rows = len(files)
    W = 3 * (T + PAD) + PAD
    H = rows * (T + 22) + 30
    sh = Image.new("RGB", (W, H), (246, 246, 246))
    dr = ImageDraw.Draw(sh)
    if title:
        dr.text((8, 6), title, fill=(20, 20, 20))
    for r, (kw, paths) in enumerate(files):
        y = 24 + r * (T + 22)
        dr.text((8, y), kw, fill=(20, 20, 20))
        for c, p in enumerate(paths):
            if p is None or not os.path.exists(p):
                continue
            im = Image.open(p).convert("RGB")
            im.thumbnail((T, T), Image.LANCZOS)
            bg = Image.new("RGB", (T, T), (255, 255, 255))
            bg.paste(im, ((T - im.width) // 2, (T - im.height) // 2))
            sh.paste(bg, (PAD + c * (T + PAD), y + 16))
    sh.save(path)
    return sh.size


def main():
    args = sys.argv[1:]
    only = int(args[args.index("--only") + 1]) if "--only" in args else None
    seeds = [int(args[args.index("--seed") + 1])] if "--seed" in args else SEEDS

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
    total_missing = []
    print(f"共 {len(items)} 张 -> {len(packs)} 组，每组 16 关键字 × {len(seeds)} 张", flush=True)

    for pi, pack in enumerate(packs):
        if only is not None and pi != only:
            continue
        tag = f"第{pi+1:02d}包" if len(pack) == 16 else f"剩余{len(pack)}张"
        pdir = os.path.join(OUT, tag)
        os.makedirs(pdir, exist_ok=True)
        t0 = time.time()
        sheet_rows = []
        zfiles = []
        print(f"\n== {tag}: {len(pack)} 个 ==", flush=True)
        for i, (name, kw) in enumerate(pack, 1):
            paths = []
            for sd in seeds:
                src = find_raw(kw, sd)
                if src is None:
                    total_missing.append(f"{kw}_s{sd}")
                    paths.append(None)
                    continue
                dst_name = f"{kw}_{i:02d}_s{sd}.png"
                dst = os.path.join(pdir, dst_name)
                shutil.copy2(src, dst)
                zfiles.append((dst, dst_name))
                paths.append(dst)
            sheet_rows.append((kw, paths))
            print(f"   {kw}_{i:02d} ok", end="", flush=True)
        print(f"  ({time.time()-t0:.1f}s)", flush=True)

        sz = contact(sheet_rows, os.path.join(pdir, "_总览.png"),
                     f"{tag}  {len(pack)} 关键字 × {len(seeds)} 张（列=seed {seeds}）")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for p, n in zfiles:
                z.writestr(n, open(p, "rb").read())
        robust_write(os.path.join(pdir, "重绘.zip"), buf.getvalue())
        print(f"   总览 {sz} | 重绘.zip {len(buf.getvalue())//1024}KB ({len(zfiles)} 张)", flush=True)

    print(f"\nALL DONE -> {OUT}")
    if total_missing:
        print(f"缺失 {len(total_missing)}: {total_missing[:20]}")


if __name__ == "__main__":
    main()

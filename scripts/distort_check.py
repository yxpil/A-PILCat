# -*- coding: utf-8 -*-
"""形变体检：成品内容宽高比 vs 源图宽高比，以及内容是否被裁歪/居中"""
import os, sys, zipfile, warnings
warnings.filterwarnings("ignore")
from PIL import Image
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pack_sr import SRC, collect_items, keyword_of, fix_keyword

_items = collect_items()
PACKS = []
for _n in _items:
    _kw = fix_keyword(keyword_of(_n))
    for _p in PACKS:
        if len(_p) < 16 and _kw not in [x[1] for x in _p]:
            _p.append((_n, _kw))
            break
    else:
        PACKS.append([(_n, _kw)])

OUT = r"C:\Users\Admin\OneDrive\Desktop\上传"


def content_box(im):
    """内容实心像素的 bbox（描边/留白之外的实际内容）"""
    a = np.array(im.split()[3])
    solid = a > 200
    if not solid.any():
        return None
    ys, xs = np.where(solid)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def box_of_arr(a):
    ys, xs = np.where(a > 200)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


print(f"{'包/文件':<26} {'源比例':>7} {'成品比例':>8} {'偏差%':>7} {'填充率':>7} {'居中偏移':>9}")
worst = []

for d in sorted(os.listdir(OUT)):
    pd = os.path.join(OUT, d)
    if not os.path.isdir(pd) or d == "Example":
        continue
    zp = os.path.join(pd, "表情缩略图.zip")
    if not os.path.exists(zp):
        continue
    # 包内成员顺序 -> zip 内文件名 f"{kw}_{i:02d}.png"
    pi = sorted(os.listdir(OUT)).index(d)
    members = PACKS[pi]
    with zipfile.ZipFile(zp) as z:
        for n in z.namelist():
            stem = os.path.splitext(os.path.basename(n))[0]
            idx = int(stem.split("_")[-1]) - 1
            if not (0 <= idx < len(members)):
                continue
            src_im = Image.open(os.path.join(SRC, members[idx][0])).convert("RGBA")
            sbb = content_box(src_im)
            if sbb is None:
                continue
            s_ratio = (sbb[2] - sbb[0]) / max(1, sbb[3] - sbb[1])

            im = Image.open(z.open(n)).convert("RGBA")
            cbb = content_box(im)
            if cbb is None:
                continue
            c_ratio = (cbb[2] - cbb[0]) / max(1, cbb[3] - cbb[1])
            dev = abs(c_ratio - s_ratio) / s_ratio * 100
            W = im.width
            fill = ((cbb[2] - cbb[0]) * (cbb[3] - cbb[1])) / (W * W) * 100
            # 居中偏移：内容 bbox 中心相对画布中心
            cx = (cbb[0] + cbb[2]) / 2 - W / 2
            cy = (cbb[1] + cbb[3]) / 2 - W / 2
            off = max(abs(cx), abs(cy))
            tag = f"{d}/{stem}"
            print(f"{tag:<26} {s_ratio:>7.3f} {c_ratio:>8.3f} {dev:>7.2f} "
                  f"{fill:>6.1f}% {off:>8.1f}px")
            worst.append((dev, tag, s_ratio, c_ratio, fill, off))

worst.sort(reverse=True)
print("\n=== 比例偏差最大的 10 张 ===")
for dev, tag, sr, cr, fill, off in worst[:10]:
    print(f"  {dev:6.2f}%  {tag:<26} 源{sr:.3f} 成品{cr:.3f} 填充{fill:.0f}%")

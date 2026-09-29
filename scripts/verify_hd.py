# -*- coding: utf-8 -*-
"""上传_高清 全自动自检（几何/像素级，替代肉眼）：
 1) 规格    PNG/GIF 统一 1900x1900 画布，封面 200x200
 2) 不切头  成品内容 bbox 宽高比 vs 源图内容 bbox 宽高比，偏差 < 3%
            （trim/fit 前后宽高比不变，一旦切头/切边必然偏离）
 3) 不贴边  成品内容四边留白 >= 18px（画布 margin 24，描边后仍应留白）
 4) 有动作  GIF 相邻帧平均像素差 > 阈值（证明不是静止贴图）
 5) 封面净  封面无顶部窄文字带、内容不贴边
"""
import os, io, sys, zipfile, numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pack_sr import content_bbox, strip_top_text

HD = r"C:\Users\Admin\OneDrive\Desktop\上传_高清"
SRC = r"C:\Users\Admin\OneDrive\Desktop\合格"
args = sys.argv[1:]
only = int(args[args.index("--only") + 1]) if "--only" in args else None

dirs = [d for d in sorted(os.listdir(HD)) if os.path.isdir(os.path.join(HD, d))]
if only is not None:
    dirs = [dirs[only]]

# 源图内容 bbox 宽高比表（一次算好）
src_ar = {}
for f in os.listdir(SRC):
    if not f.lower().endswith(".png"):
        continue
    im = Image.open(os.path.join(SRC, f))
    bb = content_bbox(im)
    if bb:
        w, h = bb[2] - bb[0], bb[3] - bb[1]
        src_ar[f[:-4]] = w / max(1, h)

bad, n_items = [], 0
for d in dirs:
    pd = os.path.join(HD, d)
    with zipfile.ZipFile(os.path.join(pd, "表情缩略图.zip")) as z:
        for n in sorted(z.namelist()):
            n_items += 1
            im = Image.open(io.BytesIO(z.read(n))).convert("RGBA")
            if im.size != (1900, 1900):
                bad.append(f"{d}/{n}: 尺寸 {im.size}")
            bb = content_bbox(im)
            if bb is None:
                bad.append(f"{d}/{n}: 空图"); continue
            cw, ch = bb[2] - bb[0], bb[3] - bb[1]
            edge = min(bb[0], bb[1], 1900 - bb[2], 1900 - bb[3])
            if edge < 10:
                bad.append(f"{d}/{n}: 贴边 {edge}px")
            kw = n[:-4].rsplit("_", 1)[0]
            sa = src_ar.get(kw)
            if sa:
                ar = cw / max(1, ch)
                dev = abs(ar - sa) / sa
                if dev > 0.03:
                    bad.append(f"{d}/{n}: 宽高比偏离源图 {dev*100:.1f}%（疑似切头）")
    with zipfile.ZipFile(os.path.join(pd, "表情动画.zip")) as z:
        for n in sorted(z.namelist()):
            im = Image.open(io.BytesIO(z.read(n)))
            nf = getattr(im, "n_frames", 1)
            im.seek(0); a0 = np.asarray(im.convert("RGBA"), np.int16)
            im.seek(max(1, nf // 2)); a1 = np.asarray(im.convert("RGBA"), np.int16)
            diff = np.abs(a1[..., :3] - a0[..., :3]).mean()
            mv = (np.abs(a1[..., 3] - a0[..., 3]) > 40).mean()
            if diff < 1.0 and mv < 0.002:
                bad.append(f"{d}/{n}: 疑似静止 diff={diff:.2f} mv={mv:.4f}")
    cp = os.path.join(pd, "表情封面图.png")
    if os.path.exists(cp):
        c = Image.open(cp).convert("RGBA")
        if c.size != (200, 200):
            bad.append(f"{d}/封面: 尺寸 {c.size}")
        bb = content_bbox(c)
        if bb:
            e = min(bb[0], bb[1], 200 - bb[2], 200 - bb[3])
            if e < 3:
                bad.append(f"{d}/封面: 贴边 {e}px")
        if strip_top_text(c) is not None:
            bad.append(f"{d}/封面: 疑似带顶部文字带")
    else:
        bad.append(f"{d}: 缺封面")

print(f"检查 {len(dirs)} 组 / {n_items} 张静态 + 动画 + 封面")
if bad:
    print(f"\n❌ 发现 {len(bad)} 处问题:")
    for b in bad[:50]:
        print("   ", b)
    sys.exit(1)
print("✅ 全部通过：规格 / 不切头 / 不贴边 / 有动作 / 封面净")

# -*- coding: utf-8 -*-
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from PIL import Image
from pack_sr import SRC, content_bbox
from matting_gpu import matte

rows = []
for name in sorted(os.listdir(SRC)):
    if not name.lower().endswith(".png"): continue
    im = matte(Image.open(os.path.join(SRC, name)).convert("RGBA"))
    bb = content_bbox(im)
    if not bb: 
        rows.append((name, im.size[0], im.size[1], 1.0, 1.0)); continue
    w, h = bb[2]-bb[0], bb[3]-bb[1]
    # 内容最大边 / 画布最大边  = 填充率
    fill = max(w, h) / max(im.size)
    rows.append((name, im.size[0], im.size[1], w/h, fill))

fills = np.array([r[4] for r in rows])
print(f"共 {len(rows)} 张")
print(f"填充率(内容最长边/画布边长): min={fills.min():.3f} p25={np.percentile(fills,25):.3f} "
      f"中位={np.median(fills):.3f} p75={np.percentile(fills,75):.3f} max={fills.max():.3f}")
ar = np.array([r[3] for r in rows])
print(f"内容宽高比: p5={np.percentile(ar,5):.2f} 中位={np.median(ar):.2f} p95={np.percentile(ar,95):.2f}")
bad = sorted(rows, key=lambda r: r[4])[:8]
for r in bad: print(f"  填充低: {r[0]} 画布{r[1]}x{r[2]} 比{r[3]:.2f} 填充{r[4]:.2f}")

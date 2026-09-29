# -*- coding: utf-8 -*-
"""几何保真测试：证明管线（superres + fit_from_big）不产生形变

管线上只有两个几何操作：
  superres()     等比缩到 lq_max 后 x4 超分
  fit_from_big() min 比例等比缩放 + 居中贴入正方形画布
其余（去溢色/降噪/描边）都是逐像素或形态学操作，几何上不会变形。

探针：正圆（非等比->椭圆）、45° 斜线（非等比->斜率偏离 1）。
每组都跑"直接 LANCZOS 放大同样倍率"作对照，用于区分"超分变形"与"测量方法偏差"。
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
from PIL import Image, ImageDraw
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pack_sr import fit_from_big, superres

SRC_SIZE = 256


# ---------- 测试图 ----------
def make_circle(w=SRC_SIZE, h=SRC_SIZE):
    im = Image.new("RGB", (w, h), (255, 255, 255))
    d = ImageDraw.Draw(im)
    R = 40
    d.ellipse([64 - R, 128 - R, 64 + R, 128 + R], fill=(0, 0, 0))      # 正圆
    L = 40
    for k in range(-2, 3):                                             # 45° 斜线
        d.line([190 - L + k * 9, 128 - L, 190 + L + k * 9, 128 + L],
               fill=(0, 0, 0), width=3)
    return im


# ---------- 量测 ----------
def dark_bbox(arr, box):
    x0, y0, x1, y1 = box
    sub = arr[y0:y1, x0:x1]
    m = (sub[..., 0] < 90) & (sub[..., 1] < 90) & (sub[..., 2] < 90)
    ys, xs = np.where(m)
    if len(xs) < 20:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def slope(arr, box):
    x0, y0, x1, y1 = box
    sub = arr[y0:y1, x0:x1]
    m = (sub[..., 0] < 90) & (sub[..., 1] < 90) & (sub[..., 2] < 90)
    pts = np.argwhere(m)
    if len(pts) < 30:
        return None
    y, x = pts[:, 0].astype(float), pts[:, 1].astype(float)
    bins = np.linspace(x.min(), x.max(), 15)
    xs, ys = [], []
    for i in range(len(bins) - 1):
        sel = (x >= bins[i]) & (x < bins[i + 1])
        if sel.sum() >= 2:
            xs.append(x[sel].mean()); ys.append(y[sel].mean())
    if len(xs) < 6:
        return None
    xs, ys = np.array(xs), np.array(ys)
    A = np.vstack([xs, np.ones_like(xs)]).T
    return abs(float(np.linalg.lstsq(A, ys, rcond=None)[0][0]))


def report(label, out):
    W = out.shape[1]
    s = W / SRC_SIZE
    box_c = (int(24 * s), int(88 * s), int(104 * s), int(168 * s))
    box_s = (int(150 * s), int(88 * s), int(230 * s), int(168 * s))
    b = dark_bbox(out, box_c)
    w, h = (b[2] - b[0] + 1), (b[3] - b[1] + 1)
    sp = slope(out, box_s)
    print(f"  {label:<16} 圆盘 {w:>3}x{h:<3} 宽高比 {w/max(1,h):.3f}   "
          f"斜线斜率 {(f'{sp:.3f}' if sp else 'n/a')}")
    return w / max(1, h), sp


if __name__ == "__main__":
    os.makedirs("bgtest", exist_ok=True)
    src = make_circle()

    # ① 对照组：不经超分，直接 LANCZOS 放大 4 倍（几何上的"真理"）
    ref = np.array(src.resize((SRC_SIZE * 4, SRC_SIZE * 4), Image.LANCZOS))
    print("=== 对照组：直接 LANCZOS x4（无超分）===")
    r_r, r_s = report("LANCZOS x4", ref)
    Image.fromarray(ref).save("bgtest/geom_ref.png")

    # ② 超分路径
    print("=== 超分路径 superres() ===")
    out = np.array(superres(src).convert("RGB"))
    g_r, g_s = report("Real-ESRGAN x4", out)
    Image.fromarray(out).save("bgtest/geom_sr.png")

    # ③ fit_from_big 等比性
    print("\n=== fit_from_big() 等比性（非正方形输入）===")
    for (w, h) in [(300, 150), (150, 300), (377, 211), (900, 900)]:
        im = Image.new("RGBA", (w, h), (0, 0, 0, 255))
        o = fit_from_big(im, 300)
        a = np.array(o.split()[3]) > 200
        ys, xs = np.where(a)
        ratio = (xs.max() - xs.min() + 1) / max(1, (ys.max() - ys.min() + 1))
        print(f"  输入 {w:>3}x{h:<3} 比例 {w/h:.3f} -> 输出 {ratio:.3f}  "
              f"偏差 {abs(ratio-w/h)/(w/h)*100:.2f}%")

    ok = abs(g_r - 1) < 0.02
    print(f"\n圆盘宽高比: 对照 {r_r:.3f} / 超分 {g_r:.3f}  -> "
          f"{'等比 ✓' if ok else '疑似变形 ✗'}")

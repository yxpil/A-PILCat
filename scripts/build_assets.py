"""生成 表情缩略图 (300x300 PNG) 与 表情封面图 (200x200 PNG，去文字)。
去文字只作用于封面图：用连通域找到顶部文字块及其描边整体抹除。
"""
import os, io
import numpy as np
from PIL import Image
from scipy import ndimage

SRC = r"C:\Users\Admin\OneDrive\Desktop\合格"
THUMB_DIR = r"C:\Users\Admin\OneDrive\Desktop\上传\表情缩略图"
COVER_DIR = r"C:\Users\Admin\OneDrive\Desktop\上传\表情封面图"
os.makedirs(THUMB_DIR, exist_ok=True)
os.makedirs(COVER_DIR, exist_ok=True)

# 上一套 16 个的编号沿用，方便和旧的 zip 一一对应
OLD = [
    ("哈哈哈", 1), ("比心", 2), ("点赞", 3), ("撒花", 4), ("加油", 5), ("害羞", 6),
    ("思考", 7), ("吃瓜", 8), ("谢谢", 9), ("震惊", 10), ("心累", 11), ("摸鱼", 12),
    ("躺平", 13), ("大无语", 14), ("愤怒", 15), ("哭泣", 16),
]
EXTRA = [("打call", None)]  # 新增，编号待定


def strip_text(im):
    """把顶部文字块（含描边）抹掉，返回新图"""
    rgba = im.convert("RGBA")
    a = np.array(rgba)
    alpha = a[..., 3] > 24
    if not alpha.any():
        return rgba
    lab, n = ndimage.label(alpha)
    if n < 2:
        return rgba
    areas = ndimage.sum(alpha, lab, range(1, n + 1))
    order = np.argsort(areas)[::-1]  # 由大到小
    main = order[0] + 1
    rgb = a[..., :3].astype(np.int16)
    lum = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    dark = lum < 110

    slices_all = []
    for idx, comp in enumerate(order):
        cid = int(comp) + 1
        if cid == main:
            continue
        ys, xs = np.where(lab == cid)
        H, W = alpha.shape
        bw, bh = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
        if bw < 0.12 * W or ys.min() > 0.45 * H:
            continue
        if dark[lab == cid].sum() < 0.05 * areas[comp]:
            continue
        slices_all.append((cid, (xs.min(), ys.min(), xs.max(), ys.max())))
    if not slices_all:
        return rgba

    tx0, ty0, tx1, ty1 = 1e9, 1e9, -1, -1
    for _, (x0, y0, x1, y1) in slices_all:
        tx0, ty0 = min(tx0, x0), min(ty0, y0)
        tx1, ty1 = max(tx1, x1), max(ty1, y1)
    # 落在文字包围盒内的附属碎片（描边、噪点）一起清掉
    keep = ~inside(lab, tx0, ty0, tx1, ty1)
    out = a.copy()
    out[..., 3] = np.where(keep, out[..., 3], 0)
    return Image.fromarray(out, "RGBA")


def inside(lab, x0, y0, x1, y1):
    H, W = lab.shape
    m = np.zeros((H, W), bool)
    y0, y1 = max(0, y0), min(H, y1 + 1)
    x0, x1 = max(0, x0), min(W, x1 + 1)
    if y1 > y0 and x1 > x0:
        m[y0:y1, x0:x1] = True
    return m


def fit_canvas(im, size, margin=0, background=None):
    """裁剪透明边 -> 等比缩放 -> 居中放到 size x size 画布"""
    src = im.convert("RGBA")
    bbox = src.split()[3].point(lambda p: 255 if p > 24 else 0).getbbox()
    if bbox:
        pad = 4
        src = src.crop((
            max(0, bbox[0] - pad), max(0, bbox[1] - pad),
            min(src.width, bbox[2] + pad), min(src.height, bbox[3] + pad)))
    inner = size - margin * 2
    scale = min(inner / src.width, inner / src.height)
    nw, nh = max(1, round(src.width * scale)), max(1, round(src.height * scale))
    src = src.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGBA", (size, size), background or (0, 0, 0, 0))
    canvas.paste(src, ((size - nw) // 2, (size - nh) // 2), src)
    return canvas


def save_codec(im, path, limit):
    """先存 RGBA，超过限制再逐步量化"""
    best = None
    for colors in [None, 256, 128]:
        buf = io.BytesIO()
        out = im
        if colors:
            if im.mode != "RGBA":
                out = im
                q = im.convert("RGBA").quantize(colors=colors, method=Image.FASTOCTREE)
                out = q.convert("RGBA")
            else:
                out = im.quantize(colors=colors, method=Image.FASTOCTREE).convert("RGBA")
        out.save(buf, "PNG", optimize=True)
        if buf.tell() <= limit:
            data = buf.getvalue()
            with open(path, "wb") as f:
                f.write(data)
            print(f"  {os.path.basename(path)} {im.width}x{im.height} "
                  f"{len(data)/1024:.1f} KB (colors={colors})")
            return True
        if best is None or len(buf.getvalue()) < len(best):
            best = buf.getvalue()
    with open(path, "wb") as f:
        f.write(best)
    print(f"  ! {os.path.basename(path)} still {len(best)/1024:.1f} KB")
    return False


jobs = OLD + EXTRA
print("== 表情缩略图 ==")
for name, num in jobs:
    src = os.path.join(SRC, f"{name}.png")
    im = Image.open(src).convert("RGBA")
    canvas = fit_canvas(im, 300, margin=6)
    save_codec(canvas, os.path.join(THUMB_DIR, f"{name}_{num if num else '17'}.png"), 200 * 1024)

print("== 表情封面图 (去文字) ==")
cover_name = "打call"
im = Image.open(os.path.join(SRC, f"{cover_name}.png")).convert("RGBA")
clean = strip_text(im)
print("  文字块清除情况:", clean.split()[3].point(lambda p: 255 if p > 24 else 0).getbbox(),
      "| 原:", im.split()[3].getbbox())
canvas = fit_canvas(clean, 200, margin=6)
save_codec(canvas, os.path.join(COVER_DIR, "表情封面图.png"), 100 * 1024)
print("done")

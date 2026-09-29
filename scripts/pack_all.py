# -*- coding: utf-8 -*-
"""按 QQ 表情包规范分包：
每包 16 个 -> 表情动画.zip(单帧gif,白描边,300x300,<300KB) + 表情缩略图.zip(png,<200KB) + 表情封面图.png(200x200,<100KB,无文字)
"""
import os, io, re, zipfile
import numpy as np
from PIL import Image, ImageFilter

SRC = r"C:\Users\Admin\OneDrive\Desktop\合格"
OUT = r"C:\Users\Admin\OneDrive\Desktop\上传"
DCALL_SRC = r"C:\Users\Admin\OneDrive\Desktop\上传\打call.png"  # 无文字版

# ---------- 基础工具 ----------

def content_bbox(im, thresh=24):
    return im.split()[3].point(lambda p: 255 if p > thresh else 0).getbbox()

def trim(im, pad=4):
    bb = content_bbox(im)
    if not bb:
        return im
    l, t, r, b = bb
    return im.crop((max(0, l - pad), max(0, t - pad),
                    min(im.width, r + pad), min(im.height, b + pad)))

def fit(im, size, margin=6):
    src = trim(im)
    s = min((size - margin * 2) / src.width, (size - margin * 2) / src.height)
    nw, nh = max(1, round(src.width * s)), max(1, round(src.height * s))
    src = src.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    canvas.paste(src, ((size - nw) // 2, (size - nh) // 2), src)
    return canvas

def add_white_stroke(im, px=3):
    """在透明图下方加白色描边"""
    a = im.split()[3]
    dil = a.filter(ImageFilter.MaxFilter(px * 2 + 1))
    stroke = Image.new("RGBA", im.size, (255, 255, 255, 0))
    white = Image.new("RGBA", im.size, (255, 255, 255, 255))
    stroke.paste(white, mask=dil)
    out = Image.alpha_composite(stroke, im)
    return out

def save_png_under(im, path, limit):
    best = None
    for colors in [None, 256, 128, 64]:
        buf = io.BytesIO()
        out = im if colors is None else \
            im.quantize(colors=colors, method=Image.FASTOCTREE).convert("RGBA")
        out.save(buf, "PNG", optimize=True)
        if best is None or buf.tell() < len(best):
            best = buf.getvalue()
        if buf.tell() <= limit:
            break
    open(path, "wb").write(best)
    return len(best)

def save_gif(im, path, limit=300 * 1024):
    """RGBA -> 单帧 GIF（透明底）"""
    q = im.quantize(colors=255, method=Image.FASTOCTREE)
    # 把完全透明的像素归到索引255并设为透明色
    arr = np.array(q)
    alpha = np.array(im.split()[3])
    arr[alpha < 24] = 255
    q = Image.fromarray(arr, "P")
    buf = io.BytesIO()
    q.save(buf, "GIF", transparency=255, optimize=True)
    data = buf.getvalue()
    if len(data) > limit:  # 减色重试
        for colors in [128, 64]:
            q2 = im.quantize(colors=colors, method=Image.FASTOCTREE)
            arr = np.array(q2); arr[np.array(im.split()[3]) < 24] = colors
            q2 = Image.fromarray(arr, "P")
            buf = io.BytesIO(); q2.save(buf, "GIF", transparency=colors, optimize=True)
            data = buf.getvalue()
            if len(data) <= limit:
                break
    open(path, "wb").write(data)
    return len(data)

# ---------- 封面：顶部文字带检测 + 裁剪 ----------

def strip_top_text(im):
    """检测顶部文字带（其后有明确空隙），裁掉；失败返回 None"""
    a = np.array(im.convert("RGBA"))
    alpha = a[..., 3] > 60
    H, W = alpha.shape
    rows = alpha.sum(axis=1)
    thr = max(3, int(0.003 * W))
    # 找第一段连续内容带
    y = 0
    while y < H and rows[y] < thr:
        y += 1
    if y >= H:
        return None
    band_top = y
    gap = 0
    band_bot = y
    yy = y
    while yy < H:
        if rows[yy] < thr:
            gap += 1
            if gap >= max(4, int(0.012 * H)):
                band_bot = yy - gap
                break
        else:
            gap = 0
            band_bot = yy
        yy += 1
    band_h = band_bot - band_top
    # 文件主体(带下方)必须还有足够内容
    below = rows[band_bot + 1:].sum()
    if band_h > 0.32 * H or below < 0.15 * rows.sum() or band_bot <= band_top:
        return None
    xs = np.where(alpha[band_top:band_bot + 1].any(axis=0))[0]
    if len(xs) == 0 or (xs.max() - xs.min()) < 0.22 * W:
        return None  # 太窄，不像标题文字
    out = im.crop((0, band_bot + 1, W, H))
    return out if content_bbox(out) else None

# ---------- 分包 ----------

def keyword_of(name):
    k = re.sub(r"_(?:2|3)$", "", os.path.splitext(name)[0])
    return k

def fix_keyword(k):
    cjk = len([c for c in k if ord(c) > 0x2E80])
    if cjk > 4:
        return k[:4] if ord(k[3]) > 0x2E80 else k[:5]
    return k

# 收集去重清单
seen, items = {}, []
for name in sorted(os.listdir(SRC)):
    if name.lower().endswith(".png"):
        import hashlib
        h = hashlib.md5(open(os.path.join(SRC, name), "rb").read()).hexdigest()
        if h in seen:
            continue
        seen[h] = name
        items.append(name)
# 打call 插到最前（有现成无文字版可做封面）
items.remove("打call.png")
items.insert(0, "打call.png")

# 贪心分组：同包关键词不重复
packs = []
for name in items:
    kw = fix_keyword(keyword_of(name))
    placed = False
    for p in packs:
        if len(p) < 16 and kw not in [x[1] for x in p]:
            p.append((name, kw))
            placed = True
            break
    if not placed:
        packs.append([(name, kw)])
print(f"共 {len(items)} 张 -> {len([p for p in packs if len(p)==16])} 个整包 + "
      f"{len(packs[-1])} 张剩余" if packs[-1] and len(packs[-1]) < 16 else "")

for pi, pack in enumerate(packs):
    tag = f"第{pi+1:02d}包" if len(pack) == 16 else f"剩余{len(pack)}张"
    pdir = os.path.join(OUT, tag)
    os.makedirs(pdir, exist_ok=True)
    print(f"== {tag}: {len(pack)} 个 ==")

    gif_paths, png_paths = [], []
    for i, (name, kw) in enumerate(pack, 1):
        src = Image.open(os.path.join(SRC, name)).convert("RGBA")
        # 动画（静态单帧，白描边）
        g = add_white_stroke(fit(src, 300, margin=6), px=2)
        gp = os.path.join(pdir, "_tmp.gif")
        n = save_gif(g, gp)
        gif_paths.append((gp, f"{kw}_{i:02d}.gif", n))
        # 缩略图（PNG，保留文字）
        t = fit(src, 300, margin=6)
        tp = os.path.join(pdir, "_tmp.png")
        n2 = save_png_under(t, tp, 200 * 1024)
        png_paths.append((tp, f"{kw}_{i:02d}.png", n2))

    # 打 zip（UTF-8 文件名）
    for zname, lst in [("表情动画.zip", gif_paths), ("表情缩略图.zip", png_paths)]:
        zpath = os.path.join(pdir, zname)
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            for tmp, final, n in lst:
                z.write(tmp, final)
                os.remove(tmp)
        print(f"   {zname}: {len(lst)} 文件, {os.path.getsize(zpath)//1024} KB")

    # 封面：优先打call(无文字源)，否则选第一个能干净裁掉文字带的
    cover = None
    for name, kw in pack:
        if name == "打call.png":
            cover = Image.open(DCALL_SRC).convert("RGBA")
            print(f"   封面: {kw} (无文字源图)")
            break
        try:
            cand = strip_top_text(Image.open(os.path.join(SRC, name)).convert("RGBA"))
        except Exception as e:
            cand = None
        if cand is not None:
            cover = cand
            print(f"   封面: {kw} (裁掉顶部文字带 {name})")
            break
    if cover is None:
        # 兜底：用打call的无文字版
        cover = Image.open(DCALL_SRC).convert("RGBA")
        print("   封面: 兜底打call")
    cp = os.path.join(pdir, "表情封面图.png")
    n = save_png_under(fit(cover, 200, margin=6), cp, 100 * 1024)
    print(f"   表情封面图.png {n/1024:.1f} KB")

print("ALL DONE")

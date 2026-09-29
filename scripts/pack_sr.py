# -*- coding: utf-8 -*-
"""QQ 表情包分包（超分版，无动画）：
每包 = 表情缩略图.zip(16 张 300x300 PNG <200KB) + 表情封面图.png(200x200 <100KB 无文字)
"""
import os, io, re, sys, time, hashlib, zipfile, warnings, tempfile
warnings.filterwarnings("ignore")
import numpy as np
import torch
from PIL import Image, ImageFilter
from rrdb import load_model

HERE = os.path.dirname(os.path.abspath(__file__))
TMPDIR = tempfile.mkdtemp(prefix="qqpack_")   # 系统临时目录，避免 OneDrive/删除钩子干扰
SRC = r"C:\Users\Admin\OneDrive\Desktop\合格"
OUT = r"C:\Users\Admin\OneDrive\Desktop\上传"
DCALL = os.path.join(OUT, "打call.png")   # 无文字版（若存在则优先）
DCALL_FALLBACK = os.path.join(SRC, "打call.png")

net, DEV = load_model()


# ---------- 基础 ----------
def content_bbox(im, thresh=24):
    return im.split()[3].point(lambda p: 255 if p > thresh else 0).getbbox()


def to_t(im):
    a = np.array(im).astype(np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(DEV)


def from_t(t):
    a = t.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    return Image.fromarray(np.clip(a * 255.0, 0, 255).astype(np.uint8), "RGB")


def superres(rgb, lq_max=256):
    if max(rgb.width, rgb.height) > lq_max:
        s = lq_max / max(rgb.width, rgb.height)
        rgb = rgb.resize((max(1, round(rgb.width * s)), max(1, round(rgb.height * s))),
                         Image.LANCZOS)
    with torch.no_grad():
        return from_t(net(to_t(rgb)))


def sr_rgba(im, lq_max=256):
    """RGBA -> 超分 x4（alpha 同步放大 + 软阈值去毛边）"""
    a = im.split()[3]
    flat = Image.new("RGB", im.size, (255, 255, 255))
    flat.paste(im, mask=a)
    big = superres(flat, lq_max)
    na = a.resize(big.size, Image.LANCZOS).filter(ImageFilter.GaussianBlur(0.5)) \
          .point(lambda p: 255 if p > 170 else (0 if p < 90 else p))
    big = big.convert("RGBA"); big.putalpha(na)
    return big


def fit_from_big(big, size, margin=6):
    """已超分的大图 -> 缩到 size 画布（供缩略图/封面复用同一次超分）"""
    s = min((size - margin * 2) / big.width, (size - margin * 2) / big.height)
    nw, nh = max(1, round(big.width * s)), max(1, round(big.height * s))
    src = big.resize((nw * 2, nh * 2), Image.LANCZOS).resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    canvas.paste(src, ((size - nw) // 2, (size - nh) // 2), src)
    return canvas


def robust_write(path, data, tries=6):
    """写文件：OneDrive 同步进程会短暂占用目标文件，覆盖/删除被拒时改用 rename 让位。
    返回写入字节数。"""
    import time
    tmp = path + ".tmp%d" % os.getpid()
    last = None
    for _ in range(tries):
        try:
            open(tmp, "wb").write(data)
            try:
                os.replace(tmp, path)          # 首选：原子替换
            except OSError:
                if os.path.exists(path):       # 被占用：先把旧文件挪开
                    os.rename(path, path + ".old%d" % os.getpid())
                os.replace(tmp, path)
            return len(data)
        except OSError as e:
            last = e
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except OSError:
                pass
            time.sleep(0.4)
    raise last


def save_png_under(im, path, limit):
    best, used = None, None
    for colors in [None, 256, 192, 128, 96, 64]:
        buf = io.BytesIO()
        out = im if colors is None else im.quantize(
            colors=colors, method=Image.FASTOCTREE,
            dither=Image.Dither.FLOYDSTEINBERG).convert("RGBA")
        out.save(buf, "PNG", optimize=True)
        if best is None or buf.tell() < len(best):
            best, used = buf.getvalue(), (colors or "full")
        if buf.tell() <= limit:
            break
    robust_write(path, best)
    return len(best), used


# ---------- 封面：顶部文字带检测 + 裁剪 ----------
def strip_top_text(im):
    a = np.array(im.convert("RGBA"))
    alpha = a[..., 3] > 60
    H, W = alpha.shape
    rows = alpha.sum(axis=1)
    thr = max(3, int(0.003 * W))
    y = 0
    while y < H and rows[y] < thr:
        y += 1
    if y >= H:
        return None
    band_top, band_bot, gap, yy = y, y, 0, y
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
    below = rows[band_bot + 1:].sum()
    if band_h > 0.32 * H or below < 0.15 * rows.sum() or band_bot <= band_top:
        return None
    xs = np.where(alpha[band_top:band_bot + 1].any(axis=0))[0]
    if len(xs) == 0 or (xs.max() - xs.min()) < 0.22 * W:
        return None
    out = im.crop((0, band_bot + 1, W, H))
    return out if content_bbox(out) else None


# ---------- 分包 ----------
def keyword_of(name):
    return re.sub(r"_(?:2|3)$", "", os.path.splitext(name)[0])


def fix_keyword(k):
    cjk = len([c for c in k if ord(c) > 0x2E80])
    if cjk > 4:
        return k[:4] if ord(k[3]) > 0x2E80 else k[:5]
    return k


def collect_items():
    seen, items = {}, []
    for name in sorted(os.listdir(SRC)):
        if not name.lower().endswith(".png"):
            continue
        h = hashlib.md5(open(os.path.join(SRC, name), "rb").read()).hexdigest()
        if h in seen:
            continue
        seen[h] = name
        items.append(name)
    if "打call.png" in items:
        items.remove("打call.png")
    items.insert(0, "打call.png")
    return items


def main():
    skip = 0
    if "--from" in sys.argv:
        skip = int(sys.argv[sys.argv.index("--from") + 1])
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
    full = [p for p in packs if len(p) == 16]
    print(f"共 {len(items)} 张 -> {len(full)} 个整包"
          + (f" + 剩余 {len(packs[-1])} 张" if packs[-1] and len(packs[-1]) < 16 else "")
          + f"  [从第 {skip+1} 组开始]")

    for pi, pack in enumerate(packs):
        if pi < skip:
            continue
        tag = f"第{pi+1:02d}包" if len(pack) == 16 else f"剩余{len(pack)}张"
        pdir = os.path.join(OUT, tag)
        os.makedirs(pdir, exist_ok=True)
        print(f"\n== {tag}: {len(pack)} 个 ==")
        t0 = time.time()

        pngs = []
        sr_cache = {}
        cover_src = None
        for i, (name, kw) in enumerate(pack, 1):
            path = os.path.join(SRC, name)
            src = Image.open(path).convert("RGBA")
            bb = content_bbox(src)
            crop = src.crop(bb) if bb else src

            if name not in sr_cache:
                sr_cache[name] = sr_rgba(crop)
            big = sr_cache[name]

            # 缩略图（保留文字）
            thumb = fit_from_big(big, 300)
            tp = os.path.join(TMPDIR, f"p{pi}_{i}.png")
            n, u = save_png_under(thumb, tp, 200 * 1024)
            pngs.append((tp, f"{kw}_{i:02d}.png", n))
            print(f"   {kw}_{i:02d}.png {n/1024:.0f}KB({u}) ", end="")

            # 封面候选：无文字版打call > 能裁掉文字带的图
            if name == "打call.png" and os.path.exists(DCALL):
                cover_src = Image.open(DCALL).convert("RGBA")
                print(f"[封面=打call无文字版]", end="")
            elif cover_src is None:
                cand = strip_top_text(src)
                if cand is not None:
                    cb = content_bbox(cand)
                    cover_src = sr_rgba(cand.crop(cb)) if cb else None
                    if cover_src is not None:
                        print(f"[封面候选={kw}]", end="")
        print(f"\n   超分耗时 {time.time()-t0:.0f}s")

        if cover_src is None:
            # 兜底：打call 原图（可能带文字）+ 顶部裁剪尝试
            base = Image.open(DCALL_FALLBACK).convert("RGBA")
            cand = strip_top_text(base)
            if cand is not None:
                base = cand
                print("   封面: 打call 裁顶部文字带")
            else:
                # 从上部 35% 之后裁一块无文字区域兜底
                bb = content_bbox(base)
                crop = base.crop((bb[0], bb[1] + int((bb[3]-bb[1])*0.35), bb[2], bb[3]))
                base = crop if content_bbox(crop) else base
                print("   封面: 打call 裁掉上部35%兜底（可能仍有残留文字，建议补无文字版）")
            cb = content_bbox(base)
            cover_src = sr_rgba(base.crop(cb)) if cb else sr_rgba(base)
            print("   封面: 兜底打call")
        cp = os.path.join(pdir, "表情封面图.png")
        n, u = save_png_under(fit_from_big(cover_src, 200), cp, 100 * 1024)
        print(f"   表情封面图.png {n/1024:.1f}KB({u})")

        # 打 zip（UTF-8 文件名，直接 writestr 不依赖临时文件存活）
        zpath = os.path.join(pdir, "表情缩略图.zip")
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            for tmp, final, n in pngs:
                with open(tmp, "rb") as f:
                    z.writestr(final, f.read())   # 不删临时文件，避免删除钩子打断
        print(f"   表情缩略图.zip: {len(pngs)} 文件 {os.path.getsize(zpath)//1024}KB")

    print("\nALL DONE")


if __name__ == "__main__":
    if "--pack" in sys.argv:
        idx = sys.argv.index("--pack")
        # 单包调试：--pack 0
        items = collect_items()
        packs = []
        for name in items:
            kw = fix_keyword(keyword_of(name))
            for p in packs:
                if len(p) < 16 and kw not in [x[1] for x in p]:
                    p.append((name, kw)); break
            else:
                packs.append([(name, kw)])
        import types
        # 只跑指定包
        target = int(sys.argv[idx + 1])
        allpacks = packs
        packs.clear(); packs.append(allpacks[target])
        # hack：重新覆盖 pack 循环
        main()
    else:
        main()

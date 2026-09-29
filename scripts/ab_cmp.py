# -*- coding: utf-8 -*-
"""300px 终稿目视 A/B：
A=原生直缩(源图多级降采样到300)  B=超分母版降回300  C=当前完整管线(实际产出)
放大 3x 只裁中间一块，专门看发丝/描边/色边这些"糊不糊"的地方。
"""
import os, sys, io, zipfile, time, warnings
warnings.filterwarnings("ignore")
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pack_sr import SRC, OUT
from polish import denoise_auto, final_sharp, add_outline
from down import down_chain
from pack_final import MASTER, LQ_MAX
from pack_sr import sr_rgba
from matting_gpu import matte

ZOOM = 3
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ab_cmp")
os.makedirs(OUTDIR, exist_ok=True)

# 取第01包里已产出的实际成品作为 C
ZP = os.path.join(OUT, "第01包", "表情缩略图.zip")
成品 = {}
if os.path.exists(ZP):
    with zipfile.ZipFile(ZP) as z:
        for n in z.namelist():
            成品[n[:-4]] = Image.open(io.BytesIO(z.read(n))).convert("RGBA")

NAMES = sys.argv[1:] or ["打call.png", "DNA动了.png", "WOW.png", "充电中.png"]


def crop_zoom(im, z=ZOOM):
    """裁内容中心 1/3 再放大，放大的是最终像素本身。"""
    if im.size[0] < z:
        return im
    w, h = im.size
    bw, bh = w // z, h // z
    if bh < 8:
        return im
    c = im.crop(((w - bw) // 2, (h - bh) // 2, (w - bw) // 2 + bw, (h - bh) // 2 + bh))
    return c.resize((bw * z, bh * z), Image.NEAREST)


rows = []
for nm in NAMES:
    src = Image.open(os.path.join(SRC, nm)).convert("RGBA")
    src = matte(src)
    t = {}
    t0 = time.time()
    t["A 原生直缩"] = down_chain(src, 300)
    tA = time.time() - t0

    t0 = time.time()
    master = sr_rgba(src, lq_max=LQ_MAX)                  # 纯超分，不做打磨
    tB_raw = down_chain(master, 300)
    tB = time.time() - t0

    t0 = time.time()
    t["B 超分母版"] = tB_raw
    t["C 完整管线"] = 成品.get(nm) or tB_raw
    tC = time.time() - t0
    print(f"{nm}: A={tA:.1f}s B={tB:.1f}s C缓存", flush=True)

    for tag, im in t.items():
        c = crop_zoom(im)
        c.save(os.path.join(OUTDIR, f"{nm[:-4]}__{tag.split()[0]}.png"))
    rows.append((nm, t))

# ---------- 拼一张总对比板：行=图片，列=方案，同一行严格对齐同一块裁切 ----------
TAGS = ["A 原生直缩", "B 超分母版", "C 完整管线"]
PAD, LAB = 8, 18
Z = ZOOM
tw, th = 100 * Z, 100 * Z        # 每格固定 100 原像素宽，3x 放大
gw, gh = tw + LAB, th + LAB

sheet = Image.new("RGB", (gw * len(TAGS) + PAD, (gh + PAD) * len(rows) + PAD), (235, 235, 235))
from PIL import ImageDraw
d = ImageDraw.Draw(sheet)
for ci, tag in enumerate(TAGS):
    x0 = PAD + ci * gw
    d.text((x0 + 2, 3), tag, fill=(20, 20, 20))
for ri, (nm, t) in enumerate(rows):
    for ci, tag in enumerate(TAGS):
        im = t[tag]
        c = im.crop((im.size[0] // 2 - 50, im.size[1] // 2 - 50,
                     im.size[0] // 2 + 50, im.size[1] // 2 + 50))
        c = c.resize((tw, th), Image.NEAREST)
        sheet.paste(c, (PAD + ci * gw + LAB, PAD + ri * gh + LAB))
    y = PAD + ri * gh + gh - 11
    d.text((PAD + LAB, y), nm, fill=(60, 60, 60))   # 画在格子上会盖住图，放下方留白

sp = os.path.join(OUTDIR, "AB对比板.png")
sheet.save(sp)
print("板尺寸:", sheet.size, "->", sp)

# 同时输出 1x 全图，看整体观感而不是只有中间块
for ri, (nm, t) in enumerate(rows):
    W = 300 * len(TAGS) + PAD * (len(TAGS) + 1)
    full = Image.new("RGB", (W, 300 + PAD * 2), (235, 235, 235))
    for ci, tag in enumerate(TAGS):
        full.paste(t[tag].resize((300, 300), Image.NEAREST),
                   (PAD + ci * (300 + PAD) + 0, PAD))
    full.crop((0, 0, W, 300 + PAD * 2)).save(
        os.path.join(OUTDIR, f"整图_{nm[:-4]}.png"))

print("OUT:", OUTDIR)
for f in sorted(os.listdir(OUTDIR)):
    print(" ", f)

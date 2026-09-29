# -*- coding: utf-8 -*-
"""高清三通道 A/B/C 抽样对比。

基准 REF = 原生 1900px 直接 box_down 到 300px（= 现有上传/300px 成品同源参考）。
判据：
  保真   A/B/C 各自回 300px 与 REF 的 RMSE（越小 = 信息越没丢）
  锐度   lap_var（越大越锐利，糊了就小）
  噪点   孤立噪点占比（越小越干净，超分伪影会把它顶起来）
  体积   GIF 单张 / 静态 PNG 单张
目视：脸部区域 2x 放大并排 + 三通道全帧动效条。
"""
import os, sys, time, io, shutil
import numpy as np
from scipy.ndimage import median_filter, convolve
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pack_hd as PH
from simple_down import box_down, alpha_fix
from polish import denoise_auto
from pack_motion import frames_to_gif

PICK = ["打call.png", "吃东西.png", "WOW.png", "不慌.png", "加油.png", "思考.png"]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hd_ab")
os.makedirs(OUT, exist_ok=True)
CHS = "ABC"


def lap_var(im):
    g = np.array(im.convert("L"), dtype=np.float64)
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
    return float(convolve(g, k, mode="nearest").var())


def noise_rate(im):
    rgb = np.array(im.convert("RGB")).astype(np.int16)
    m3 = median_filter(rgb, size=3, mode="nearest")
    m5 = median_filter(rgb, size=5, mode="nearest")
    d1 = np.abs(rgb - m3).max(axis=2)
    d2 = np.abs(m3 - m5).max(axis=2)
    return float(np.clip((d1 - d2) * 0.45, 0, 1).mean())


def to300(im):
    """高清帧 -> 300px 画布（与 REF 同构图：box_down 居中 fit）"""
    from pack_motion import extend_colors
    ec = extend_colors(im)
    w0, h0 = ec.size
    s = min((300 - 12) / w0, (300 - 12) / h0)
    nw, nh = max(1, round(w0 * s)), max(1, round(h0 * s))
    sm = ec.resize((nw, nh), Image.BOX)
    cv = Image.new("RGBA", (300, 300), (0, 0, 0, 0))
    cv.paste(alpha_fix(sm), ((300 - nw) // 2, (300 - nh) // 2), sm)
    return cv


rows, boards = [], []
t_all = time.time()
for name in PICK:
    kw = name[:-4]
    p = os.path.join(PH.SRC, name)
    if not os.path.exists(p):
        print("缺", name); continue
    base = PH.fit_native(PH.prep(name))
    ref = to300(PH.to_hd(base))          # 参考：原生直通 300
    print(f"\n### {kw}  base={base.size}", flush=True)
    tf = {}
    for ch in CHS:
        t0 = time.time()
        fr = PH.frames_hd(base, kw, ch)
        hd = fr[0]
        tb = time.time() - t0
        tf[ch] = (fr, hd)
        # 体积
        gp = os.path.join(OUT, f"{kw}_{ch}.gif")
        ng, nf = frames_to_gif(fr, gp, duration=PH.GIF_MS,
                               limit_kb=PH.GIF_LIMIT_KB, colors=64)
        ib = io.BytesIO(); hd.save(ib, "PNG", optimize=True)
        # 客观
        tb300 = to300(hd)
        d = np.asarray(tb300.convert("RGB"), np.float32) - np.asarray(ref.convert("RGB"), np.float32)
        rmse = float(np.sqrt((d * d).mean()))
        rows.append((kw, ch, f"{hd.width}x{hd.height}",
                     rmse, lap_var(hd), noise_rate(hd),
                     ng / 1024, ib.tell() / 1024, tb))
        boards.append((kw, ch, fr, hd))
        print(f"   {ch}: {hd.width}x{hd.height} RMSE300={rmse:.2f} "
              f"lap={lap_var(hd):.1f} noise={noise_rate(hd)*100:.2f}% "
              f"gif={ng/1024:.0f}KB png={ib.tell()/1024:.0f}KB {tb:.1f}s",
              flush=True)
    shutil.copy(os.path.join(OUT, f"{kw}_A.gif"), os.path.join(OUT, f"{kw}_REF.gif"))
    print(f"   参考 REF lap={lap_var(ref):.1f} noise={noise_rate(ref)*100:.2f}%", flush=True)

print("\n===== 汇总（通道 / RMSE↓ / 锐度↑ / 噪点↓ / GIF KB↓ / PNG KB） =====")
hdr = f"{'图':<8}{'通道':<5}{'尺寸':<12}{'RMSE300':>9}{'lap_var':>9}{'noise%':>8}{'GIF KB':>8}{'PNG KB':>8}"
print(hdr)
for r in rows:
    print(f"{r[0]:<8}{r[1]:<5}{r[2]:<12}{r[3]:>9.2f}{r[4]:>9.1f}"
          f"{r[5]*100:>8.2f}{r[6]:>8.0f}{r[7]:>8.0f}")

# ---------- 目视板 ----------
ZW = 430
cells = {}
for kw, ch, fr, hd in boards:
    cells.setdefault(kw, {})[ch] = fr
    cells.setdefault(kw + "##REF", {})[ch] = None
for key in list(cells):
    k2 = key.replace("##REF", "")
    if k2 in cells:
        cells[key] = dict(cells[k2]); cells[key]["REF"] = None

for key in cells:
    d = cells[key]
    if "A" not in d:
        continue
    kw = key.replace("##REF", "")
    face = None
    try:
        from head_motion import detect_face
        bg = Image.new("RGB", d["A"][0].size, (255, 255, 255))
        bg.paste(d["A"][0], mask=d["A"][0].split()[3])
        face = detect_face(d["A"][0])
    except Exception:
        pass
    box = None
    if face:
        x, y, w, h = face
        box = (max(0, x - w), max(0, y - h * 0.5),
               min(d["A"][0].width, x + 2 * w), min(d["A"][0].height, y + h * 1.8))
    if box is None:
        c = d["A"][0]
        box = (int(c.width * 0.2), int(c.height * 0.05),
               int(c.width * 0.85), int(c.height * 0.7))
    orders = ["REF"] + list(CHS)
    ims = []
    for ch in orders:
        if ch not in d or d[ch] is None:
            continue
        f = d[ch][0]
        z = f.crop(box).resize((ZW, ZW), Image.LANCZOS)
        b = Image.new("RGB", (ZW, ZW), (255, 255, 255))
        b.paste(z, mask=z.split()[3])
        ims.append((ch, b))
    lab = 22
    sh = Image.new("RGB", (ZW * len(ims) + 8 * (len(ims) + 1), ZW + lab), (246, 246, 246))
    from PIL import ImageDraw
    dr = ImageDraw.Draw(sh)
    for i, (ch, b) in enumerate(ims):
        x = 8 + i * (ZW + 8)
        sh.paste(b, (x, lab))
        dr.text((x + 4, 4), ch, fill=(20, 20, 20))
    sh.save(os.path.join(OUT, f"view_{kw}.png"))

print(f"\n总耗时 {time.time()-t_all:.0f}s -> {OUT}")

# -*- coding: utf-8 -*-
"""300x300 最终空间方案选型：
A 超分母版降采样 + 锐化  /  B 源图直缩不超分（对照组）  / C 超分母版 + 无锐化
用梯度能量量化清晰度，用局部对比度和过冲率量化"糊/假"。
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
from PIL import Image, ImageFilter, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pack_sr import SRC
from pack_final import matte, sr_polish, down_chain, denoise_auto, final_sharp, add_outline
from polish import superres

MASTER, LQ_MAX, SIZE = 1900, 475, 300
os.makedirs("cmp_sharp", exist_ok=True)


def _g(rgb, a):
    gp = np.pad(rgb, ((1, 0), (1, 0), (0, 0)), mode="edge")
    gy, gx = np.gradient(gp, axis=(0, 1))
    g = np.abs(gx).mean(axis=2) + np.abs(gy).mean(axis=2)
    return g[:a.shape[0], :a.shape[1]]


def metrics(im):
    rgb = np.array(im.convert("RGB")).astype(np.float32)
    a = (np.array(im.split()[3]) > 200)
    if a.sum() < 100:
        return 0, 0, 0
    g = _g(rgb, a)
    sharp = float(g[a].mean())

    # 局部对比度：3x3 窗口内标准差均值 = "贴脸看有没有东西"
    from numpy.lib.stride_tricks import sliding_window_view
    win = sliding_window_view(rgb, (3, 3, 1)).mean(axis=4)      # (H,W,3)
    sd = win.std(axis=(2, 3))
    lctx = float(sd[a[:sd.shape[0], :sd.shape[1]]].mean())

    # 过冲率：边缘处与模糊版的差值过大 = 光晕/白边
    blur = np.array(im.convert("RGB").filter(ImageFilter.GaussianBlur(1.6))).astype(np.float32)
    d = np.abs(rgb - blur).mean(axis=2)
    over = float((d > 26)[a].mean() * 100)
    return sharp, lctx, over


picks = ["打call.png", "充电中.png", "有被笑到.png", "已读乱回.png", "真拿你没办法.png"]
picks = [p for p in picks if os.path.exists(os.path.join(SRC, p))][:4]
print(f"样本 {len(picks)} 张   (梯度 / 局部对比度 / 过冲%)")

for name in picks:
    src = matte(Image.open(os.path.join(SRC, name)).convert("RGBA"))
    t0 = time.time()
    master = sr_polish(src, size=MASTER, ow=0, lq_max=LQ_MAX, denoise=0.0)
    tsr = time.time() - t0

    TAGB = "B  源图直缩(对照)"
    made = {}
    made["A0 超分降采样"] = denoise_auto(down_chain(master, SIZE), .45)
    for t, sh in [("A1 +USM40", 40), ("A2 +USM75", 75), ("A3 +USM110", 110)]:
        made[t] = final_sharp(denoise_auto(down_chain(master, SIZE), .45), sh)
    made[TAGB] = denoise_auto(down_chain(src, SIZE), .45)   # 无超分，纯降采样

    rows = []
    for tag, im in made.items():
        im = add_outline(im, width=2)
        s, l, o = metrics(im)
        rows.append((tag, s, l, o))
    made = {t: add_outline(im, width=2) for t, im in made.items()}

    print(f"\n{name}   超分 {tsr:.1f}s")
    for tag, s, l, o in rows:
        print(f"   {tag:<18} 梯度={s:6.2f}  局部对比={l:5.2f}  过冲={o:5.2f}%"
              + ("   <- 光晕风险" if o > 3.0 else ""))

    made[TAGB].save(f"cmp_sharp/B_{name}")
    made["A2 +USM75"].save(f"cmp_sharp/A2_{name}")

    lb = (60, 30, 240, 190)
    cells = [made[TAGB].crop(lb).resize((360, 360), Image.LANCZOS),
             made["A0 超分降采样"].crop(lb).resize((360, 360), Image.LANCZOS),
             made["A2 +USM75"].crop(lb).resize((360, 360), Image.LANCZOS)]
    board = Image.new("RGB", (360 * 3 + 24, 360), (250, 250, 250))
    for i, (c, t) in enumerate(zip(cells, ["B 对照(无超分)", "A0 超分不锐", "A2 超分+USM75"])):
        board.paste(c, (i * 360 + i * 6, 0))
        ImageDraw.Draw(board).text((i * 360 + 8, 5), t, fill=(0, 0, 0))
    board.save(f"cmp_sharp/{name}.png")
print("\ndone -> cmp_sharp/")

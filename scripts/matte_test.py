# -*- coding: utf-8 -*-
"""去背对比：现状 unwhite() vs 新的 matte()，输出数值指标 + 棋盘格对比图。"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from PIL import Image
from bgtest import unwhite
from matting_gpu import matte, metrics

SRC = r"C:\Users\Admin\OneDrive\Desktop\合格"
OUTD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bgtest", "matte")
os.makedirs(OUTD, exist_ok=True)

picks = ["打call.png", "充电中.png", "笑死.png", "摸鱼.png", "已读乱回.png"]
picks = [p for p in picks if os.path.exists(os.path.join(SRC, p))]

CHECK = Image.new("RGB", (32, 32), (208, 208, 214))


def checker(size):
    c = Image.new("RGB", size, (214, 214, 220))
    w, h = size
    for y in range(0, h, 32):
        for x in range(0, w, 32):
            if (x // 32 + y // 32) % 2:
                c.paste(CHECK, (x, y))
    return c


def frow(d):
    return ("内部白保留 %5.1f%%  外部白去除 %5.1f%%  灰晕残留 %5.1f%%"
            % (d["内部白保留"], d["外部白去除率"], d["灰晕残留"]))


rows = 3
board = Image.new("RGB", (300 * 4 + 60, 300 * len(picks) + 40), (250, 250, 250))

for i, name in enumerate(picks, 1):
    src = Image.open(os.path.join(SRC, name)).convert("RGBA")
    t0 = time.time(); old = unwhite(src); t_old = time.time() - t0
    t0 = time.time(); new = matte(src); t_new = time.time() - t0

    mo, mn = metrics(src, old), metrics(src, new)
    print(name)
    print("  现状 unwhite :", frow(mo), "(%.2fs)" % t_old)
    print("  新   matte   :", frow(mn), "(%.2fs)" % t_new)

    for j, im in enumerate([src, old, new]):
        bg = checker(im.size)
        bg.paste(im, (0, 0), im)
        board.paste(bg, (20 + j * 320, 20 + (i - 1) * 300))
p = os.path.join(OUTD, "compare.png")
board.save(p)
print("saved", p, board.size)

# 放大看边缘：贴图本体的边缘带
z = 2
edge = new.resize((new.width * z, new.height * z), Image.LANCZOS)
edge.save(os.path.join(OUTD, "matte_zoom.png"))
print("saved zoom")

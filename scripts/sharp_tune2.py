# -*- coding: utf-8 -*-
"""关键验证：超分->300 是否真比 原生直缩->300 更清晰；以及锐化/降噪折中搜索。"""
import os, sys, numpy as np, warnings
warnings.filterwarnings("ignore")
from PIL import Image, ImageFilter
from scipy.ndimage import convolve
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pack_sr import SRC, fit_from_big, content_bbox
from polish import sr_polish, final_sharp, denoise_auto, add_outline
from down import down_chain
from matting_gpu import matte

MASTER, LQ_MAX = 1900, 475
os.makedirs("sharp_tune", exist_ok=True)

def lap_var(im):
    a = np.array(im.convert("L"), dtype=np.float64)
    k = np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=np.float64)
    return float(convolve(a, k, mode="nearest").var())

def halo(im):
    rgb = np.array(im.convert("RGB"), dtype=np.float64)
    bl = np.array(Image.fromarray(np.clip(rgb,0,255).astype(np.uint8))
                  .filter(ImageFilter.GaussianBlur(2.0)), dtype=np.float64)
    m = np.array(im.split()[3]) > 200
    return float(np.clip(rgb-bl, 0, None)[m].mean())

picks = ["打call.png", "充电中.png", "看戏.png", "已读乱回.png"]
picks = [p for p in picks if os.path.exists(os.path.join(SRC, p))]

# ---- 实验 1：三条路径对比 ----
print("=== 路径对比（锐70/降噪0.25/thresh2，全图已加描边）===")
print(f"{'图':<12}{'原生直缩':>22}{'超分->300':>22}{'比值':>8}")
for name in picks:
    src = matte(Image.open(os.path.join(SRC, name)).convert("RGBA"))
    master = sr_polish(src, size=MASTER, ow=0, lq_max=LQ_MAX, denoise=0.0)

    def build(im, sh=70, dn=0.25, th=2):
        x = down_chain(im, 300)
        if dn > 0: x = denoise_auto(x, dn)
        if sh: x = final_sharp(x, sh, radius=1.1, thresh=th)
        return add_outline(x, width=2)

    a = build(src)                       # 原生直缩
    b = build(master)                    # 超分后降采样
    print(f"{name:<12}{lap_var(a):>14.0f}{lap_var(b):>19.0f}{lap_var(b)/max(lap_var(a),1):>8.2f}")

# ---- 实验 2：折中搜索 ----
print("\n=== 锐化 x 降噪 折中搜索（超分路径）===")
for name in picks[:2]:
    src = matte(Image.open(os.path.join(SRC, name)).convert("RGBA"))
    master = sr_polish(src, size=MASTER, ow=0, lq_max=LQ_MAX, denoise=0.0)
    print(f"\n-- {name} --")
    print(f"{'参数':<26}{'锐度':>10}{'光晕':>8}")
    for tag, sh, dn, th in [("锐70/噪.25/th2",70,.25,2), ("锐70/噪.25/th3",70,.25,3),
                            ("锐85/噪.25/th3",85,.25,3), ("锐70/噪.15/th2",70,.15,2),
                            ("锐85/噪.15/th2",85,.15,2), ("锐70/噪0/th2",70,0,2),
                            ("锐100/噪0/th3",100,0,3)]:
        x = down_chain(master, 300)
        if dn>0: x = denoise_auto(x, dn)
        if sh:   x = final_sharp(x, sh, radius=1.1, thresh=th)
        x = add_outline(x, width=2)
        print(f"  {tag:<24}{lap_var(x):>10.0f}{halo(x):>8.2f}")

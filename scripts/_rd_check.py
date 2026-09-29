# -*- coding: utf-8 -*-
import os, sys, glob
import numpy as np
from PIL import Image
from scipy.ndimage import convolve
sys.path.insert(0, '.')
from redraw3 import prep, SEEDS

def lap(im):
    g = np.array(im.convert("L"), np.float64)
    k = np.array([[0,1,0],[1,-4,1],[0,1,0]], np.float64)
    return float(convolve(g, k, mode="nearest").var())

def ssim_quick(a, b):
    a = np.array(a.convert("L"), np.float64); b = np.array(b.convert("L"), np.float64)
    mu_a, mu_b = a.mean(), b.mean()
    va, vb = a.var(), b.var(); cov = ((a-mu_a)*(b-mu_b)).mean()
    c1, c2 = (0.01*255)**2, (0.03*255)**2
    return float(((2*mu_a*mu_b+c1)*(2*cov+c2))/((mu_a**2+mu_b**2+c1)*(va+vb+c2)))

for kw in ["打call", "吃东西", "加油"]:
    init, alpha, size = prep(kw + ".png")
    print(f"\n[{kw}] 输入 {init.size} alpha{alpha.size}  清晰度 lap={lap(init):.1f}")
    for sd in SEEDS:
        p = f"redraw3/{kw}_s{sd}.png"
        if not os.path.exists(p): print("  缺", p); continue
        im = Image.open(p)
        print(f"  s{sd}: {im.size} {'8对齐OK' if im.width%8==0 and im.height%8==0 else '8未对齐!'} "
              f"lap={lap(im):.1f} SSIM={ssim_quick(init, im):.3f} 色偏={abs(np.array(im.convert('RGB'),np.float32).mean()-np.array(init,np.float32).mean()):.1f}")

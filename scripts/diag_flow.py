# -*- coding: utf-8 -*-
"""
光流场客观诊断：证明 SVD 产生的是非刚体动作，而不是"整体放大缩小"。

输出三张图：
  1. 位移场热图（区域化：头/手/身/底，看位移是否均匀）
  2. 网格变形图（原始网格 -> 位移后网格，肉眼看出局部扭曲）
  3. 反前后 alpha 差异（证明动作真的改变了画面结构）
"""
import os
import sys
import numpy as np
import cv2
from PIL import Image, ImageDraw

sys.path.insert(0, '.')
svd2 = __import__('svd2')
OUT = 'diag_out'
os.makedirs(OUT, exist_ok=True)


def region_map(h, w):
    """把画布分区：上1/3=头，中1/3=身，下1/3=底；再切左右两半分手。"""
    y, x = np.mgrid[0:h, 0:w]
    up = y < h * 0.34
    mid = (y >= h * 0.34) & (y < h * 0.72)
    low = y >= h * 0.72
    left = x < w * 0.5
    m = {}
    m['头'] = up
    m['左手'] = up & left
    m['右手'] = up & ~left
    m['身'] = mid
    m['底'] = low
    return m


def flow_of(g0, g1):
    w2, h2 = g0.shape[1] // 2, g0.shape[0] // 2
    g0s = cv2.resize(g0, (w2, h2), interpolation=cv2.INTER_AREA)
    g1s = cv2.resize(g1, (w2, h2), interpolation=cv2.INTER_AREA)
    f = cv2.calcOpticalFlowFarneback(g0s, g1s, None, 0.5, 3, 21, 3, 5, 1.2, 0)
    f = cv2.GaussianBlur(f, (0, 0), 1.2)
    f = cv2.resize(f, (g0.shape[1], g0.shape[0]), interpolation=cv2.INTER_LINEAR) * 2
    return f


def draw_mesh(img, flow, step=64, scale=1.0):
    """把规则网格按光流扭曲后画出来。扭曲量越大线越偏。"""
    im = np.asarray(img).copy()
    h, w = flow.shape[:2]
    yy = np.arange(step, h, step)
    xx = np.arange(step, w, step)
    for y in yy:
        for x in xx:
            p = (int(x), int(y))
            q = (int(x + flow[y, x, 0] * scale), int(y + flow[y, x, 1] * scale))
            cv2.line(im, p, q, (255, 60, 0), 2)
    for x in xx:
        for y in yy:
            p = (int(x), int(y))
            q = (int(x + flow[y, x, 0] * scale), int(y + flow[y, x, 1] * scale))
            cv2.line(im, p, q, (0, 120, 255), 2)
    return Image.fromarray(im)


for kw in ['打call']:
    cache = os.path.join('svd_out', f'{kw}_svdframes.npz')
    if not os.path.exists(cache):
        continue
    fr = list(np.load(cache)['f'])
    rgb0, _, bbox = svd2.layout(
        Image.open(os.path.join(svd2.SRC, kw + '.png')).convert('RGBA'))
    W, H = 576, 1024
    g0 = cv2.cvtColor(svd2._fit(fr[0], (W, H)), cv2.COLOR_RGB2GRAY)

    rows = []
    for idx in [4, 12, 24]:
        g1 = cv2.cvtColor(svd2._fit(fr[idx], (W, H)), cv2.COLOR_RGB2GRAY)
        f = flow_of(g0, g1)
        mag = np.sqrt(f[..., 0] ** 2 + f[..., 1] ** 2)

        # ---- 分区位移统计 ----
        rm = region_map(H, W)
        x0, y0, x1, y1 = bbox
        stat = []
        for name, mask in rm.items():
            sel = mask & np.zeros_like(mag, dtype=bool)
            sel[:, :] = False
            sel[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
            stat.append((name, float(mag[sel].mean()), float(mag[sel].max())))

        # ---- 与"纯缩放"的残差：拟合最佳仿射后看残余 ----
        ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
        src_pts = np.stack([xs, ys], -1).reshape(-1, 2)
        dst_pts = (src_pts + f.reshape(-1, 2)).astype(np.float32)
        m = np.zeros_like(mag, dtype=bool)
        m[y0:y1, x0:x1] = True
        sp = src_pts[m.reshape(-1).size // 2].reshape(-1, 2) if False else src_pts
        sel = m.reshape(-1)
        sp, dp = src_pts[sel], dst_pts[sel]
        A = np.zeros((2 * len(sp), 6), dtype=np.float32)
        A[0::2, 0] = sp[:, 0]; A[0::2, 1] = sp[:, 1]; A[0::2, 2] = 1
        A[1::2, 3] = sp[:, 0]; A[1::2, 4] = sp[:, 1]; A[1::2, 5] = 1
        b = np.zeros((2 * len(sp),), dtype=np.float32)
        b[0::2] = dp[:, 0]; b[1::2] = dp[:, 1]
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        proj = np.stack([sol[0] * sp[:, 0] + sol[1] * sp[:, 1] + sol[2],
                         sol[3] * sp[:, 0] + sol[4] * sp[:, 1] + sol[5]], 1)
        resid = float(np.sqrt(((dp - proj) ** 2).sum(1).mean()))

        rows.append((idx, stat, resid, f))

        print(f'--- {kw} 第{idx}帧 ---')
        for n, mm, mx in stat:
            print(f'   {n:4s} 平均位移 {mm:6.2f}px  峰值 {mx:6.2f}px')
        print(f'   与最佳仿射(缩放/平移)的残差: {resid:.2f}px')

    # ---- 拼版：启动帧 / 网格变形 / 位移热图 ----
    C = 300
    im0 = Image.fromarray(svd2._fit(fr[0], (W, H))).resize((C, int(C * H / W)))
    n = len(rows)
    sh = Image.new('RGB', (C * (1 + n * 2), int(C * H / W)), (250, 250, 250))
    sh.paste(im0, (0, 0))
    for i, (idx, stat, resid, f) in enumerate(rows):
        fv = (f[:, :, 0] * 0.5 + 255).astype(np.uint8)
        fvv = cv2.applyColorMap(fv, cv2.COLORMAP_JET)
        fvv = cv2.resize(fvv, (C, int(C * H / W)), interpolation=cv2.INTER_NEAREST)
        sh.paste(Image.fromarray(fvv), (C * (1 + i * 2), 0))
        mesh = draw_mesh(Image.fromarray(svd2._fit(fr[idx], (W, H))), f, step=64, scale=1.0)
        mesh = mesh.resize((C, int(C * H / W)), Image.NEAREST)
        sh.paste(mesh, (C * (2 + i * 2), 0))
    sh.save(os.path.join(OUT, f'_光流诊断_{kw}.png'))
    print('saved', os.path.join(OUT, f'_光流诊断_{kw}.png'))

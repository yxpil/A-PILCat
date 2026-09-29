# -*- coding: utf-8 -*-
"""
表情动效引擎：连续位移场 + 反向采样。

设计要点
--------
1. 位移场 u(x,y) 是**连续函数**，不是网格块，所以没有接缝，采样天然平滑。
2. 反向采样：dst(x,y) = src(x + ux, y + uy)，用 map_coordinates(order=1 双线性)。
   order=1 无振铃、边缘最锐，在 300px 尺度上比 order=3 更适合线稿。
3. 变形是重建过程，插值必然"软"一点，所以形变后要补锐化（见 pack_motion.py）。
4. 透明通道跟着走，绝不丢背景。

动作模型（都是"倒立摆"语义：越靠上摆幅越大，符合大头贴纸）
------------------------------------------------------------
nod    点头：头部上下 + 颈部压缩
sway   摆头：头部左右大幅，身体几乎不动
shake  抖动：高频左右 + 微旋转，用于"无语/生气"
bounce 弹跳：整体缩放脉冲 + 落地压缩
breathe 呼吸：极缓的整体缩放
zoom   凑近：头部放大 + 轻微前倾
tilt   歪头：整体微旋转 + 头部剪切
"""

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates

_EYE_L = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


# ---------------------------------------------------------------- 眼睛定位
def find_eyes(rgba, bbox):
    """在头部区域找左右两个暗斑（眼睛）。失败返回 None。

    表情包都是大头正面/半侧面，眼睛是最显著的两个暗团，用暗度加权重心足够稳。
    只要求：找出两个 x 相距够远的暗区中心，不追求解剖学精确。
    """
    x0, y0, x1, y1 = bbox
    crop = rgba.crop((x0, y0, x1, y1))
    a = np.asarray(crop.split()[3], dtype=np.float32) / 255.0
    if a.max() < 0.3:
        return None
    rgb = np.asarray(crop.convert("RGB"), dtype=np.float32)
    lum = rgb @ _EYE_L
    head_h = max(crop.height, 1)

    # 头部上 1/2 找眼睛（避开嘴/头发）
    hy = int(head_h * 0.50)
    if hy < 8:
        return None
    sub_lum = lum[:hy]
    sub_al = a[:hy]
    # 暗 + 不透明 = 眼睛
    w = np.clip(1.0 - sub_lum / 255.0, 0, 1) ** 3 * sub_al
    if w.sum() < 20:
        return None

    # 左半 / 右半 各取一次暗度加权重心
    hw_, w_ = crop.width, float(hy)
    cy_rng = np.arange(hy, dtype=np.float32)
    out = []
    for start in (0, crop.width // 2):
        hw = w[:, start:start + crop.width // 2]
        if hw.sum() < 3.0:
            return None
        col_rng = np.arange(start, start + crop.width // 2, dtype=np.float32)
        cx = float((hw.sum(axis=0) * col_rng).sum() / hw.sum())
        cy = float((hw.sum(axis=1) * cy_rng).sum() / max(hw.sum(), 1e-6))
        out.append((cx, cy))
    (lx, ly), (rx, ry) = out
    if abs(rx - lx) < crop.width * 0.10:      # 两眼太近，多半是误判
        return None
    return (x0 + (lx + rx) * 0.5, y0 + ly, x0 + lx, y0 + rx, x0, y0 + ry)


# ---------------------------------------------------------------- 位移场
def _body_pivot(bbox):
    """颈部枢轴：主体底部往上 22% 处。头部绕它摆，身体几乎不动。"""
    x0, y0, x1, y1 = bbox
    return (x0 + x1) * 0.5, y0 + (y1 - y0) * 0.78


def warp(rgba, kind, phase, amp=1.0):
    """返回形变后的 RGBA。phase 取 0..1 描述动作进度。"""
    src = np.asarray(rgba, dtype=np.float32)
    H, W = src.shape[:2]
    bbox = _subject_bbox(rgba)
    if bbox is None:
        return rgba
    px, py = _body_pivot(bbox)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    ux = np.zeros_like(xx)
    uy = np.zeros_like(yy)
    bx0, by0, bx1, by1 = bbox
    bw, bh = max(bx1 - bx0, 1), max(by1 - by0, 1)

    # 到枢轴的高度归一化量：头(小)~1，身体(大)~0
    hgt = np.clip((py - yy) / bh, 0.0, 1.4)
    # 距枢轴的横向归一化量
    wid = (xx - px) / max(bw, 1)
    fall = np.clip(hgt, 0.0, 1.0) ** 1.6      # 摆动衰减：越往下越小

    if kind == "sway":
        s = np.sin(phase * 2 * np.pi) * amp
        ux += s * fall * bh * 0.085
        uy += -abs(s) * fall * bh * 0.012
        ux += -s * np.clip(wid, -1, 1) * fall * bh * 0.028      # 顺带一个剪切

    elif kind == "shake":
        s = np.sin(phase * 2 * np.pi * 3.0) * amp
        ux += s * fall * bh * 0.055
        uy += -abs(s) * fall * bh * 0.020
        ux += np.sin(phase * 2 * np.pi * 3.0 + 1.0) * fall * bh * 0.012

    elif kind == "nod":
        s = np.sin(phase * 2 * np.pi) * amp
        uy += s * fall * bh * 0.055
        ux += s * fall * bw * 0.010
        uy += -s * np.clip(hgt, 0, 1) ** 2 * bh * 0.018

    elif kind == "bounce":
        k = np.sin(phase * 2 * np.pi) * amp
        sc = 1.0 + k * 0.055
        ux += (xx - px) * (sc - 1.0)
        uy += (yy - py) * (sc - 1.0) * 1.35          # 竖向更夸张
        uy += np.maximum(0.0, -k) * bh * 0.030 * np.clip(hgt * 1.4, 0, 1)

    elif kind == "breathe":
        k = np.sin(phase * 2 * np.pi) * amp
        sc = 1.0 + k * 0.022
        ux += (xx - px) * (sc - 1.0)
        uy += (yy - py) * (sc - 1.0)

    elif kind == "zoom":
        k = np.sin(phase * 2 * np.pi) * amp
        sc = 1.0 + k * 0.075
        ux += (xx - px) * (sc - 1.0)
        uy += (yy - py) * (sc - 1.0) * 1.2
        uy += -k * fall * bh * 0.045

    elif kind == "tilt":
        s = np.sin(phase * 2 * np.pi) * amp
        ux += s * np.clip(wid, -1, 1) * fall * bh * 0.055
        uy += -abs(s) * fall * bh * 0.010
        ux += s * fall * bh * 0.018

    # ---- 眨眼：眼部区域竖向压扁（叠加在摆动之上） ----
    if kind != "breathe":
        eyes = find_eyes(rgba, bbox)
        if eyes is not None:
            ex, _ey, _lx, _rx, _ly, _ry = eyes
            # 动作进度前 1/4 段完成一次眨眼
            blink = 1.0 if (phase > 0.05 and phase < 0.22) else 0.0
            if blink > 0.01:
                half_w = max(int(bw * 0.26), 8)
                dx = np.abs(xx - ex)
                wgt = np.clip(1.0 - dx / half_w, 0.0, 1.0) ** 1.5
                dy = np.clip(1.0 - np.abs(yy - (_ly + _ry) * 0.5) / (bh * 0.16), 0, 1)
                uy += blink * (wgt * dy + wgt * dy * 0.0) * bh * 0.045

    # ---- 反向采样 ----
    coord_y = np.clip(yy + uy, 0, H - 1).astype(np.float32)
    coord_x = np.clip(xx + ux, 0, W - 1).astype(np.float32)
    out = np.empty_like(src)
    for c in range(src.shape[2]):
        out[..., c] = map_coordinates(src[..., c], [coord_y, coord_x],
                                      order=1, mode="nearest")
    return Image.fromarray(out.astype(np.uint8), "RGBA")


def _half_w_fix(v):
    return float(max(v, 1e-3))


def _subject_bbox(rgba):
    """主体包围盒，含少量padding，保证摆动时边缘不出空。"""
    a = np.asarray(rgba.split()[3], dtype=np.float32)
    if a.max() < 0.2:
        return None
    rows = np.where(a.max(axis=1) > 0.5)[0]
    cols = np.where(a.max(axis=0) > 0.5)[0]
    if len(rows) < 10 or len(cols) < 10:
        return None
    y0, y1 = rows[0], rows[-1]
    x0, x1 = cols[0], cols[-1]
    p = 4
    H, W = rgba.size[1], rgba.size[0]
    return (max(0, x0 - p), max(0, y0 - p), min(W, x1 + 1 + p), min(H, y1 + 1 + p))

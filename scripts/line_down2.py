# -*- coding: utf-8 -*-
"""线稿专用降采样 v3（细线连续性保障）：

事实：1900->300 约 6.5x，源图 2px 细线在 300px 只占 0.3px。
纯面积平均在物理上没错（等价于原生 0.3px 线的抗锯齿渲染），
但画师在 300px 原生作画时，细线至少画 1px 且是实色的——
所以"原生感"的正确目标不是物理保真，而是：
    细笔触所在的块 -> 画出【连续的、笔触真实颜色的、至少 ~1px】的线
    其余块        -> 保持面积平均（色准/形状完全不动）

每块判定：
    ink_peak  = 块内最暗(经 min 预滤波) —— 有没有笔触经过
    ink_mean  = 块内加权平均墨
    is_line   = ink_peak 高 且 ink_peak-ink_mean 大（细线特征：峰值高但被平均稀释）
    命中      -> alpha 提到 floor，颜色取块内笔触真实色（最暗点邻域均值）
"""
import numpy as np
from PIL import Image
from scipy.ndimage import minimum_filter, uniform_filter

E = np.float32


def _bounds(N, n):
    b = [int(round(i * N / float(n))) for i in range(n + 1)]
    for i in range(1, n + 1):
        if b[i] <= b[i - 1]:
            b[i] = b[i - 1] + 1
    b[n] = min(b[n], N)
    return b


def down_line(rgba, size, floor=0.62, ink_thr=0.55, gap_thr=0.25, margin=6):
    """floor: 细线保底 alpha；ink_thr: 判定笔触的墨量下限；gap_thr: 峰均差下限"""
    from pack_sr import fit_from_big
    w0, h0 = rgba.size
    ratio = max(w0, h0) / float(size)
    a = np.asarray(rgba).astype(E) / 255.0
    rgb, al = a[..., :3], a[..., 3]
    pre = max(3, int(round(ratio)) * 2 + 1)

    lum = rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=E)
    lum_min = minimum_filter(lum, size=pre, mode="nearest")

    for n, axis in ((min(size, lum.shape[0]), 0), (min(size, lum.shape[1]), 1)):
        N = lum.shape[axis]
        b = _bounds(N, n)
        cnt = np.diff(np.array(b, dtype=E)).astype(E)
        shp = [n if i == axis else 1 for i in range(2)]
        shp3 = [n if i == axis else 1 for i in range(3)]

        sum_al = np.add.reduceat(al, b[:-1], axis=axis)
        sum_col = np.add.reduceat(rgb * al[..., None], b[:-1], axis=axis)
        al_mean = sum_al / cnt.reshape(shp)
        col_mean = sum_col / np.maximum(sum_al, 1e-4)[..., None]

        ink_peak = 1.0 - np.minimum.reduceat(lum_min, b[:-1], axis=axis)
        ink_mean = 1.0 - (np.add.reduceat(lum * al, b[:-1], axis=axis)
                          / np.maximum(sum_al, 1e-4))

        # 块内笔触真实色：暗部加权重心（权重 (1-lum)^3 * al，天然聚焦笔触色，无需索引）
        w = ((1.0 - lum) ** 3) * al
        sum_w = np.add.reduceat(w, b[:-1], axis=axis)
        stroke_rgb = np.add.reduceat(rgb * w[..., None], b[:-1], axis=axis) \
            / np.maximum(sum_w, 1e-4)[..., None]

        is_line = (ink_peak > ink_thr) & ((ink_peak - ink_mean) > gap_thr) & (al_mean > 0.01)
        al_out = np.where(is_line, np.maximum(al_mean, floor), al_mean)
        rgb_out = np.where(is_line[..., None], stroke_rgb, col_mean)

        lum_new = rgb_out @ np.array([0.2126, 0.7152, 0.0722], dtype=E)
        al, rgb, lum = al_out, rgb_out, lum_new
        lum_min = np.minimum.reduceat(lum_min, b[:-1], axis=axis)  # 继承到下一轴

    out = Image.fromarray(np.clip(np.dstack([rgb, al]) * 255.0, 0, 255).astype(np.uint8))
    return fit_from_big(out, size, margin=margin)

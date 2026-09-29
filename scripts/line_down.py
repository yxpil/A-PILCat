# -*- coding: utf-8 -*-
"""线稿专用降采样：保笔触(min/暗部)+保结构(max/alpha)，浮点分块一步直达目标尺寸。

为什么需要：1900->300 是 ~6.5x 降采样。一条 2px 宽的线在 300px 里只占 0.3px，
纯面积平均(mean)会把它冲淡成一片灰，且落点随亚像素相位漂移 -> 线看起来断、脏、不流畅。

    color  = 面积平均（预乘 alpha，保色准、透明区不渗色）
    lum    = 预做 min 滤波后按块取暗 -> 细笔触不被冲掉，线保持连续
    alpha  = 预做 max 滤波后按块取亮 -> 发丝/描边等细结构不被抹掉
    实现    = np.ufunc.reduceat 沿两轴做浮点分块，每个源像素恰好归入一个目标像素，
              无"先 324 再 fit 300"的二次缩放损耗（实测二次缩放会把保住的笔触磨掉）。
"""
import numpy as np
from PIL import Image
from scipy.ndimage import minimum_filter, maximum_filter

E = np.float32


def _bounds(N, n):
    """N 个源像素划到 n 个目标块的边界（浮点比例，块大小在 floor/ceil 间浮动）。"""
    b = [int(round(i * N / float(n))) for i in range(n + 1)]
    for i in range(1, n + 1):
        if b[i] <= b[i - 1]:
            b[i] = b[i - 1] + 1
    b[n] = min(b[n], N)
    return b


def down_line(rgba, size, stroke=0.35, margin=6):
    """线稿降采样到 size（保持内容比例 + margin）。stroke: 0=纯面积平均 1=纯暗部保笔触"""
    from pack_sr import fit_from_big
    w0, h0 = rgba.size
    ratio = max(w0, h0) / float(size)
    a = np.asarray(rgba).astype(E) / 255.0
    rgb, al = a[..., :3], a[..., 3]
    pre = max(3, int(round(ratio)) * 2 + 1)           # 预滤波核随降采样倍数走

    # ---- 预滤波：把细笔触"加粗"成块内一定撞得上的暗团 / 细结构摸得着的亮团 ----
    lum = rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=E)
    lum_min = minimum_filter(lum, size=pre, mode="nearest")
    al_max = maximum_filter(al, size=pre, mode="nearest")

    lum_mix = lum * (1.0 - stroke) + lum_min * stroke
    rgb_mix = np.clip(rgb * (lum_mix / np.maximum(lum, 1e-4))[..., None], 0, 1)
    premult = np.concatenate([rgb_mix * al[..., None], al[..., None]], axis=2)

    def _reduce(arr, n, axis, which):
        """which='mean' 加权平均；'max' 逐块最大。返回 (结果, 边界)"""
        N = arr.shape[axis]
        b = _bounds(N, n)
        if which == "mean":
            cnt = np.diff(np.array(b, dtype=E)).astype(E)
            shp = [-1 if i == axis else 1 for i in range(arr.ndim)]
            r = np.add.reduceat(arr, b[:-1], axis=axis) / cnt.reshape(shp)
        else:
            r = np.maximum.reduceat(arr, b[:-1], axis=axis)
        return r, b

    H, W = premult.shape[:2]
    nh, nw = min(size, H), min(size, W)
    m1, _ = _reduce(premult, nh, 0, "mean")
    x1, _ = _reduce(al_max, nh, 0, "max")
    row = np.concatenate([m1[..., :3], np.maximum(m1[..., 3:], x1[..., None])], axis=-1)
    m2, _ = _reduce(row, nw, 1, "mean")
    x2, _ = _reduce(al_max, nw, 1, "max")

    out_a = np.clip(m2[..., 3], 0, 1)
    out_rgb = np.clip(np.where(out_a[..., None] > 1e-4,
                               m2[..., :3] / np.maximum(out_a, 1e-4)[..., None], 0.0), 0, 1)
    out = Image.fromarray(np.clip(np.dstack([out_rgb, out_a]) * 255.0, 0, 255).astype(np.uint8))
    return fit_from_big(out, size, margin=margin)

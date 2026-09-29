# -*- coding: utf-8 -*-
"""简单降采样定稿：8像素合一(BOX) + 轻微透明通道补齐。

替代 line_down2.down_line（墨量模型 v3）——A/B 细节板证实 BOX 方案
线条干净自然，v3 在直接原图输入下回染出黑噪块（ab_simple/_细节对比.png）。
"""
import numpy as np
from PIL import Image


def box_down(rgba, size=300, margin=6):
    """8像素合一：extend_colors 后一步 Image.BOX 面积平均，等比 fit 画布。"""
    from pack_motion import extend_colors
    ec = extend_colors(rgba)
    w0, h0 = ec.size
    s = min((size - margin * 2) / w0, (size - margin * 2) / h0)
    nw, nh = max(1, round(w0 * s)), max(1, round(h0 * s))
    small = ec.resize((nw, nh), Image.BOX)
    canvas = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    canvas.paste(small, ((size - nw) // 2, (size - nh) // 2), small)
    return canvas


def alpha_fix(rgba, gamma=0.82, faint=0.03):
    """轻微透明通道补齐：半透明区 gamma 提实，孤立淡点清除。"""
    a = np.asarray(rgba).astype(np.float32)
    al = a[..., 3] / 255.0
    al = np.where(al > 0.01, al ** gamma, 0.0)
    al[al < faint] = 0.0
    a[..., 3] = np.clip(al * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(a.astype(np.uint8), 'RGBA')

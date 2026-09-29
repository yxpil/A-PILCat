# -*- coding: utf-8 -*-
"""简单降采样 A/B：用户的建议（原图裁切 / 8像素合一） vs 现行墨量模型 v3。

四列：
  A v3墨量   —— 现行管线（down_line，细笔触回染）
  B 8合一    —— extend_colors 后一步 Image.BOX（面积平均=像素合一）
  C 8合一+补alpha —— B 基础上轻微 alpha 补齐（半透明提实、孤立淡点清除）
  D 原图裁切 —— 主体 bbox 紧裁 → BOX 填满 300（cover，允许轻微出边）
"""
import os, sys
import numpy as np
from PIL import Image

sys.path.insert(0, '.')
from pack_sr import SRC, content_bbox
from line_down2 import down_line
from pack_motion import extend_colors
from pack_final import to_final, GIF_SIZE, FINAL_SHARPEN, OUTLINE

NAMES = ['打call', 'WOW', '不慌', '吃东西', '加油', '听我说']
S = 300
MARGIN = 6


def box_down(rgba, size=S, margin=MARGIN):
    """8像素合一：extend_colors 后一步 BOX 面积平均，等比 fit 到画布。"""
    ec = extend_colors(rgba)
    w0, h0 = ec.size
    s = min((size - margin * 2) / w0, (size - margin * 2) / h0)
    nw, nh = max(1, round(w0 * s)), max(1, round(h0 * s))
    small = ec.resize((nw, nh), Image.BOX)
    canvas = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    canvas.paste(small, ((size - nw) // 2, (size - nh) // 2), small)
    return canvas


def alpha_fix(rgba, gamma=0.82, faint=0.03):
    """轻微透明通道补齐：半透明区提实（gamma<1 提升中段），孤立淡点清除。"""
    a = np.asarray(rgba).astype(np.float32)
    al = a[..., 3] / 255.0
    al = np.where(al > 0.01, al ** gamma, 0.0)
    al[al < faint] = 0.0
    a[..., 3] = np.clip(al * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(a.astype(np.uint8), 'RGBA')


def crop_fill(rgba, size=S):
    """原图裁切：主体 bbox 紧裁，cover 填满 300。"""
    bb = content_bbox(rgba, thresh=24)
    if bb is None:
        return box_down(rgba, size, margin=0)
    crop = rgba.crop(bb)
    ec = extend_colors(crop)
    s = size / max(ec.size)
    nw, nh = round(ec.width * s), round(ec.height * s)
    small = ec.resize((nw, nh), Image.BOX)
    canvas = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    canvas.paste(small, ((size - nw) // 2, (size - nh) // 2), small)
    return canvas


def v3_down(rgba, size=S):
    """现行管线：down_line + 锐化 + 描边。"""
    return to_final(rgba, size, FINAL_SHARPEN, denoise=0.2, ow=OUTLINE)


def white_bg(im):
    bg = Image.new('RGB', im.size, (255, 255, 255))
    bg.paste(im, mask=im.split()[3])
    return bg


if __name__ == '__main__':
    os.makedirs('ab_simple', exist_ok=True)
    # ---- 全图对比板 ----
    CELL, PAD, LBL = 300, 8, 22
    W = 4 * (CELL + PAD) + PAD
    H = len(NAMES) * (CELL + LBL + PAD) + LBL + 6
    board = Image.new('RGB', (W, H), (244, 244, 244))
    from PIL import ImageDraw
    d = ImageDraw.Draw(board)
    heads = ['A v3墨量(现行)', 'B 8合一BOX', 'C 8合一+补alpha', 'D 原图裁切']
    for j, h in enumerate(heads):
        d.text((PAD + j * (CELL + PAD) + 90, 4), h, fill=(150, 0, 0))
    for i, nm in enumerate(NAMES):
        src = Image.open(os.path.join(SRC, nm + '.png')).convert('RGBA')
        cols = [v3_down(src), box_down(src), alpha_fix(box_down(src)), crop_fill(src)]
        y = LBL + i * (CELL + LBL + PAD)
        d.text((PAD, y + 2), nm, fill=(0, 0, 0))
        for j, c in enumerate(cols):
            board.paste(white_bg(c), (PAD + j * (CELL + PAD), y + LBL))
    board.save('ab_simple/_全图对比.png')
    print('全图板 ok', board.size)

    # ---- 细节对比板：中心 130px 放大 2.6x ----
    Z, RC = 2.6, 130
    CELL2 = int(RC * Z)
    W2 = 4 * (CELL2 + PAD) + PAD
    H2 = len(NAMES) * (CELL2 + LBL + PAD) + LBL + 6
    bd2 = Image.new('RGB', (W2, H2), (244, 244, 244))
    d2 = ImageDraw.Draw(bd2)
    for j, h in enumerate(heads):
        d2.text((PAD + j * (CELL2 + PAD) + 40, 4), h, fill=(150, 0, 0))
    for i, nm in enumerate(NAMES):
        src = Image.open(os.path.join(SRC, nm + '.png')).convert('RGBA')
        cols = [v3_down(src), box_down(src), alpha_fix(box_down(src)), crop_fill(src)]
        y = LBL + i * (CELL2 + LBL + PAD)
        d2.text((PAD, y + 2), nm, fill=(0, 0, 0))
        for j, c in enumerate(cols):
            cx = (S - RC) // 2
            crop = white_bg(c).crop((cx, cx, cx + RC, cx + RC))
            crop = crop.resize((CELL2, CELL2), Image.NEAREST)
            bd2.paste(crop, (PAD + j * (CELL2 + PAD), y + LBL))
    bd2.save('ab_simple/_细节对比.png')
    print('细节板 ok', bd2.size)

# -*- coding: utf-8 -*-
"""多级降采样：一步大比例缩放在低分辨率端会出锯齿/振铃，逐级靠近目标更干净。"""
from PIL import Image
from pack_sr import fit_from_big


def down_chain(rgba, size, margin=6):
    w, h = rgba.size
    while max(w, h) > size * 2:
        w, h = max(size // 2, w // 2), max(size // 2, h // 2)
        rgba = rgba.resize((w, h), Image.LANCZOS)
    return fit_from_big(rgba, size, margin=margin)

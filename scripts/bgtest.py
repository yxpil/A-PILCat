# -*- coding: utf-8 -*-
"""白底去背（仅清除与画布边缘连通的白色，保护人像内部白色）+ 顶部文字带裁剪验证"""
import os, sys
import numpy as np
from PIL import Image
from scipy.ndimage import label

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pack_sr import SRC, content_bbox, strip_top_text

OUTD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bgtest")
os.makedirs(OUTD, exist_ok=True)


def unwhite(im, white_thr=242):
    """与边缘连通的近白像素 -> 透明。人像内部白色不受影响。"""
    a = np.array(im.convert("RGBA"))
    r, g, b, al = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    white = (r >= white_thr) & (g >= white_thr) & (b >= white_thr) & (al > 60)
    lab, _ = label(white)
    border = set(lab[0, :]) | set(lab[-1, :]) | set(lab[:, 0]) | set(lab[:, -1])
    border.discard(0)
    if not border:
        return im
    bg = np.isin(lab, list(border))
    out = a.copy()
    out[bg, 3] = 0
    return Image.fromarray(out, "RGBA")


def main():
    for name in ["充电中.png", "比心.png", "闪人.png", "笑死.png"]:
        p = os.path.join(SRC, name)
        if not os.path.exists(p):
            print("缺", name); continue
        src = Image.open(p).convert("RGBA")
        cut = unwhite(src)
        cut.save(os.path.join(OUTD, f"cut_{name}"))

        # 裁内容 bbox 再裁文字带
        bb = content_bbox(cut)
        tight = cut.crop(bb) if bb else cut
        st = strip_top_text(tight)
        final = st if st is not None else tight
        how = "裁文字带" if st is not None else "未裁(保留完整)"
        final.save(os.path.join(OUTD, f"final_{name}"))
        print(f"{name}: 内容bbox后 {tight.size}, {how}, 最终 {final.size}")


if __name__ == "__main__":
    main()

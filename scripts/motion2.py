# -*- coding: utf-8 -*-
"""
动效引擎 v2 —— 在 motion.py（整图位移）之上补**局部器官动作**。

motion.py 的 7 种动作全是"倒立摆"整体位移， amp 上限 0.055*BH，
看起来就是一张贴图在平移缩放 —— 用户反馈"没有意义"，根子在这里。

v2 增补：
  * talk / eat / surprised —— 嘴部真实开合（张口时填充口腔色），
    这是让贴纸"说话/吃东西/惊讶"最直接的信号。
  * hop —— 大幅弹跳（幅度 3 倍于 bounce 的缩放 + 明显腾空）。
  * 幅度上限放开到 amp=2.0，配合 SVD/AnimateDiff 做语义细化。

嘴部定位复用 find_eyes 给出的头部框：眼睛下方 ~16%BH 处为嘴，
横向取头部宽的 26%，与贴纸常见的 Q 版比例相符。
"""
ENABLE_MOUTH = False   # 2026-09-28 关停：嘴部定位不可靠（打在贴纸文字/头顶），形变会涂抹出大椭圆色块
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates, gaussian_filter
from motion import find_eyes, _subject_bbox, _body_pivot


# ---------------------------------------------------------------- 嘴部
def find_mouth(rgba, bbox):
    """返回 (cx, cy, half_w, half_h)。找不到返回 None。"""
    eyes = find_eyes(rgba, bbox)
    if eyes is None:
        return None
    ex, _ey, lx, rx, ly, _ry = eyes
    x0, y0, x1, y1 = bbox
    bw = max(x1 - x0, 1)
    bh = max(y1 - y0, 1)
    cx = (lx + rx) * 0.5
    cy = ly + bh * 0.19                 # 眼下一寸
    half_w = bw * 0.14
    if not (0 <= cx <= rgba.width and 0 <= cy <= rgba.height):
        return None
    return (cx, cy, half_w, bh * 0.055)


def _mouth_field(shape, cx, cy, hw, hh, open_amt):
    """张开量 -> 竖向位移场 + 内部填黑掩码。
    张嘴 = 把嘴部像素向下拉、两侧向上收，形成椭圆口腔。"""
    H, W = shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    dx = (xx - cx) / max(hw, 1e-3)
    dy = (yy - cy) / max(hh, 1e-3)
    r = np.sqrt(dx * dx + dy * dy)
    inner = r < 1.0
    # 越靠近中心下拉越狠 -> 嘴竖着张开
    uy = np.where(inner, open_amt * hh * 2.1 * (1.0 - r * r * 0.55), 0.0)
    ux = np.where(inner, -open_amt * hw * open_amt * 0.35 * dx, 0.0)
    # 嘴唇外沿轻微外扩，避免"被吃掉"
    lip = (r > 0.92) & (r < 1.18)
    ux += np.where(lip, open_amt * hw * 0.22 * dx, 0.0)
    uy += np.where(lip, open_amt * hh * 0.5, 0.0)
    return ux, uy, inner


# ---------------------------------------------------------------- 采样
def _sample(src, ux, uy):
    H, W = src.shape[:2]
    cy_ = np.clip(np.mgrid[0:H, 0:W][0] + uy, 0, H - 1).astype(np.float32)
    cx_ = np.clip(np.mgrid[0:H, 0:W][1] + ux, 0, W - 1).astype(np.float32)
    out = np.empty_like(src)
    for c in range(src.shape[2]):
        out[..., c] = map_coordinates(src[..., c], [cy_, cx_], order=1, mode="nearest")
    return out


def warp2(rgba, kind, phase, amp=1.0, open_mode=None, mouth_amp=1.0):
    """v2 形变。phase 0..1。open_mode: None/talk/eat/chew/surprised/angry/sing。"""
    src = np.asarray(rgba, dtype=np.float32)
    H, W = src.shape[:2]
    bbox = _subject_bbox(rgba)
    if bbox is None:
        return rgba
    px, py = _body_pivot(bbox)
    bx0, by0, bx1, by1 = bbox
    bw, bh = max(bx1 - bx0, 1), max(by1 - by0, 1)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    ux = np.zeros_like(xx)
    uy = np.zeros_like(yy)
    hgt = np.clip((py - yy) / bh, 0.0, 1.4)
    wid = (xx - px) / max(bw, 1)
    fall = np.clip(hgt, 0.0, 1.0) ** 1.6

    # ---- 整体动作（沿用 motion.py，幅度随 amp 放大）----
    if kind == "sway":
        s = np.sin(phase * 2 * np.pi) * amp
        ux += s * fall * bh * 0.13
        uy += -abs(s) * fall * bh * 0.02
        ux += -s * np.clip(wid, -1, 1) * fall * bh * 0.04
    elif kind == "shake":
        s = np.sin(phase * 2 * np.pi * 3.0) * amp
        ux += s * fall * bh * 0.09
        uy += -abs(s) * fall * bh * 0.03
    elif kind == "nod":
        s = np.sin(phase * 2 * np.pi) * amp
        uy += s * fall * bh * 0.085
        ux += s * fall * bw * 0.015
        uy += -s * np.clip(hgt, 0, 1) ** 2 * bh * 0.024
    elif kind == "hop":
        # 真实跳跃：腾空上移 + 落地压扁（squash & stretch），不做整体放大
        k = np.sin(phase * 2 * np.pi)
        lift = np.maximum(0.0, k)
        squash = np.maximum(0.0, -k)
        uy += -lift * bh * 0.11 * (0.35 + 0.65 * fall)      # 越靠上位移越大
        foot_y = by1                                        # 以脚底为轴压扁
        uy += squash * (yy - foot_y) * 0.055
        ux += squash * (xx - px) * 0.035
    elif kind == "bounce":
        k = np.sin(phase * 2 * np.pi) * amp
        sc = 1.0 + k * 0.065
        ux += (xx - px) * (sc - 1.0)
        uy += (yy - py) * (sc - 1.0) * 1.35
        uy += np.maximum(0.0, -k) * bh * 0.03 * np.clip(hgt * 1.4, 0, 1)
    elif kind == "breathe":
        k = np.sin(phase * 2 * np.pi) * amp
        sc = 1.0 + k * 0.03
        ux += (xx - px) * (sc - 1.0)
        uy += (yy - py) * (sc - 1.0)
    elif kind == "zoom":
        k = np.sin(phase * 2 * np.pi) * amp
        sc = 1.0 + k * 0.09
        ux += (xx - px) * (sc - 1.0)
        uy += (yy - py) * (sc - 1.0) * 1.2
        uy += -k * fall * bh * 0.05
    elif kind == "tilt":
        s = np.sin(phase * 2 * np.pi) * amp
        ux += s * np.clip(wid, -1, 1) * fall * bh * 0.085
        uy += -abs(s) * fall * bh * 0.015
        ux += s * fall * bh * 0.03

    # ---- 眨眼 ----
    eyes = find_eyes(rgba, bbox)
    if eyes is not None and kind != "breathe":
        ex, _ey, _lx, _rx, _ly, _ry = eyes
        blink = 1.0 if (phase > 0.05 and phase < 0.22) else 0.0
        if blink > 0.01:
            half_w = max(int(bw * 0.26), 8)
            dx = np.abs(xx - ex)
            wgt = np.clip(1.0 - dx / half_w, 0.0, 1.0) ** 1.5
            dy = np.clip(1.0 - np.abs(yy - (_ly + _ry) * 0.5) / (bh * 0.16), 0, 1)
            uy += blink * (wgt * dy) * bh * 0.06

    # ---- 嘴部开合（v2 新增：由 open_mode 独立控制，和整体位移解耦）----
    open_amt = 0.0
    if open_mode and mouth_amp > 0:
        if open_mode == "talk":       # 连续张合，说话
            open_amt = (0.5 + 0.5 * np.sin(phase * 2 * np.pi * 4.0)) * mouth_amp
        elif open_mode == "eat":      # 咬一大口后合上咀嚼
            open_amt = (1.0 if phase < 0.30 else (0.25 if phase < 0.42 else 0.0)) * mouth_amp
        elif open_mode == "chew":     # 反复小幅咀嚼
            open_amt = (0.45 + 0.45 * np.sin(phase * 2 * np.pi * 6.0)) * mouth_amp
        elif open_mode == "surprised":  # 张大嘴，中段最大
            t = np.sin(phase * np.pi)
            open_amt = t * mouth_amp
        elif open_mode == "angry":    # 吼：前半张嘴后半咬牙
            open_amt = (1.0 if phase < 0.55 else 0.25) * mouth_amp
        elif open_mode == "sing":     # 长开并随节拍变化
            open_amt = (0.7 + 0.3 * np.sin(phase * 2 * np.pi * 3.0)) * mouth_amp

    if open_amt > 0.01 and ENABLE_MOUTH:
        m = find_mouth(rgba, bbox)
        if m is not None:
            mx, my, mhw, mhh = m
            vux, vuy, inner = _mouth_field(src.shape, mx, my, mhw, mhh, open_amt)
            ux += vux
            uy += vuy
            arr = _sample(src, ux, uy)
            # 口腔填暗色（舌/口腔），避免采样后是一块背景色的糊斑
            if inner.any():
                H_open = np.array([0.55, 0.18, 0.22], dtype=np.float32) * 255.0
                img = arr.copy()
                idx = inner
                # 只填"原本不透明"的像素，绝不往透明区 paints
                al = img[..., 3]
                img[idx, 0] = np.where(al[idx] > 128, H_open[0], img[idx, 0])
                img[idx, 1] = np.where(al[idx] > 128, H_open[1], img[idx, 1])
                img[idx, 2] = np.where(al[idx] > 128, H_open[2], img[idx, 2])
                arr = img
            return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGBA")

    out = _sample(src, ux, uy)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGBA")


def frame_seq(rgba, kind, n=12, amp=1.0, open_mode=None, mouth_amp=1.0, shrink=0.78):
    """shrink：先给主体留 22% 画布安全边，防止大幅动作顶出画布。"""
    if shrink and shrink < 1.0:
        from pack_motion import shrink_content
        rgba = shrink_content(rgba, shrink)
    return [warp2(rgba, kind, i / (n - 1) if n > 1 else 0.0, amp,
                  open_mode, mouth_amp)
            for i in range(n)]


if __name__ == "__main__":
    import os
    from PIL import ImageDraw
    from pack_sr import SRC
    from motion_map import motion_of

    os.makedirs("motion2_out", exist_ok=True)
    for nm in ["打call.png", "吃饭.png", "WOW.png", "生气.png"]:
        p = os.path.join(SRC, nm)
        if not os.path.exists(p):
            print("缺", nm); continue
        src = Image.open(p).convert("RGBA")
        kw = nm[:-4]
        kind = motion_of(kw)
        seq = frame_seq(src, kind, n=10, amp=1.4, mouth_amp=1.0)
        z, c = 3, 150
        sh = Image.new("RGB", (10 * c * z, c * z + 22), (240, 240, 240))
        d = ImageDraw.Draw(sh)
        for i, f in enumerate(seq):
            bg = Image.new("RGB", f.size, (255, 255, 255)); bg.paste(f, mask=f.split()[3])
            sh.paste(bg.resize((c * z, c * z), Image.NEAREST), (i * c * z, 22))
        d.text((6, 5), f"{kw} / {kind}", fill=(0, 0, 0))
        o = f"motion2_out/{kw}_{kind}.png"
        sh.save(o); print("saved", o)

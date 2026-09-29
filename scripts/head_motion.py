# -*- coding: utf-8 -*-
"""
头部骨骼式形变引擎 v3 —— "角色在动"而非"贴图在动"。

原理：动漫脸检测定位头部 -> 头部区域绕颈部独立旋转（点头/摇头/歪头），
身体反向微摆（Secondary motion），边界羽化无缝融合。

与"放大缩小"的本质区别：非刚体局部形变，头部动、身体不动，
视觉上是角色点头/摇头，而不是图被拖拽。

用法:
  detect_face(im) -> (x,y,w,h) or None
  frame_seq(rgba, kind, n=8) -> [RGBA]
"""
import os
import sys
import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, '.')
from pack_sr import SRC

CAS = cv2.CascadeClassifier('models/cascade/lbpcascade_animeface.xml')


def detect_face(im_rgba):
    """在 RGBA 贴纸上检测动漫脸。返回 (x,y,w,h) 或 None。"""
    bg = Image.new('RGB', im_rgba.size, (255, 255, 255))
    bg.paste(im_rgba, mask=im_rgba.split()[3])
    gray = cv2.cvtColor(np.asarray(bg), cv2.COLOR_RGB2GRAY)
    faces = CAS.detectMultiScale(gray, 1.02, 1,
                                 minSize=(int(min(im_rgba.size) * 0.04),) * 2)
    if not len(faces):
        return None
    # 取最大的脸
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return int(x), int(y), int(w), int(h)


def _head_mask(shape, face, feather=0.06):
    """椭圆头部掩码（覆盖脸+头发+耳朵），高斯羽化。"""
    H, W = shape[:2]
    x, y, w, h = face
    cx, cy = x + w / 2, y + h * 0.52
    ax, ay = w * 1.15, h * 1.75          # 覆盖头发/耳朵
    m = np.zeros((H, W), np.float32)
    cv2.ellipse(m, (int(cx), int(cy)), (int(ax / 2), int(ay / 2)),
                0, 0, 360, 1.0, -1)
    f = int(min(w, h) * feather * 4) | 1
    m = cv2.GaussianBlur(m, (f, f), 0)
    return m[..., None]


def _rot_field(shape, pivot, angle_deg):
    """绕 pivot 旋转 angle 的位移场（整图像素新位置的偏移）。"""
    H, W = shape[:2]
    px, py = pivot
    a = np.deg2rad(angle_deg)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    dx, dy = xx - px, yy - py
    # 反向映射：目标点(x,y)在旋转后图像中的源位置
    sx = px + dx * np.cos(a) - dy * np.sin(a)
    sy = py + dx * np.sin(a) + dy * np.cos(a)
    return sx - xx, sy - yy


def warp_head(rgba, face, angle_head=0.0, angle_body=0.0, body_ratio=0.75):
    """头部绕颈部旋转 angle_head，身体绕下半轴点反向旋转 angle_body。"""
    src = np.asarray(rgba)
    H, W = src.shape[:2]
    x, y, w, h = face
    pivot_head = (x + w / 2, y + h * 1.30)          # 颈部
    pivot_body = (x + w / 2, y + h + (H - y - h) * 0.5)

    mh = _head_mask(src.shape, face)                # (H,W,1)
    mb = 1.0 - mh                                   # (H,W,1)

    hx, hy = _rot_field(src.shape, pivot_head, angle_head)      # (H,W)
    if angle_body:
        bx, by = _rot_field(src.shape, pivot_body, angle_body)  # (H,W)
    else:
        bx = np.zeros((H, W), np.float32)
        by = np.zeros((H, W), np.float32)
    hx, hy, bx, by = (a[..., None] for a in (hx, hy, bx, by))   # -> (H,W,1)

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    fx = (xx[..., None] + mh * hx + mb * bx).astype(np.float32)
    fy = (yy[..., None] + mh * hy + mb * by).astype(np.float32)
    fx = np.ascontiguousarray(fx[..., 0])
    fy = np.ascontiguousarray(fy[..., 0])

    out = np.empty_like(src)
    for c in range(4):
        out[..., c] = cv2.remap(src[..., c], fx, fy,
                                interpolation=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return Image.fromarray(out, 'RGBA')


def frame_seq(rgba, kind, n=8, cycles=2.0, amp=1.0):
    """生成动作帧序列。kind: nod/shake/tilt/hop/breathe。"""
    face = detect_face(rgba)
    frames = []
    if kind in ('nod', 'shake', 'tilt') and face is not None:
        for i in range(n):
            p = i / max(1, n)
            t = 2 * np.pi * cycles * p
            if kind == 'nod':
                ah = np.sin(t) * 7 * amp            # 点头（前后倾用旋转近似）
                ab = -np.sin(t) * 1.2 * amp
            elif kind == 'shake':
                ah = np.sin(t) * 11 * amp           # 摇头（左右歪）
                ab = -np.sin(t) * 1.8 * amp
            else:  # tilt
                ah = np.sin(t) * 13 * amp           # 歪头，幅度大节奏慢
                ab = -np.sin(t) * 2.0 * amp
            frames.append(warp_head(rgba, face, ah, ab))
    elif kind == 'hop':
        # squash & stretch 跳跃（沿用已验证逻辑）
        from motion2 import warp2
        for i in range(n):
            frames.append(warp2(rgba, 'hop', i / max(1, n), amp=1.0))
    elif kind == 'breathe':
        from motion2 import warp2
        for i in range(n):
            frames.append(warp2(rgba, 'breathe', i / max(1, n), amp=1.0))
    else:
        # 无脸/未知 -> 退化为轻摇（整图微旋转，幅度极小不糊弄）
        H = rgba.size[1]
        for i in range(n):
            p = i / max(1, n)
            t = 2 * np.pi * cycles * p
            frames.append(warp_head(rgba, (rgba.size[0] // 2 - H // 5, H // 8,
                                           H // 2.5, H // 2.5),
                                    np.sin(t) * 5 * amp, -np.sin(t) * 1.0))
    return frames, face


def make_gif300(kw, kind, n=8, cycles=2.0, amp=1.0, ms=100, limit_kb=250,
                out_dir='head_out', sharpen=25):
    """源图 -> 300 墨量模型 -> 头部骨骼形变 -> 合规 GIF。"""
    from simple_down import box_down, alpha_fix
    from pack_motion import extend_colors
    from polish import final_sharp, add_outline
    from pack_motion import frames_to_gif, shrink_content

    src = Image.open(os.path.join(SRC, kw + '.png')).convert('RGBA')
    r300 = alpha_fix(box_down(src, 300))
    r300 = shrink_content(r300, 0.92)          # 动作安全边
    fr, face = frame_seq(r300, kind, n=n, cycles=cycles, amp=amp)
    if sharpen:
        fr = [final_sharp(f, sharpen, radius=0.8, thresh=2) for f in fr]
    fr = [add_outline(f, width=2) for f in fr]
    os.makedirs(out_dir, exist_ok=True)
    ng, nf = frames_to_gif(fr, os.path.join(out_dir, f'{kw}_{kind}.gif'),
                           duration=ms, limit_kb=limit_kb, colors=64)
    return face, nf, ng


if __name__ == '__main__':
    for kw, kind in [('打call', 'nod'), ('不慌', 'shake'), ('开心', 'tilt'),
                     ('吃东西', 'nod'), ('牛逼', 'shake'), ('WOW', 'tilt')]:
        face, nf, ng = make_gif300(kw, kind)
        print(f'{kw} [{kind}]: face={face} {nf}帧 {ng/1024:.0f}KB', flush=True)

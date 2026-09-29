# -*- coding: utf-8 -*-
"""v2 动效 GIF 生成：motion2 语义动作序列 -> 64色多帧 GIF。"""
import os, sys
import numpy as np
from PIL import Image

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from simple_down import box_down, alpha_fix
from motion2 import frame_seq
from motion_map import motion2_of
from pack_motion import frames_to_gif, extend_colors
from polish import final_sharp


def make_frames_v2(r300, kw, n_frames=12, amp=1.0, ms=80, sharpen=25,
                   mouth_amp=1.0):
    """输入 300px v3 成品，返回 (帧列表, 位移动作, 口型)。供主管线调用。"""
    kind, mouth = motion2_of(kw)
    frames = frame_seq(r300, kind, n=n_frames, amp=amp,
                       open_mode=mouth, mouth_amp=mouth_amp)
    if sharpen:
        frames = [final_sharp(f, sharpen, radius=0.8, thresh=2) for f in frames]
    return frames, kind, mouth


def make_gif_v2(src, kw, n_frames=12, amp=1.0, ms=80, out_path=None,
                limit_kb=250, colors=64, sharpen=25):
    """源图 RGBA -> 300 墨量模型 -> v2 语义动作 -> GIF。"""
    r300 = alpha_fix(box_down(src, 300))
    frames, kind, mouth = make_frames_v2(r300, kw, n_frames, amp, ms, sharpen)
    if out_path:
        frames_to_gif(frames, out_path, duration=ms, limit_kb=limit_kb, colors=colors)
    return frames, kind, mouth


if __name__ == "__main__":
    from pack_sr import SRC, robust_write
    os.makedirs("v2_out", exist_ok=True)
    NAMES = ["加油.png", "吃东西.png", "WOW.png", "听我说.png", "牛逼.png", "不慌.png"]
    for nm in NAMES:
        p = os.path.join(SRC, nm)
        if not os.path.exists(p):
            print("缺", nm); continue
        src = Image.open(p).convert("RGBA")
        kw = nm[:-4]
        frames, kind, mouth = make_gif_v2(src, kw)
        o = os.path.join("v2_out", f"{kw}.gif")
        size, nf = frames_to_gif(frames, o, duration=80, limit_kb=250, colors=64)
        print(f"{kw}  kind={kind}/{mouth}  {nf}帧  {size//1024}KB", flush=True)

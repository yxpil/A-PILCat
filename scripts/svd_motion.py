# -*- coding: utf-8 -*-
"""
SVD（Stable Video Diffusion 图生视频）语义动作链。

与旧的两条路线的本质区别：
  * pack_motion / motion.py  —— 对整张图做位移场，动作是"贴图在抖"，没有语义。
  * ai_motion.py (AnimateDiff v2v) —— 仍是整图渐变，等价于上面那条路的 AI 版。
  * 本脚本 —— 把表情关键字映射成**具体动作描述**（举手/张嘴/跺脚/比心…），
             交给 SVD 真正渲染出来。肢体和口型会动，而不是整体平移。

流程：
  源图 -> 建 576x1024 白底输入 -> SVD 生成 25 帧 -> 裁中间正方形
      -> 逐帧抠 alpha -> 缩 300 -> 组装 GIF
"""
import os, sys, time, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
from PIL import Image, ImageDraw

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from matting_gpu import matte
from actions import action_of

MODEL = os.path.join(BASE, "models", "svd")
N_FRAMES = 25          # SVD-XT 原生帧数
USE_FRAMES = 16        # 最终用的帧数
FPS_MS = 62            # 16 帧 * 62ms ≈ 1.0s
MOTION_BUCKET = 150    # 越大动作幅度越大
STEPS = 25


def load_pipe():
    from diffusers import StableVideoDiffusionPipeline
    pipe = StableVideoDiffusionPipeline.from_pretrained(MODEL, torch_dtype=torch.float16)
    try:
        pipe.enable_model_cpu_offload()      # 16G 显存下最稳，速度损失可接受
    except Exception:
        pass
    try:
        pipe.enable_vae_slicing()
    except Exception:
        pass
    pipe.set_progress_bar_config(disable=True)
    return pipe


def build_input(rgba, H=1024, W=576):
    """把透明贴纸贴到 SVD 建议分辨率 576x1024 的白底画布上，居中。"""
    bg = Image.new("RGB", (W, H), (255, 255, 255))
    a = np.array(rgba.split()[3]).astype(np.float32) / 255.0
    ys, xs = np.where(a > 0.25)
    if len(ys) == 0:
        t = rgba
    else:
        t = rgba.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    # 贴纸按高度放进画布（贴纸多为方图，这样宽度不会溢出 W）
    sc = min(H * 0.96 / max(t.size[1], 1), W * 0.96 / max(t.size[0], 1))
    nw, nh = max(1, round(t.size[0] * sc)), max(1, round(t.size[1] * sc))
    t = t.resize((nw, nh), Image.LANCZOS)
    bg.paste(t, ((W - nw) // 2, (H - nh) // 2), t)
    return bg


def gen(pipe, rgba, kw, seed=0):
    inp = build_input(rgba)
    act = action_of(kw)
    prompt = (f"anime chibi cartoon mascot sticker, {act}, "
              "white background, clean lineart, consistent character, smooth motion")
    neg = ("blurry, low quality, extra limbs, extra fingers, deformed face, "
           "text, watermark, dark background, camera shake")
    g = torch.Generator(device="cuda").manual_seed(seed)
    t0 = time.time()
    out = pipe(image=inp, prompt=prompt, negative_prompt=neg,
               num_frames=N_FRAMES, motion_bucket_id=MOTION_BUCKET,
               num_inference_steps=STEPS, decode_chunk_size=2,
               generator=g).frames[0]
    print(f"    SVD 推理 {time.time()-t0:.0f}s | {act[:60]}", flush=True)
    return out   # list[PIL RGB]


def pick(frames, n):
    """等间隔抽样 n 帧，保证动作铺满整段。"""
    if len(frames) <= n:
        return list(frames)
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    return [frames[i] for i in idx]


def to_rgba300(fr, size=300):
    """SVD 输出 1024x576 -> 裁中间正方形 -> 抠 alpha -> 300。"""
    w, h = fr.size
    s = min(w, h)
    l = (w - s) // 2
    top = (h - s) // 2
    fr = fr.crop((l, top, l + s, top + s)).resize((size, size), Image.LANCZOS)
    return matte(fr.convert("RGBA"))


def sheet(rgba_list, path, cols=8, cell=110, zoom=2):
    n = len(rgba_list)
    rows = (n + cols - 1) // cols
    W = cols * cell * zoom
    H = rows * cell * zoom + 22
    sh = Image.new("RGB", (W, H), (238, 238, 238))
    dr = ImageDraw.Draw(sh)
    for i, f in enumerate(rgba_list):
        bg = Image.new("RGB", f.size, (255, 255, 255))
        bg.paste(f, mask=f.split()[3])
        c = bg.resize((cell * zoom, cell * zoom), Image.NEAREST)
        sh.paste(c, ((i % cols) * cell * zoom, 22 + (i // cols) * cell * zoom))
    dr.text((6, 5), os.path.basename(path), fill=(0, 0, 0))
    sh.save(path)

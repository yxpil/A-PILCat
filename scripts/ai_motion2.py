# -*- coding: utf-8 -*-
"""
语义动作 AI 细化（AnimateDiff v2v）—— v2 修正版。

v1 的致命缺陷：把**同一帧重复 8 次**喂给 v2v。模型收到的是一段静止画面，
除了 prompt 之外没有任何运动信号，只能做"整体渐变插值"，
成片观感 = 程序化位移场的 AI 版，仍是"贴图在抖"。

v2 修正：先用 motion2 生成**语义动作序列**（举手/张嘴/弹跳，amp 放大），
把这 12 帧真实运动作为 v2v 输入。模型看到的是一段会动的片子，
motion latent 里才有东西可插值 -> 动作连贯且贴合表情语义。
prompt 再叠加 actions.py 的具体描述做进一步约束。

流程：源图 -> motion2 语义序列 -> 白底 512 序列 -> v2v -> 逐帧抠 alpha -> 300 GIF
"""
import os, sys, time, argparse
import numpy as np
import torch
from PIL import Image, ImageDraw

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from matting_gpu import matte
from actions import action_of
from motion_map import motion_of

STRENGTH = float(os.environ.get("AD_STRENGTH", "0.62"))
N_IN = 12
N_FINAL = 12


def load_pipe():
    from diffusers import MotionAdapter, AnimateDiffVideoToVideoPipeline, EulerAncestralDiscreteScheduler
    adapter = MotionAdapter.from_pretrained(
        os.path.join(BASE, "models", "animatediff"), torch_dtype=torch.float16)
    pipe = AnimateDiffVideoToVideoPipeline.from_pretrained(
        os.path.join(BASE, "models", "sd15"), motion_adapter=adapter, torch_dtype=torch.float16)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipe.scheduler.config, beta_schedule="linear")
    try:
        pipe.enable_vae_slicing()
    except AttributeError:
        pass
    pipe.set_progress_bar_config(disable=True)
    return pipe.to("cuda")


def seq_to_white512(seq_rgba, size=512):
    """RGBA 序列 -> 白底 RGB 序列（AnimateDiff 只吃 RGB 视频）。"""
    bg = Image.new("RGB", seq_rgba[0].size, (255, 255, 255))
    out = []
    for f in seq_rgba:
        c = Image.alpha_composite(bg.convert("RGBA"), f).convert("RGB")
        out.append(c.resize((size, size), Image.LANCZOS))
    return out


def white512_to_rgba300(fr, size=300):
    """v2v 输出 RGB -> 抠 alpha -> 300。"""
    im = fr.resize((size, size), Image.LANCZOS).convert("RGBA")
    return matte(im)


def run(pipe, src, kw, amp, seed=0, steps=20):
    from motion2 import frame_seq
    kind = motion_of(kw)
    seq = frame_seq(src, kind, n=N_IN, amp=amp, mouth_amp=1.0)
    frames_in = seq_to_white512(seq)
    act = action_of(kw)
    prompt = (f"anime chibi sticker mascot, {act}, clean lineart, flat pastel colors, "
              "white background, cohesive character, smooth natural animation, looping")
    neg = ("blurry, low quality, extra limbs, extra fingers, deformed face, text, "
           "watermark, dark background, flickering")
    g = torch.Generator("cuda").manual_seed(seed)
    t0 = time.time()
    out = pipe(prompt=prompt, negative_prompt=neg, video=frames_in,
               strength=STRENGTH, generator=g, num_inference_steps=steps).frames[0]
    print(f"    v2v {time.time()-t0:.0f}s  kind={kind}  strength={STRENGTH}", flush=True)
    return [white512_to_rgba300(f) for f in out]


def gif(frames, path, ms=80):
    ps = []
    for f in frames:
        bg = Image.new("RGB", f.size, (255, 255, 255))
        bg.paste(f, mask=f.split()[3])
        ps.append(bg.convert("P", palette=Image.ADAPTIVE, colors=64))
    ps[0].save(path, "GIF", save_all=True, append_images=ps[1:], duration=ms,
               loop=0, transparency=64, disposal=2, optimize=True)


def sheet(frames, path, cols=6, cell=100, zoom=2):
    n = len(frames)
    rows = (n + cols - 1) // cols
    sh = Image.new("RGB", (cols * cell * zoom, rows * cell * zoom + 20), (238, 238, 238))
    dr = ImageDraw.Draw(sh)
    for i, f in enumerate(frames):
        bg = Image.new("RGB", f.size, (255, 255, 255))
        bg.paste(f, mask=f.split()[3])
        sh.paste(bg.resize((cell * zoom, cell * zoom), Image.NEAREST),
                 ((i % cols) * cell * zoom, 20 + (i // cols) * cell * zoom))
    dr.text((6, 4), os.path.basename(path), fill=(0, 0, 0))
    sh.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="+", default=["打call.png"])
    ap.add_argument("--amp", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="ai2_out")
    a = ap.parse_args()
    from pack_sr import SRC
    os.makedirs(a.outdir, exist_ok=True)
    pipe = load_pipe()
    for nm in a.names:
        p = os.path.join(SRC, nm)
        if not os.path.exists(p):
            print("缺", nm); continue
        src = Image.open(p).convert("RGBA")
        kw = nm[:-4]
        t = time.time()
        frames = run(pipe, src, kw, a.amp, a.seed)
        base = os.path.join(a.outdir, f"AI2_{kw}")
        sheet(frames, base + ".png")
        gif(frames, base + ".gif", 80)
        print(f"  {nm} 完成 {time.time()-t:.0f}s -> {base}.gif", flush=True)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
AI 动效对照实验：AnimateDiff v2v。
原图重复 8 帧作输入视频 -> v2v strength 控制改动 -> 512 生成 -> 缩回 300 -> GPU 抠图回透明。
目的：验证 AI 能否在保持画风的前提下加动作，与程序化形变对比。
"""
import os, sys, time, io
import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from matting_gpu import matte

BASE = os.path.dirname(os.path.abspath(__file__))
N_FRAMES = 8
STRENGTH = float(os.environ.get("AD_STRENGTH", "0.45"))


def load_pipe():
    from diffusers import MotionAdapter, AnimateDiffVideoToVideoPipeline, EulerAncestralDiscreteScheduler
    adapter = MotionAdapter.from_pretrained(
        os.path.join(BASE, "models", "animatediff"), torch_dtype=torch.float16)
    pipe = AnimateDiffVideoToVideoPipeline.from_pretrained(
        os.path.join(BASE, "models", "sd15"),
        motion_adapter=adapter, torch_dtype=torch.float16)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipe.scheduler.config, beta_schedule="linear")
    try:
        pipe.enable_vae_slicing()
    except AttributeError:
        pass
    pipe.set_progress_bar_config(disable=True)
    return pipe.to("cuda")


def run(pipe, rgba300, kw, seed=0):
    # 白底方形 512 输入（AnimateDiff 对分辨率敏感，非训练分辨率会崩）
    flat = Image.alpha_composite(Image.new("RGBA", rgba300.size, (255, 255, 255, 255)),
                                 rgba300).convert("RGB")
    frames_in = [flat.resize((512, 512), Image.LANCZOS)] * N_FRAMES
    prompt = (f"chibi anime sticker, {kw}, cute girl with cat ears, soft pastel colors, "
              "clean lineart, white background, subtle animation, looping")
    neg = "blurry, realistic, dark, text artifacts, extra limbs, deformed"
    g = torch.Generator("cuda").manual_seed(seed)
    t0 = time.time()
    # v2v 的帧数由输入 video 长度决定，无需 num_frames
    out = pipe(prompt=prompt, negative_prompt=neg, video=frames_in,
               strength=STRENGTH, generator=g,
               num_inference_steps=20).frames[0]
    print(f"  AI推理 {time.time()-t0:.0f}s", flush=True)
    return out   # list[PIL RGB 512]


def to_rgba300(frame_rgb512):
    im = frame_rgb512.resize((300, 300), Image.LANCZOS)
    return matte(im.convert("RGBA"))


def main():
    from pack_sr import SRC
    names = ["打call.png", "DNA动了.png", "WOW.png"]
    pipe = load_pipe()
    os.makedirs("ai_out", exist_ok=True)
    for nm in names:
        p = os.path.join(SRC, nm)
        if not os.path.exists(p):
            print("缺", nm); continue
        src = matte(Image.open(p).convert("RGBA"))
        # 先降到 300（模拟真实输入），再进 AI
        from line_down2 import down_line
        r300 = down_line(src, 300)
        kw = os.path.splitext(nm)[0]
        frames = run(pipe, r300, kw)
        rgba = [to_rgba300(f) for f in frames]
        # 白底拼版预览
        Z, C = 2, 150
        sheet = Image.new("RGB", (N_FRAMES * C * Z + 20, C * Z + 30), (245, 245, 245))
        dr = ImageDraw.Draw(sheet)
        for i, f in enumerate(rgba):
            bg = Image.new("RGB", f.size, (250, 250, 250))
            bg.paste(f, mask=f.split()[3])
            c = bg.crop((75, 60, 75 + C, 60 + C)).resize((C * Z, C * Z), Image.NEAREST)
            sheet.paste(c, (10 + i * (C * Z + 4), 22))
        dr.text((10, 4), f"{nm} v2v strength={STRENGTH}", fill=(0, 0, 0))
        out = os.path.join("ai_out", f"AI_{nm[:-4]}.png")
        sheet.save(out)
        print("saved", out, flush=True)


if __name__ == "__main__":
    main()

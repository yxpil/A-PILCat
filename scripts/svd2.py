# -*- coding: utf-8 -*-
"""
SVD 语义动效链 v2（img2vid -> 光流反投影，全程 576 空间对齐）

AI 只提供"怎么动"（光流场），原图提供"长什么样/哪里透明"。

流程：
  1. 贴纸 RGBA -> 白底 -> 576x576 启动帧（记录布局 bbox）
  2. StableVideoDiffusionPipeline 出 25 帧 RGB
  3. 对每帧算相对第 0 帧的 Farneback 稠密光流
  4. 光流应用在 576 空间的 RGBA 贴纸上（同一布局，精确对齐）-> warp
  5. warp 后 RGBA 裁回 bbox -> 墨量模型降到 300 -> 锐化 + 描边
  6. frames_to_gif 打包
"""
import os
import sys
import time
import argparse

sys.path.insert(0, '.')
import numpy as np
import torch
from PIL import Image
import cv2

from pack_sr import SRC
from matting_gpu import matte
from polish import sr_polish, final_sharp, add_outline
from line_down2 import down_line
from pack_motion import shrink_content, frames_to_gif

MODEL = 'models/svd'
S_W, S_H = 576, 1024         # SVD 原生竖屏分辨率，启动帧按此比例布局，避免内部拉伸变形
S = S_W
NUM_FRAMES = 25              # SVD 原生 25 帧
STEPS = 25
MOTION_ID = 127              # 动作幅度桶


def layout(rgba, size=(S_W, S_H), fill=0.92):
    """RGBA -> (白底RGB启动帧, RGBA同布局, bbox)。
    贴纸居中缩放到 SVD 原生画布，模型不做任何变形。"""
    iw, ih = rgba.size
    W, H = size
    scale = min(W / iw, H / ih) * fill
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    ox, oy = (W - nw) // 2, (H - nh) // 2
    im = rgba.resize((nw, nh), Image.LANCZOS)

    rgb = Image.new('RGB', (W, H), (255, 255, 255))
    rgb.paste(im.convert('RGB'), (ox, oy))

    rgba576 = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    rgba576.paste(im, (ox, oy))

    return rgb, rgba576, (ox, oy, ox + nw, oy + nh)


def _fit(a, size):
    """SVD 输出是原生 576x1024，统一 resize 回工作分辨率再算光流。"""
    return np.asarray(Image.fromarray(a).resize(size, Image.LANCZOS))


def flow_warp(src_rgba576, g0, cur_rgb, flow_scale=1.0, smooth=1.2, down=2):
    """在 576 空间，用 SVD 第0帧->当前帧的光流 warp RGBA 贴纸。
    down: 光流先在 1/down 分辨率计算再上采样 —— 高频光流噪声直接消失，
    细节区（头发/衣边）不再拉出毛刺。"""
    W, H = src_rgba576.size
    cur = np.asarray(Image.fromarray(cur_rgb).resize((W, H), Image.LANCZOS))
    g1 = cv2.cvtColor(cur, cv2.COLOR_RGB2GRAY)

    w2, h2 = W // down, H // down
    g0s = cv2.resize(g0, (w2, h2), interpolation=cv2.INTER_AREA)
    g1s = cv2.resize(g1, (w2, h2), interpolation=cv2.INTER_AREA)
    flow = cv2.calcOpticalFlowFarneback(
        g0s, g1s, None, 0.5, 3, 21, 3, 5, 1.2, 0)
    if smooth:
        flow = cv2.GaussianBlur(flow, (0, 0), smooth)
    flow = cv2.resize(flow, (W, H), interpolation=cv2.INTER_LINEAR) * down
    mx, my = np.meshgrid(np.arange(W, dtype=np.float32),
                         np.arange(H, dtype=np.float32))
    fx, fy = mx + flow[..., 0] * flow_scale, my + flow[..., 1] * flow_scale

    src = np.asarray(src_rgba576)
    out = np.empty_like(src)
    for c in range(4):
        out[..., c] = cv2.remap(src[..., c], fx, fy,
                                interpolation=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=0)
    return Image.fromarray(out, 'RGBA')


def run_one(kw, outdir, sample=8, bucket=MOTION_ID, steps=STEPS,
            flow_scale=1.0, sharpen=25, limit_kb=250, colors=64, ms=80,
            master_cache=None, pipe_cache=None):
    """跑一张：SVD 出帧 -> 光流反投影 -> GIF 帧。返回 (frames, dt)。"""
    from diffusers import StableVideoDiffusionPipeline

    src = matte(Image.open(os.path.join(SRC, kw + '.png')).convert('RGBA'))
    master = (master_cache or {}).get(kw) or sr_polish(
        src, size=1900, ow=0, lq_max=475, denoise=0.0)

    rgb0, rgba576, bbox = layout(master, (S_W, S_H))

    if pipe_cache is not None and 'p' in pipe_cache:
        pipe = pipe_cache['p']
    else:
        pipe = StableVideoDiffusionPipeline.from_pretrained(
            MODEL, torch_dtype=torch.float16)
        pipe.enable_attention_slicing()
        pipe.to('cuda')
        if pipe_cache is not None:
            pipe_cache['p'] = pipe

    os.makedirs(outdir, exist_ok=True)
    cache = os.path.join(outdir, f'{kw}_svdframes.npz')
    if os.path.exists(cache):
        svd_frames = list(np.load(cache)['f'])
        dt = 0.0
    else:
        t0 = time.time()
        with torch.inference_mode():
            out = pipe(image=rgb0, num_frames=NUM_FRAMES, num_inference_steps=steps,
                       motion_bucket_id=bucket, fps=7,
                       noise_aug_strength=0.0).frames[0]
        dt = time.time() - t0
        svd_frames = [np.asarray(f.convert('RGB')) for f in out]
        np.savez_compressed(cache, f=np.stack(svd_frames))
    for i, f in enumerate(svd_frames[::8][:3]):
        Image.fromarray(f).save(os.path.join(outdir, f'{kw}_svd{i}.png'))
    # 存启动帧对照
    rgb0.save(os.path.join(outdir, f'{kw}_start.png'))

    W0, H0 = rgba576.size
    g0 = cv2.cvtColor(_fit(svd_frames[0], (W0, H0)), cv2.COLOR_RGB2GRAY)
    k = len(svd_frames)
    picks = sorted(set(int(round(j * (k - 1) / max(1, sample - 1)))
                       for j in range(sample)))

    frames = []
    for i in picks:
        w = flow_warp(rgba576, g0, svd_frames[i], flow_scale=flow_scale)
        x0, y0, x1, y1 = bbox
        w = w.crop((x0, y0, x1, y1))                    # 裁回贴纸区域
        small = down_line(w, 300)                       # 墨量模型 300
        small = shrink_content(small, 0.92)             # 动作安全边
        if sharpen:
            small = final_sharp(small, sharpen, radius=0.8, thresh=2)
        small = add_outline(small, width=2)
        frames.append(small)
    return frames, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--names', nargs='*', default=None)
    ap.add_argument('--out', default='svd_out')
    ap.add_argument('--bucket', type=int, default=MOTION_ID)
    ap.add_argument('--steps', type=int, default=STEPS)
    ap.add_argument('--sample', type=int, default=8)
    ap.add_argument('--flow', type=float, default=1.0)
    a = ap.parse_args()

    names = a.names or ['打call', 'WOW', '吃东西', '牛逼']
    os.makedirs(a.out, exist_ok=True)

    pipe = {}
    for kw in names:
        t0 = time.time()
        fr, dt = run_one(kw, a.out, sample=a.sample, bucket=a.bucket,
                         steps=a.steps, flow_scale=a.flow, pipe_cache=pipe)
        gp = os.path.join(a.out, f'{kw}.gif')
        ng, nf = frames_to_gif(fr, gp, duration=80, limit_kb=250, colors=64)
        print(f'{kw}: SVD {dt:.0f}s  {nf}帧  GIF {ng/1024:.0f}KB  总 {time.time()-t0:.0f}s',
              flush=True)
        torch.cuda.empty_cache()   # 防止多张推理显存累积导致掉速(60s/step)


if __name__ == '__main__':
    main()

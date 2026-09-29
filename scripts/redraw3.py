# -*- coding: utf-8 -*-
"""SD1.5 + ControlNet(lineart_anime) img2img 真重绘 —— 每张贴纸画 3 张。

与上一版 pack_hd（形变 + 放大）的本质区别：
  * 走扩散模型重画细节，lineart ControlNet 锁住原线稿结构 -> 构图不变、画面重画
  * 每张贴纸 3 个 seed -> 3 张候选
  * 提示词按贴纸语义逐条单独写（prompt_map.py），不是通用套话

输出: redraw3/<kw>_s<seed>.png   768 长边（8 的倍数，与输入严格 1:1 像素对应）

用法:
  python redraw3.py --test          # 3 张贴纸试跑
  python redraw3.py --pick a,b,c    # 指定贴纸
  python redraw3.py                 # 全量 154 张
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pack_sr import SRC, content_bbox, collect_items, keyword_of, fix_keyword
from matting_gpu import matte
from prompt_map import prompt_of

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models")
SD15 = os.path.join(MODELS, "sd15")
CN_PATH = os.path.join(MODELS, "cn_lineart_anime")
OUT = os.path.join(HERE, "redraw3")

WORK = 768          # 重绘工作分辨率（长边，8 的倍数对齐）
STRENGTH = 0.55     # 重绘幅度：0.55 = 结构保留、细节重画
CN_SCALE = 0.85     # lineart 控制强度
STEPS = 28
GUIDE = 7.0
SEEDS = [11, 22, 33]
PAD = 0.06          # 输入留边（白底），避免重绘顶到边缘


# ---------- 输入准备 ----------
def prep(name, size=WORK):
    """源图 -> GPU 抠图 -> trim -> 白底 + 等比 fit(long=size, 8 对齐)。
    返回 (白底 RGB 输入, alpha 蒙版, 缩放系数)"""
    src = Image.open(os.path.join(SRC, name)).convert("RGBA")
    m = matte(src)
    bb = content_bbox(m)
    m = m.crop(bb) if bb else m
    w, h = m.size
    s = (size * (1 - 2 * PAD)) / max(w, h)
    nw = max(8, int(round(w * s / 8)) * 8)
    nh = max(8, int(round(h * s / 8)) * 8)
    small = m.resize((nw, nh), Image.LANCZOS)
    bg = Image.new("RGB", (nw, nh), (255, 255, 255))
    bg.paste(small, mask=small.split()[3])
    return bg, small.split()[3], (nw, nh)


# ---------- 模型 ----------
_PIPE = None
_LINEART = None


def get_pipe():
    global _PIPE, _LINEART
    if _PIPE is None:
        from diffusers import (StableDiffusionControlNetImg2ImgPipeline,
                               ControlNetModel, UniPCMultistepScheduler)
        cn = ControlNetModel.from_pretrained(CN_PATH, torch_dtype=torch.float16)
        p = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            SD15, controlnet=cn, torch_dtype=torch.float16,
            safety_checker=None, requires_safety_checker=False)
        p.scheduler = UniPCMultistepScheduler.from_config(p.scheduler.config)
        p.set_progress_bar_config(disable=True)
        p.to("cuda")
        p.enable_attention_slicing()
        _PIPE = p
        from controlnet_aux import LineartAnimeDetector
        _LINEART = LineartAnimeDetector.from_pretrained("lllyasviel/Annotators")
        print("[redraw] pipeline + lineart_anime ready", flush=True)
    return _PIPE, _LINEART


def redraw(name, tag=None, work=WORK, seeds=SEEDS, strength=STRENGTH, up=0, sharp=0):
    """一张贴纸 -> n 张重绘候选。up>0 超分放大；sharp>0 重绘后 USM 补锐度。
    返回 (候选列表, 输入图, 线稿图)"""
    pipe, lineart = get_pipe()
    init, alpha, _ = prep(name, work)
    cond = lineart(init.convert("RGB"))
    kw = tag or fix_keyword(keyword_of(name))
    pos, neg = prompt_of(kw)
    outs = []
    for sd in seeds:
        g = torch.Generator("cuda").manual_seed(sd)
        img = pipe(prompt=pos, negative_prompt=neg, image=init, control_image=cond,
                   width=init.width, height=init.height,
                   strength=strength, num_inference_steps=STEPS,
                   guidance_scale=GUIDE, controlnet_conditioning_scale=CN_SCALE,
                   generator=g).images[0]
        if up:
            img = upscale(img, up)
        if sharp:
            from polish import final_sharp
            img = final_sharp(img.convert("RGBA"), sharp, radius=1.2, thresh=2).convert("RGB")
        outs.append((sd, img))
    return outs, init, cond


def upscale(rgb, scale=2):
    """RealESRGAN x4 超分（占位放大到目标倍数）。"""
    from pack_sr import superres
    big = superres(rgb, max(rgb.size))
    if scale == 4:
        return big
    return big.resize((max(1, big.width // 2), max(1, big.height // 2)),
                      Image.LANCZOS)


def sheet(name, outs, init, cond, path):
    """对比板：输入 / 线稿 / 3 张重绘"""
    T = 300
    cells = [("输入(768)", init.convert("RGBA")), ("lineart", cond.convert("RGBA"))]
    for sd, im in outs:
        cells.append((f"seed {sd}", im.convert("RGBA")))
    W = len(cells) * (T + 8) + 8
    sh = Image.new("RGB", (W, T + 30), (245, 245, 245))
    dr = ImageDraw.Draw(sh)
    for i, (lab, im) in enumerate(cells):
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[3] if im.mode == "RGBA" else None)
        sh.paste(bg.resize((T, T), Image.LANCZOS), (8 + i * (T + 8), 24))
        dr.text((10 + i * (T + 8), 6), lab, fill=(20, 20, 20))
    sh.save(path)


def main():
    args = sys.argv[1:]
    os.makedirs(OUT, exist_ok=True)
    work = int(args[args.index("--work") + 1]) if "--work" in args else WORK
    up = int(args[args.index("--up") + 1]) if "--up" in args else 0
    sharp = int(args[args.index("--sharp") + 1]) if "--sharp" in args else 0
    strength = float(args[args.index("--strength") + 1]) if "--strength" in args else STRENGTH
    tag = f"_w{work}" + (f"_st{strength}" if strength != STRENGTH else "") \
        + (f"_up{up}" if up else "") + (f"_sh{sharp}" if sharp else "")
    if "--test" in args:
        names = ["打call.png", "吃东西.png", "加油.png"]
    elif "--pick" in args:
        names = [a + ".png" for a in args[args.index("--pick") + 1].split(",")]
    else:
        names = collect_items()
    print(f"待重绘 {len(names)} 张，每张 {len(SEEDS)} 个 seed，"
          f"工作分辨率 {work}，strength {strength}，超分 x{up}，锐化 {sharp}", flush=True)
    t_all = time.time()
    for i, name in enumerate(names, 1):
        kw = fix_keyword(keyword_of(name))
        t0 = time.time()
        outs, init, cond = redraw(name, kw, work=work, up=up, sharp=sharp, strength=strength)
        for sd, im in outs:
            im.save(os.path.join(OUT, f"{kw}_s{sd}{tag}.png"))
        sheet(name, outs, init, cond, os.path.join(OUT, f"_view_{kw}{tag}.png"))
        print(f"[{i}/{len(names)}] {kw}  {time.time()-t0:.1f}s  "
              f"-> {kw}_s{SEEDS[0]}{tag}.png ...", flush=True)
    print(f"\nALL DONE {time.time()-t_all:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()

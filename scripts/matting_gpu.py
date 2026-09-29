# -*- coding: utf-8 -*-
"""GPU 增强抠图：白底贴纸 -> 干净人物 alpha。

传统 unwhite() 是"白度>242 且与画布边缘连通 -> 透明"的**二值**切割，
问题有两个：
  1) 白底到内容之间的抗锯齿/阴影过渡带（灰度 180~242）不满足白度阈值，
     残留成灰晕 -> 视觉上的"粘连"
  2) 白色阈值一刀切，容易把贴纸本体的白描边、人像内部白色一起削掉 -> 切边/切脑袋

这里改成软 alpha：
  p_bg = 白度(软) x 连通性先验(与边缘连通才是背景)
再用原图亮度做 guided filter 细化边缘，最后轻度闭运算补内部空洞。
白度场与滤波都在 CUDA 上跑。
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
from scipy.ndimage import label, distance_transform_edt, binary_closing, binary_opening
from PIL import Image, ImageFilter

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _tensor(x, dtype=torch.float32):
    return torch.as_tensor(np.ascontiguousarray(x), dtype=dtype, device=DEVICE)


# ---------- 1) 软白度场（GPU） ----------
def whiteness(rgb01, lo=0.42, hi=0.86, sat_max=0.14):
    """三通道都亮且低饱和 -> 白度趋近 1；含色调/偏暗 -> 趋近 0。软过渡。"""
    t = _tensor(rgb01)
    mn = t.min(dim=-1).values
    mx = t.max(dim=-1).values
    sat = mx - mn
    lit = torch.clamp((mn - lo) / (hi - lo), 0.0, 1.0)      # 亮度侧的软斜坡
    unsat = torch.clamp(1.0 - sat / sat_max, 0.0, 1.0)      # 饱和度侧的软斜坡
    return (lit * unsat).cpu().numpy()


def _smoothstep(x, a, b):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def _disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (y * y + x * x) <= r * r


def _close(mask, r):
    """形态学闭合（圆盘半径 r），用距离变换实现，1920px 下比结构元素扫描快得多"""
    d1 = distance_transform_edt(~mask)      # 每点到内容核的距离
    dil = d1 <= r                           # 膨胀
    d2 = distance_transform_edt(dil)        # dil 内每点到补集的距离
    return d2 > r                           # 腐蚀


# ---------- 2) 连通性先验（CPU，scipy 一次即可） ----------
def _bg_prior(white_mask, thresh=0.5):
    """与画布边缘 8-连通的白色块 = 外部背景；其余白色 = 人像内部白/白描边，保留。"""
    wm = white_mask > thresh
    if not wm.any():
        return np.zeros_like(wm, dtype=bool)
    lab, n = label(wm)
    border = set(lab[0, :]) | set(lab[-1, :]) | set(lab[:, 0]) | set(lab[:, -1])
    border.discard(0)
    return np.isin(lab, list(border))


# ---------- 3) guided filter（GPU） ----------
def _box_mean(x, r):
    """半径 r 的 box 均值（conv2d 平均池化，GPU 上很快）"""
    k = 2 * r + 1
    w = torch.ones(1, 1, k, k, dtype=x.dtype, device=x.device) / float(k * k)
    return torch.nn.functional.conv2d(x[None, None], w, padding=r).squeeze(0).squeeze(0)


def guided_filter(guide, src, r=4, eps=1e-3):
    """用 guide 的细节把 src 的边缘粘回来（guided image filter, Ho & Lee）"""
    g, s = _tensor(guide), _tensor(src)
    mean_g, mean_s = _box_mean(g, r), _box_mean(s, r)
    cov = _box_mean(g * s, r) - mean_g * mean_s
    var = _box_mean(g * g, r) - mean_g * mean_g
    a = cov / (var + eps)
    b = mean_s - a * mean_g
    return (a * g + b).cpu().numpy()


# ---------- 4) 主入口 ----------
def matte(src_rgba, soft=0.55, close_r=3, guide_r=4, seal_r=18):
    """白底贴纸 -> 人物 RGBA（透明背景 + 保留内部白/白描边/白发）。

    关键：白发填充色可与背景一样白（实测亮度 0.97-0.99）且连通，
    任何白度阈值洪泛都必然漏进头发。屏障只有描边线。
    所以先对内容核(W<soft)做形态学闭合封住描边缝隙，再洪泛：
      coarse = 0.80 白度洪泛（有缝隙泄漏，但决定灰晕去除范围）
      sealed = 闭合后洪泛（漏不进内容，但可能把窄凹缝误封）
      prior  = coarse AND sealed  ->  头发被 sealed 否决，灰晕仍被 coarse 覆盖
    """
    rgb = np.array(src_rgba.convert("RGB")).astype(np.float32) / 255.0
    # 源图已带真实透明通道 -> 无需抠图，直接透传（透明像素转 RGB 会变黑，
    # 会被误当成"深色内容"整图保留）
    if src_rgba.mode == "RGBA":
        a0 = np.array(src_rgba.split()[3])
        if (a0 < 250).mean() > 0.01:
            return src_rgba
    W = whiteness(rgb)
    w_bg = _smoothstep(W, soft, soft + 0.30)                # 白度 -> 背景概率
    coarse = _bg_prior(W, thresh=0.80)                      # 与边缘连通的白色（可能漏）
    sealed = _bg_prior(~_close(W < soft, seal_r), thresh=0.5)
    prior = (coarse & sealed).astype(np.float32)            # 两者都到得了 = 真背景
    inside = (W > 0.80) & (prior < 0.5)                     # 人像内部白，强制留

    p_bg = np.clip(w_bg * prior, 0, 1)
    p_bg[inside] = 0.0
    alpha = 1.0 - p_bg

    # guided filter：用原图亮度把 alpha 的边缘细节还原回去
    lum = rgb.mean(axis=2)
    alpha = guided_filter(lum, alpha.astype(np.float32), r=guide_r)
    alpha = np.clip(alpha, 0, 1)

    # 形态：闭运算补内部空洞，开运算抹掉贴纸外沿的碎屑
    A = binary_closing(alpha > 0.5, structure=np.ones((close_r, close_r)))
    A = binary_opening(A, structure=np.ones((2, 2)))
    A = A | ((alpha > 0.55))

    # 闭运算区域内强制不透明（填掉发丝间的背景小洞），区域外归零
    alpha = np.where(A, np.maximum(alpha, 0.92), 0.0)
    alpha = np.clip(alpha, 0, 1)

    out = Image.fromarray((rgb * 255).astype(np.uint8), "RGB").convert("RGBA")
    out.putalpha(Image.fromarray((alpha * 255).astype(np.uint8)))
    return out


# ---------- 5) 评估：内部白保留率 / 外部白去除率 ----------
def metrics(src_rgba, out):
    """内部白 = 白像素里不与边缘连通的块（人像的一部分，必须保留）
       外部白 = 与边缘连通的白像素（必须透明）"""
    rgb = np.array(src_rgba.convert("RGB")).astype(np.float32) / 255.0
    W = whiteness(rgb)
    # 严格定义：只有"够白且不与画布边缘连通"才算人像内部白（浅灰过渡带算背景，该删）
    white = W > 0.80
    conn = _bg_prior(white, thresh=0.80)
    inside, outside = white & ~conn, white & conn

    a = np.array(out.split()[3]).astype(np.float32) / 255.0
    keep_in = a[inside].mean() * 100 if inside.any() else float("nan")
    del_bg = 1.0 - a[outside].mean() if outside.any() else float("nan")

    # 灰晕残留：亮度 180~242 的过渡带像素（白底与内容之间的抗锯齿/阴影）
    rgb8 = np.array(src_rgba.convert("RGB")).astype(np.float32) / 255.0
    lum = rgb8.mean(axis=2)
    gray_band = (lum > 0.70) & (lum < 0.95) & (outside | inside)
    gray_res = a[gray_band].mean() * 100 if gray_band.any() else float("nan")
    return dict(内部白保留=keep_in, 外部白去除率=del_bg * 100, 灰晕残留=gray_res,
                内部白像素=int(inside.sum()), 外部白像素=int(outside.sum()))

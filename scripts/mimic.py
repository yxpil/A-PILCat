# -*- coding: utf-8 -*-
"""
MimicMotion 姿势驱动动效链（真正"角色做动作"）。

架构（由权重结构确认）：
  - MimicMotion UNet = SVD UNet（1428 键完全匹配）+ pose_net（GuideEncoder）
  - pose_net: 3 通道骨架图 -> 320 通道特征，在 UNet conv_in 输出后相加
  - 复用 SVD 的 VAE / CLIP image_encoder / EulerDiscreteScheduler

流程：
  1. 贴纸 -> 白底 512x512 -> SVD VAE encode 成 image_latents
  2. 合成骨架序列（挥手/点头等，纯程序化，逐帧可控）
  3. MimicMotion UNet 去噪 25 帧
  4. VAE temporal decode -> 视频 -> GIF（反投影保 alpha）

用法:
  python mimic.py --names 打call --action wave --out mimic_out
"""
import os
import sys
import time
import argparse

sys.path.insert(0, '.')
import numpy as np
import torch
import cv2
from PIL import Image

from pack_sr import SRC
from matting_gpu import matte
from polish import sr_polish
from pack_motion import shrink_content, frames_to_gif
from line_down2 import down_line

MODEL_SVD = 'models/svd'
CKPT = 'models/mimic/MimicMotion_1-1.pth'
S = 512
NUM_FRAMES = 25
STEPS = 25
FPS = 6
GUIDANCE = 2.0

# ---------------- OpenPose BODY_18 渲染 ----------------
# 0 nose, 1 neck, 2 Rsho, 3 Relb, 4 Rwri, 5 Lsho, 6 Lelb, 7 Lwri,
# 8 Rhip, 9 Rkne, 10 Rank, 11 Lhip, 12 Lkne, 13 Lank, 14 Reye, 15 Leye,
# 16 Rear, 17 Lear
LIMBS = [(0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), (1, 8),
         (8, 9), (9, 10), (1, 11), (11, 12), (12, 13), (0, 14), (14, 16),
         (0, 15), (15, 17)]
LIMB_COLORS = [(255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
               (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85),
               (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
               (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255),
               (255, 0, 170)]
KP_COLORS = [(255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
             (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85),
             (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
             (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255),
             (255, 0, 170), (255, 0, 85)]


def render_pose(kps, size=512, r_limb=7, r_kp=5):
    """kps: (18,2) ndarray -> 黑底 RGB 骨架图。"""
    im = np.zeros((size, size, 3), np.uint8)
    for i, (a, b) in enumerate(LIMBS):
        if a >= len(kps) or b >= len(kps):
            continue
        pa, pb = tuple(kps[a].astype(int)), tuple(kps[b].astype(int))
        cv2.line(im, pa, pb, LIMB_COLORS[i], r_limb, cv2.LINE_AA)
    for i, p in enumerate(kps):
        if i >= len(KP_COLORS):
            break
        cv2.circle(im, tuple(p.astype(int)), r_kp, KP_COLORS[i], -1, cv2.LINE_AA)
    return im


def base_skeleton(bbox=None):
    """按角色实际身体位置定制的站姿骨架。
    bbox: 角色 alpha 外接框 (x0,y0,x1,y1)，按 Q 版比例（头大身短）布置关键点。
    无 bbox 时用标准人形。"""
    if bbox is None:
        return np.array([
            [256, 78], [256, 132], [296, 142], [318, 212], [330, 280],
            [216, 142], [194, 212], [182, 280], [286, 330], [292, 422],
            [294, 500], [226, 330], [220, 422], [218, 500],
            [242, 68], [270, 68], [234, 80], [278, 80],
        ], np.float32)
    x0, y0, x1, y1 = bbox
    cx = (x0 + x1) / 2
    H = y1 - y0
    W = x1 - x0
    hw = min(W, H) * 0.5

    def P(fy, fx=0.0):
        return np.array([cx + fx * hw, y0 + fy * H], np.float32)

    pts = np.stack([
        P(0.16),            # 0 nose
        P(0.27),            # 1 neck
        P(0.31, 0.42),      # 2 Rsho
        P(0.42, 0.62),      # 3 Relb
        P(0.50, 0.72),      # 4 Rwri
        P(0.31, -0.42),     # 5 Lsho
        P(0.42, -0.62),     # 6 Lelb
        P(0.50, -0.72),     # 7 Lwri
        P(0.56, 0.30),      # 8 Rhip
        P(0.74, 0.32),      # 9 Rkne
        P(0.90, 0.32),      # 10 Rank
        P(0.56, -0.30),     # 11 Lhip
        P(0.74, -0.32),     # 12 Lkne
        P(0.90, -0.32),     # 13 Lank
        P(0.13, -0.10),     # 14 Reye
        P(0.13, 0.10),      # 15 Leye
        P(0.15, -0.18),     # 16 Rear
        P(0.15, 0.18),      # 17 Lear
    ])
    return pts


def pose_wave(n=25, cycles=2.0, bbox=None, amp=30):
    """双手举高交替挥舞（打call）。amp 控制摆幅，Q 版角色用小幅度。"""
    seq = []
    for i in range(n):
        t = 2 * np.pi * cycles * i / max(1, n - 1)
        k = base_skeleton(bbox).copy()
        # 双腕举到头侧上方，绕肩摆动
        sh_r, sh_l = k[2].copy(), k[5].copy()
        head_r = np.array([k[0][0] + 0.55 * amp, k[0][1] - 0.2 * amp])
        head_l = np.array([k[0][0] - 0.55 * amp, k[0][1] - 0.2 * amp])
        k[4] = head_r + np.array([np.sin(t), np.cos(t) * 0.4]) * amp
        k[7] = head_l + np.array([-np.sin(t), -np.cos(t) * 0.4]) * amp
        k[3] = (sh_r + k[4]) / 2          # 肘取肩腕中点（自然弯曲）
        k[6] = (sh_l + k[7]) / 2
        # 头部轻摆
        k[0] += [np.sin(t + 0.8) * amp * 0.12, 0]
        k[14:18] += [np.sin(t + 0.8) * amp * 0.12, 0]
        seq.append(k)
    return seq


def pose_nod(n=25, cycles=2.0, bbox=None, amp=30):
    """点头。"""
    seq = []
    for i in range(n):
        t = 2 * np.pi * cycles * i / max(1, n - 1)
        k = base_skeleton(bbox).copy()
        dy = np.sin(t) * amp * 0.25
        k[0] += [0, dy]
        k[14:18] += [0, dy]
        seq.append(k)
    return seq


def pose_bounce(n=25, cycles=1.0, bbox=None, amp=30):
    """跳跃：整体上下。"""
    seq = []
    for i in range(n):
        t = 2 * np.pi * cycles * i / max(1, n - 1)
        k = base_skeleton(bbox).copy()
        dy = -abs(np.sin(t)) * amp
        k[:8] += [0, dy]
        k[8:14] += [0, dy * 0.3]
        seq.append(k)
    return seq


ACTIONS = {'wave': pose_wave, 'nod': pose_nod, 'bounce': pose_bounce}

# ---------------- pose_net (GuideEncoder) ----------------
class GuideEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        ch = [3, 3, 16, 16, 32, 32, 64, 64, 128]
        layers = []
        cfg = [(3, 3, 3, 1, 1), (3, 16, 4, 2, 1), (16, 16, 3, 1, 1),
               (16, 32, 4, 2, 1), (32, 32, 3, 1, 1), (32, 64, 4, 2, 1),
               (64, 64, 3, 1, 1), (64, 128, 3, 1, 1)]
        for i, (ci, co, k, s, p) in enumerate(cfg):
            layers.append(torch.nn.Conv2d(ci, co, k, s, p))
            layers.append(torch.nn.SiLU())
        self.conv_layers = torch.nn.ModuleList(layers)
        self.final_proj = torch.nn.Conv2d(128, 320, 1)
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        for m in self.conv_layers:
            x = m(x)
        return self.final_proj(x)


class Mimic:
    def __init__(self):
        from diffusers import StableVideoDiffusionPipeline

        self.device = 'cuda'
        # 官方 pipeline 负责全部条件处理（CLIP/VAE/scheduler/guidance），
        # 我们只做两件事：替换 UNet 权重 + conv_in 后注入 pose 特征
        self.pipe = StableVideoDiffusionPipeline.from_pretrained(
            MODEL_SVD, torch_dtype=torch.float16)
        self.pipe.enable_attention_slicing()
        self.pipe.enable_model_cpu_offload()

        ck = torch.load(CKPT, map_location='cpu')
        unet_sd = {k[len('unet.'):]: v for k, v in ck.items() if k.startswith('unet.')}
        missing, unexpected = self.pipe.unet.load_state_dict(unet_sd, strict=False)
        assert not missing and not unexpected, (missing[:4], unexpected[:4])

        self.pose_net = GuideEncoder()
        pg = {k[len('pose_net.'):]: v for k, v in ck.items() if k.startswith('pose_net.')}
        self.pose_net.load_state_dict(pg, strict=True)
        self.pose_net.to(self.device, torch.float16).eval()
        self.pipe.unet.conv_in.register_forward_hook(self._conv_in_hook)
        self._pose_feat = None
        torch.cuda.empty_cache()

    def _conv_in_hook(self, module, args, output):
        if self._pose_feat is not None:
            return output + self._pose_feat
        return output

    @staticmethod
    def _guidance_grid(F, dtype, device, gmin=1.0, gmax=3.0):
        """SVD 逐帧 guidance：线性 (gmax -> gmin)。"""
        gs = torch.linspace(gmax, gmin, F, device=device, dtype=dtype)
        return gs.view(1, F, 1, 1, 1)

    @torch.no_grad()
    def run(self, ref_rgb512, pose_seq, steps=STEPS, guidance=GUIDANCE,
            motion_bucket=127, noise_aug=0.02, seed=0, progress=False,
            use_pose=True):
        """ref_rgb512: PIL RGB 512; pose_seq: list of (18,2) 骨架。
        返回 list of np.uint8 RGB 帧。官方 pipeline 内部处理 fps-1 / CFG / guidance。"""
        g = torch.Generator('cpu').manual_seed(seed)
        F = len(pose_seq)
        dtype = torch.float16

        pose_imgs = [render_pose(k, S) for k in pose_seq]
        pt = torch.from_numpy(np.stack(pose_imgs)).to(self.device, dtype) \
            .permute(0, 3, 1, 2) / 255.0                            # (F,3,H,W)
        with torch.no_grad():
            pf = self.pose_net(pt) * self.pose_net.scale            # (F,320,h,w)
        # 官方 pipeline CFG 时 sample flatten 后为 (2F,320,h,w)
        self._pose_feat = pf.repeat(2, 1, 1, 1) if use_pose else None

        # height/width 必须显式给：默认 = sample_size(128)*vae_scale_factor(8)=1024，
        # 与 pose_net 在 512 下的特征分辨率不一致
        out = self.pipe(image=ref_rgb512, num_frames=F, height=S, width=S,
                        num_inference_steps=steps, motion_bucket_id=motion_bucket,
                        fps=FPS, noise_aug_strength=noise_aug,
                        generator=g, output_type='np').frames[0]
        self._pose_feat = None
        return [(f * 255).astype(np.uint8) for f in out]


def layout512(rgba, fill=0.9):
    iw, ih = rgba.size
    scale = min(S / iw, S / ih) * fill
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    ox, oy = (S - nw) // 2, (S - nh) // 2
    im = rgba.resize((nw, nh), Image.LANCZOS)
    rgb = Image.new('RGB', (S, S), (255, 255, 255))
    rgb.paste(im.convert('RGB'), (ox, oy))
    box = (ox, oy, ox + nw, oy + nh)
    return rgb, box


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--names', nargs='*', default=['打call'])
    ap.add_argument('--action', default='wave')
    ap.add_argument('--out', default='mimic_out')
    ap.add_argument('--steps', type=int, default=STEPS)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    m = Mimic()
    for kw in a.names:
        t0 = time.time()
        src = matte(Image.open(os.path.join(SRC, kw + '.png')).convert('RGBA'))
        master = sr_polish(src, size=1900, ow=0, lq_max=475, denoise=0.0)
        rgb0, box = layout512(master)
        seq = ACTIONS[a.action](bbox=box)
        frames = m.run(rgb0, seq, steps=a.steps, progress=False)
        # 骨架对照存档
        for i in [0, 12, 24]:
            Image.fromarray(pose_imgs_check(seq, i, S)).save(
                os.path.join(a.out, f'{kw}_pose{i}.png'))
        for i, f in enumerate(frames[::8][:3]):
            Image.fromarray(f).save(os.path.join(a.out, f'{kw}_mimic{i}.png'))
        np.savez_compressed(os.path.join(a.out, f'{kw}_mimicframes.npz'),
                            f=np.stack(frames))
        torch.cuda.empty_cache()
        print(f'{kw}: {time.time()-t0:.0f}s', flush=True)


def pose_imgs_check(seq, i, S):
    return render_pose(seq[i], S)


if __name__ == '__main__':
    main()

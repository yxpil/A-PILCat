# -*- coding: utf-8 -*-
"""
动效 GIF 生成：300px v3 成品 -> 缩主体留安全边 -> 形变 N 帧 -> 全局调色板多帧 GIF。
独立于 pack_final 的静态链路，便于对比/回退。
"""
import io
import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter

from motion import warp, _subject_bbox
from motion_map import motion_of
from polish import final_sharp


def extend_colors(rgba, r=7):
    """把主体颜色向透明区扩散——RGBA 直接 LANCZOS 会把透明区黑 RGB 混进边缘
    （premultiplied 污染），先外扩颜色再缩放，边缘混出的才是主体色。"""
    arr = np.asarray(rgba, dtype=np.float32)
    rgb, al = arr[..., :3], arr[..., 3] / 255.0
    wb = uniform_filter(al, r)
    ext = np.stack([uniform_filter(rgb[..., c] * al, r) / np.maximum(wb, 1e-4)
                    for c in range(3)], axis=-1)
    m = (al > 0.6)[..., None]
    out = np.concatenate([rgb * m + ext * (1 - m), arr[..., 3:]], axis=-1)
    return Image.fromarray(out.clip(0, 255).astype(np.uint8), "RGBA")


def shrink_content(rgba, factor=0.88):
    """把主体缩到 factor 并居中，给动作留出画布内安全空间。"""
    bbox = _subject_bbox(rgba)
    if bbox is None:
        return rgba
    w, h = rgba.size
    src = extend_colors(rgba)
    im2 = src.resize((max(1, int(w * factor)), max(1, int(h * factor))), Image.LANCZOS)
    canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    canvas.paste(im2, ((w - im2.width) // 2, (h - im2.height) // 2), im2)
    return canvas


def make_frames(rgba300, kind, n_frames=8, amp=0.7, sharpen=25):
    """输入 300px v3 成品，输出 n_frames 帧动效 RGBA。amp=0.7 动作含蓄耐看，
    也显著减小 GIF 体积（帧间公共像素更多）。"""
    base = shrink_content(rgba300, 0.88)
    frames = []
    for i in range(n_frames):
        f = warp(base, kind, i / n_frames, amp)
        if sharpen:
            f = final_sharp(f, sharpen, radius=0.8, thresh=2)
        frames.append(f)
    return frames


def frames_to_gif(frames, path, duration=100, limit_kb=300, colors=64):
    """多帧透明 GIF：全帧统一调色板（colors 色 + 1 透明槽），disposal=2。
    位移动画帧间公共像素少，255 色会到 300-460KB；64 色对这种平涂画风视觉无损且 ~218KB。"""
    w, h = frames[0].size

    def _encode(fs, ncol):
        flats = []
        for f in fs:
            bg = Image.new("RGB", (w, h), (0, 0, 0))
            bg.paste(f, mask=f.split()[3])
            flats.append(bg)
        strip = Image.new("RGB", (w, h * len(flats)))
        for i, fl in enumerate(flats):
            strip.paste(fl, (0, i * h))
        q = strip.quantize(colors=ncol, method=Image.FASTOCTREE,
                           dither=Image.Dither.NONE)
        pal = (q.getpalette() + [0] * 768)[:768]
        pal[765:768] = [0, 0, 0]
        ps = []
        for i, fl in enumerate(flats):
            idx = np.asarray(q.crop((0, i * h, w, (i + 1) * h)),
                             dtype=np.uint8).copy()
            idx[np.asarray(fs[i].split()[3]) <= 128] = ncol
            p = Image.fromarray(idx, mode="P")
            p.putpalette(pal)
            ps.append(p)
        buf = io.BytesIO()
        ps[0].save(buf, "GIF", save_all=True, append_images=ps[1:],
                   duration=duration, loop=0, transparency=ncol,
                   disposal=2, optimize=True)
        return buf.getvalue()

    data = _encode(frames, colors)
    if len(data) > limit_kb * 1024:                      # 降色 -> 减帧
        for c2, step in ((48, 1), (32, 2), (24, 2)):
            sub = frames[::step]
            data = _encode(sub, c2)
            if len(data) <= limit_kb * 1024:
                break
    from pack_sr import robust_write
    robust_write(path, data)
    return len(data), len(frames)

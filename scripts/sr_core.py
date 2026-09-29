# -*- coding: utf-8 -*-
"""超分核心：Real-ESRGAN x4 + alpha 保持 + 高质量合成"""
import os
import numpy as np
import torch
from PIL import Image, ImageFilter
from spandrel import ModelLoader

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "models", "RealESRGAN_x4plus.pth")
_device = None
_model = None


def load_model():
    global _model, _device
    if _model is None:
        torch.set_grad_enabled(False)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            _device = torch.device("cuda")
        else:
            _device = torch.device("cpu")
        _model = ModelLoader().load_from_file(MODEL_PATH).to(_device).eval()
        print(f"[sr] model on {_device}, scale={_model.scale}")
    return _model, _device


def _to_t(im):
    a = np.array(im).astype(np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(_device)


def _from_t(t):
    a = t.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    return Image.fromarray(np.clip(a * 255.0, 0, 255).astype(np.uint8), "RGB")


def smart_upscale(im, lq_short=None):
    """RGBA -> 模型超分 x4（保险：内部 vector_estimate 也用 lanczos 过一遍）
    lq_short: 先把长边裁到该像素再喂模型，避免超出模型 LQ 分布产生块状伪影"""
    m, dev = load_model()
    src = im
    if lq_short and max(src.width, src.height) > lq_short:
        s = lq_short / max(src.width, src.height)
        src = src.resize((max(1, round(src.width * s)), max(1, round(src.height * s))),
                         Image.LANCZOS)

    a = src.split()[3]
    flat = Image.new("RGB", src.size, (255, 255, 255))
    flat.paste(src, mask=a)

    with torch.no_grad():
        out = _from_t(m(_to_t(flat)))

    # alpha 同步放大：先放大，轻度模糊后软阈值，去掉半透明毛边
    na = a.resize(out.size, Image.LANCZOS)
    na = na.filter(ImageFilter.GaussianBlur(na.width / 900))
    na = na.point(lambda p: 255 if p > 175 else (0 if p < 85 else p))
    res = out.convert("RGBA")
    res.putalpha(na)
    return res


def fit_super(im, size, margin=6, lq_short=None, stroke_px=0):
    """trim -> 超分 -> 高质量缩到 size -> 居中贴到方形画布"""
    bb = im.split()[3].point(lambda p: 255 if p > 24 else 0).getbbox()
    src = im.crop(bb) if bb else im
    s = min((size - margin * 2) / src.width, (size - margin * 2) / src.height)
    nw, nh = max(1, round(src.width * s)), max(1, round(src.height * s))

    big = smart_upscale(src, lq_short=lq_short or max(nw, nh) * 2)
    src = big.resize((nw * 4, nh * 4), Image.LANCZOS).resize((nw, nh), Image.LANCZOS)

    if stroke_px:
        from build_shared import add_white_stroke
        src = add_white_stroke(src, px=stroke_px)
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    canvas.paste(src, ((size - nw) // 2, (size - nh) // 2), src)
    return canvas

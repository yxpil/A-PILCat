# -*- coding: utf-8 -*-
"""RRDBNet（Real-ESRGAN 架构）官方结构复刻 + 权重加载，不依赖 torchvision/spandrel"""
import os
import torch
import torch.nn as nn


class ResidualDenseBlock(nn.Module):
    """RRDB 内层：5 层 conv+BN+LeakyReLU，逐层 concat 特征，末层 0.2 衰减残差"""
    def __init__(self, nf=64, nf2=32):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, nf2, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf + nf2, nf2, 3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(nf + 2 * nf2, nf2, 3, 1, 1, bias=True)
        self.conv4 = nn.Conv2d(nf + 3 * nf2, nf2, 3, 1, 1, bias=True)
        self.conv5 = nn.Conv2d(nf + 4 * nf2, nf, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDBBlock(nn.Module):
    """外层：三个 ResidualDenseBlock 串联 + 0.2 衰减残差融合"""
    def __init__(self, nf=64, num_grow_ch=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(nf, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(nf, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(nf, num_grow_ch)

    def forward(self, x):
        return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_feat=64,
                 num_block=23, num_grow_ch=32):
        super().__init__()
        self.conv_first = nn.Conv2d(in_channels, num_feat, 3, 1, 1, bias=True)
        self.body = nn.ModuleList(
            [RRDBBlock(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        # 上采样头（x4 = 两次 x2）
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(num_feat, out_channels, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.conv_first(x)
        deep = x
        for block in self.body:
            deep = block(deep)
        out = self.conv_body(deep) + x
        out = self.lrelu(self.conv_up1(torch.nn.functional.interpolate(out, scale_factor=2)))
        out = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(out, scale_factor=2)))
        out = self.conv_hr(out)
        return self.conv_last(self.lrelu(out))


def _extract_state(ck):
    if isinstance(ck, dict):
        for key in ("params_ema", "params", "state_dict"):
            if key in ck and isinstance(ck[key], dict):
                return ck[key]
        for v in ck.values():
            if isinstance(v, dict) and any("conv" in k for k in v):
                return v
    return ck


def _cuda_works():
    """torch.cuda.is_available() 在 sm_120 等不支持的卡上会假阳性，必须真跑一次"""
    if not torch.cuda.is_available():
        return False
    try:
        with torch.no_grad():
            torch.randn(1, device="cuda").sum().item()
        return True
    except Exception:
        return False


def load_model(path=None, device=None):
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "models", "RealESRGAN_x4plus.pth")
    if device is None:
        device = torch.device("cuda" if _cuda_works() else "cpu")
    if device.type == "cpu" and not _cuda_works():
        print("[rrdb] CUDA 自检失败，回退 CPU")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = _extract_state(ck)
    sd = {k.split("module.", 1)[-1]: v for k, v in sd.items()}

    net = RRDBNet()
    net.load_state_dict(sd, strict=True)   # 严格校验，加载不上立刻报错
    n = sum(p.numel() for p in net.parameters())
    net.to(device).eval()
    print(f"[rrdb] {os.path.basename(path)}  params={n/1e6:.1f}M  -> {device}")
    return net, device

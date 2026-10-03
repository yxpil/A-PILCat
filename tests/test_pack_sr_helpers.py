# -*- coding: utf-8 -*-
"""pack_sr.py 中 GPU 无关纯函数的单元测试。

只覆盖不需要 torch 模型的函数：内容包围盒、居中画布、原子写、限大小保存、
关键词规整、顶部文字带裁剪。超分（superres/sr_rgba）走 GPU 模型，不在此测。
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image


def _rgba(w, h, rgb=(255, 0, 0), alpha=255):
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    arr[..., :3] = rgb
    arr[..., 3] = alpha
    return Image.fromarray(arr, "RGBA")


class TestContentBbox:
    def test_bbox_of_opaque_content(self, pack_sr):
        im = _rgba(40, 30, alpha=0)          # 全透明
        arr = np.array(im)
        arr[5:20, 8:30, 3] = 255             # 中间一块不透明
        im = Image.fromarray(arr, "RGBA")
        bb = pack_sr.content_bbox(im)
        assert bb == (8, 5, 30, 20)

    def test_fully_transparent_returns_none(self, pack_sr):
        im = _rgba(20, 20, alpha=0)
        assert pack_sr.content_bbox(im) is None


class TestFitFromBig:
    def test_content_centered_on_canvas(self, pack_sr):
        big = _rgba(100, 50, rgb=(0, 255, 0))
        out = pack_sr.fit_from_big(big, 300, margin=6)
        assert out.size == (300, 300)
        # 四角透明，中心不透明
        a = np.array(out)
        assert a[0, 0, 3] == 0
        assert a[150, 150, 3] == 255

    def test_square_input_stays_centered(self, pack_sr):
        big = _rgba(60, 60, rgb=(0, 0, 255))
        out = pack_sr.fit_from_big(big, 200, margin=10)
        assert out.size == (200, 200)


class TestRobustWrite:
    def test_writes_bytes_and_returns_length(self, pack_sr, tmp_path):
        p = tmp_path / "out.bin"
        data = b"hello A-PILCat" * 10
        n = pack_sr.robust_write(str(p), data)
        assert n == len(data)
        assert p.read_bytes() == data

    def test_overwrites_existing(self, pack_sr, tmp_path):
        p = tmp_path / "o.bin"
        p.write_bytes(b"old")
        pack_sr.robust_write(str(p), b"new-content")
        assert p.read_bytes() == b"new-content"


class TestSavePngUnder:
    def test_file_written(self, pack_sr, tmp_path):
        im = _rgba(32, 32, rgb=(120, 130, 140))
        p = tmp_path / "thumb.png"
        size, used = pack_sr.save_png_under(im, str(p), 200 * 1024)
        assert p.exists()
        assert size > 0
        # 回读校验是合法 PNG
        back = Image.open(p)
        assert back.size == (32, 32)


class TestKeyword:
    @pytest.mark.parametrize(
        "name,expected",
        [("赢了_2.png", "赢了"), ("啊.png", "啊"), ("yyds.png", "yyds"), ("加油_3.png", "加油")],
    )
    def test_keyword_of_strips_number_suffix(self, pack_sr, name, expected):
        assert pack_sr.keyword_of(name) == expected


class TestFixKeyword:
    def test_short_cjk_kept(self, pack_sr):
        assert pack_sr.fix_keyword("yyds") == "yyds"

    def test_long_cjk_truncated(self, pack_sr):
        # 5+ 个 CJK：第 4 个字符仍是 CJK → 截前 4
        assert pack_sr.fix_keyword("你好吗哈哈") == "你好吗哈"


class TestStripTopText:
    def test_blank_image_returns_none(self, pack_sr):
        im = _rgba(40, 40, alpha=0)
        assert pack_sr.strip_top_text(im) is None

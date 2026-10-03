# -*- coding: utf-8 -*-
"""pytest 公共配置。

A-PILCat 的 scripts/ 是 GPU 生成管线，import 时会加载 torch 超分模型
（``net, DEV = load_model()``）。本目录只测其中**不依赖 GPU 的纯函数**
（content_bbox / fit_from_big / robust_write / save_png_under / keyword_of /
fix_keyword / strip_top_text），因此在 import pack_sr 之前向 sys.modules
注入桩模块 torch 与 rrdb——被测函数根本不会触达这些桩。
"""
from __future__ import annotations

import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, "scripts")
for p in (ROOT, SCRIPTS):
    if p not in sys.path:
        sys.path.insert(0, p)

# ---- 桩：torch（仅用于通过 import，被测路径不调用它）
torch_stub = types.ModuleType("torch")
torch_stub.no_grad = lambda: (lambda f: f)
sys.modules.setdefault("torch", torch_stub)

# ---- 桩：rrdb.load_model 返回占位模型对象
rrdb_stub = types.ModuleType("rrdb")
rrdb_stub.load_model = lambda: (object(), "cpu")
sys.modules.setdefault("rrdb", rrdb_stub)

import pytest  # noqa: E402


@pytest.fixture()
def pack_sr():
    import importlib

    return importlib.import_module("pack_sr")

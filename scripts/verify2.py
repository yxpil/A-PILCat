# -*- coding: utf-8 -*-
"""校验三个产物：表情缩略图.zip(png) / 表情动画.zip(gif) / 表情封面图.png"""
import os, sys, zipfile, io
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pack_final import GIF_SIZE

base = r"C:\Users\Admin\OneDrive\Desktop\上传"
only = sys.argv[1] if len(sys.argv) > 1 else None

dirs = [d for d in sorted(os.listdir(base))
        if os.path.isdir(os.path.join(base, d)) and d != "Example"]
if only:
    dirs = [d for d in dirs if d == only]

ok = True
for d in dirs:
    pd = os.path.join(base, d)
    line = f"{d:<10}"
    # 缩略图 300x300 PNG / 表情动画 GIF 按用户要求走源图原生像素（不降色不降档）
    _gsz = GIF_SIZE
    for zname, ext, lim, esize in [("表情缩略图.zip", ".png", 200, (300, 300)),
                                   ("表情动画.zip", ".gif", 1024, (_gsz, _gsz))]:
        zp = os.path.join(pd, zname)
        if not os.path.exists(zp):
            line += f" {zname}:缺失"
            ok = False
            continue
        try:
            with zipfile.ZipFile(zp) as z:
                assert z.testzip() is None
                names = z.namelist()
                bad_ext = [n for n in names if not n.lower().endswith(ext)]
                mx = 0
                for n in names:
                    b = z.read(n)
                    mx = max(mx, len(b))
                    im = Image.open(io.BytesIO(b))
                    if im.size != esize or im.format != ext.lstrip(".").upper():
                        bad_ext.append(f"{n}尺寸/格式异常")
                line += f" {zname}:{len(names)}个 单张最大{mx/1024:.0f}KB({lim}限)"
                if bad_ext:
                    line += f" 异常{bad_ext[:2]}"
                    ok = False
        except Exception as e:
            line += f" {zname}:坏({e})"
            ok = False
    cov = os.path.join(pd, "表情封面图.png")
    if os.path.exists(cov):
        im = Image.open(cov)
        kb = os.path.getsize(cov) / 1024
        line += f" 封面{im.size[0]}x{im.size[1]} {kb:.0f}KB"
        if im.size != (200, 200) or kb > 100:
            line += " <封面不合规>"; ok = False
    else:
        line += " 封面缺失"; ok = False
    print(line)

print("\nALL OK" if ok else "\n有项目不合规")

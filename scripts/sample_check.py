# -*- coding: utf-8 -*-
"""分层抽样检查：每种动作类型抽 2 张，走 v2 全链路生成 GIF，
抽 4 帧拼成检查板（含原图对照），供目视验收。
用法: python sample_check.py [--n 2] [--out 目录]
"""
import os
import sys
import random
import collections

sys.path.insert(0, '.')
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pack_sr import SRC
from matting_gpu import matte
from pack_sr import robust_write
from polish import sr_polish
from pack_final import to_final, GIF_SIZE, FINAL_SHARPEN, OUTLINE, MASTER, LQ_MAX
from make_gif_v2 import make_frames_v2
from pack_motion import frames_to_gif
from motion_map import motion2_of


def pick(lst, n, seed=0):
    r = random.Random(seed)
    return r.sample(lst, min(n, len(lst)))


def run(names, outdir, frames_per=4):
    os.makedirs(outdir, exist_ok=True)
    rows = []
    for kw in names:
        src = matte(Image.open(os.path.join(SRC, kw + '.png')).convert('RGBA'))
        master = sr_polish(src, size=MASTER, ow=0, lq_max=LQ_MAX, denoise=0.0)
        g300 = to_final(master, GIF_SIZE, FINAL_SHARPEN, denoise=0.2, ow=OUTLINE)
        fr, kind, mouth = make_frames_v2(g300, kw)

        # GIF 落盘（体积/帧数校验）
        gp = os.path.join(outdir, f'{kw}.gif')
        ng, nf = frames_to_gif(fr, gp, duration=80, limit_kb=250, colors=64)
        kbs = os.path.getsize(gp) / 1024

        # 抽取代表性帧：0、1/3、2/3、末帧
        idxs = sorted(set([0] + [i * (nf - 1) // (frames_per - 1) for i in range(1, frames_per)]))
        slabs = []
        for i in idxs:
            f = fr[i]
            bg = Image.new('RGB', f.size, (255, 255, 255))
            bg.paste(f, mask=f.split()[3])
            slabs.append(bg)

        rows.append((kw, kind, mouth, len(slabs), kbs, nf, slabs))
        print(f'  {kw:14s} kind={kind:8s} mouth={str(mouth):10s} {nf}帧 {kbs:.0f}KB')

    # 拼板：每行一个表情，横向 4 帧
    CELL, Z, COLS = 96, 2, frames_per
    fh = CELL * Z
    W = COLS * fh
    board = Image.new('RGB', (W, len(rows) * fh + 18 * len(rows)), (238, 238, 238))
    d = ImageDraw.Draw(board)
    for r, (kw, kind, mouth, nsl, kbs, nf, slabs) in enumerate(rows):
        y0 = r * (fh + 18)
        d.text((4, y0 + 4), f'{kw}  [{kind}/{mouth}] {nf}帧 {kbs:.0f}KB', fill=(190, 20, 20))
        for c, s in enumerate(slabs):
            board.paste(s.resize((fh, fh), Image.NEAREST), (c * fh, y0 + 18))
    p = os.path.join(outdir, '_抽样检查板.png')
    board.save(p)
    print('拼板 ->', p)
    return board


def main():
    n = 2
    args = sys.argv
    if '--n' in args:
        n = int(args[args.index('--n') + 1])
    outdir = args[args.index('--out') + 1] if '--out' in args else 'sample_out'

    by_kind = collections.defaultdict(list)
    for f in sorted(os.listdir(SRC)):
        if not f.endswith('.png'):
            continue
        kw = f[:-4]
        by_kind[motion2_of(kw)[0]].append(kw)

    print('各动作类型样本数:', {k: len(v) for k, v in by_kind.items()})
    picks = []
    for k in ['bounce', 'shake', 'nod', 'hop', 'breathe', 'tilt', 'zoom', 'sway']:
        if k in by_kind:
            picks += pick(sorted(by_kind[k]), n, seed=7)

    print('抽样 %d 张:' % len(picks))
    run(picks, outdir)


if __name__ == '__main__':
    main()

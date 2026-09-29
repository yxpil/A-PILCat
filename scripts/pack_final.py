# -*- coding: utf-8 -*-
"""QQ 表情包最终打包：
每包 = 表情缩略图.zip(16 张 300x300 PNG <200KB) + 表情动画.zip(16 张 300x300 静态 GIF)
     + 表情封面图.png(200x200 <100KB 无文字)
统一走"原生感"打磨（去溢色边 + 压塑料噪 + 官方描边）。
"""
import os, io, re, sys, time, hashlib, zipfile, warnings, tempfile
warnings.filterwarnings("ignore")
import numpy as np
import torch
from PIL import Image, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pack_sr import SRC, OUT, DCALL, DCALL_FALLBACK, TMPDIR, \
    content_bbox, sr_rgba, fit_from_big, save_png_under, robust_write, \
    strip_top_text, keyword_of, fix_keyword, collect_items
from polish import sr_polish, denoise_auto, final_sharp, add_outline
from line_down2 import down_line
from matting_gpu import matte

OUTLINE = 2          # 描边宽度（缩略图 300px 空间里等效 2px）
OUTLINE_ON = True

# 平台硬性尺寸要求：表情动画 GIF 必须 300x300。
# 线稿降采样 (line_down2.down_line)：细笔触在 ~6.5x 降采样下只占 0.3px，
# 纯面积平均会把它冲淡成断续灰线 —— 这就是"线条放大不流畅"的根源。
# down_line 检测细笔触所在的块，用笔触真实颜色 + 保底 alpha 画出连续细线，
# 等价于画师在 300px 原生作画时的最小线宽习惯。
GIF_SIZE = 300
LQ_MAX = 475         # 超分输入上限（源图内容 ~1900px 的 1/4）
MASTER = LQ_MAX * 4  # 1900：超分母版
FINAL_SHARPEN = 35   # v3 线条已实，锐化从 75 降到 35 防光晕
GIF_LIMIT_KB = 512   # 仅作异常兜底，正常图不会触发降级
MOTION = True        # True: 表情动画.zip 输出语义动效 GIF（8帧64色≤300KB）；False: 静态单帧


def down_chain(rgba, size):
    """大比例降采样逐级靠近目标，避免一步 LANCZOS 在低分辨率端产生锯齿/振铃。"""
    w, h = rgba.size
    while max(w, h) > size * 2:
        w, h = max(size // 2, w // 2), max(size // 2, h // 2)
        rgba = rgba.resize((w, h), Image.LANCZOS)
    return fit_from_big(rgba, size)


# ---------- PNG -> 静态 GIF（保留透明，单帧） ----------
def to_gif(rgba, path, limit_kb=None, quality=False):
    """最高清：先试 255 色（GIF 上限，1 个槽位留给透明），只有异常体积才降档。
    默认关闭抖动：300px 尺度下 FLOYDSTEINBERG 会把色边打散成可见沙砾。"""
    if limit_kb is None:
        limit_kb = GIF_LIMIT_KB
    dither = Image.Dither.FLOYDSTEINBERG if quality else Image.Dither.NONE
    a = np.array(rgba.split()[3])
    flat = Image.new("RGB", rgba.size, (0, 0, 0))
    flat.paste(rgba, mask=rgba.split()[3])
    for colors in (255, 255, 192, 128, 64):
        q = flat.quantize(colors=colors, method=Image.FASTOCTREE, dither=dither)
        idx = np.array(q, dtype=np.uint8).copy()      # 0..colors-2
        idx[a <= 128] = colors                        # colors 槽位留给透明
        p = Image.fromarray(idx, mode="P")
        pal = list(q.getpalette())[:colors * 3] + [0, 0, 0]
        p.putpalette(pal[:768] + [0] * max(0, 768 - len(pal[:768])))
        buf = io.BytesIO()
        p.save(buf, "GIF", save_all=True, duration=0, loop=0,
               transparency=colors, optimize=True)
        if buf.tell() <= limit_kb * 1024 or colors == 64:
            robust_write(path, buf.getvalue())
            return len(buf.getvalue()), colors


def fin(src_rgba, size=300, outline=None, ow=None):
    """超分 + 打磨 + （在最终尺寸上）描边"""
    return sr_polish(src_rgba, size=size,
                     outline=OUTLINE_ON if outline is None else outline,
                     ow=OUTLINE if ow is None else ow)


def to_final(master, size, sharpen=FINAL_SHARPEN, denoise=0.2, ow=OUTLINE):
    """把超分母版逐级降到目标画布，再在【最终像素空间】做降噪 + 锐化 + 描边。
    关键：所有质量决策都在 300px 上做，只在 300px 上才看得见真正会被用户看到的东西。
    2026-09-28 定稿：8像素合一(BOX)+补alpha 替代墨量模型 v3（A/B 细节板 ab_simple/_细节对比.png）。"""
    from simple_down import box_down, alpha_fix
    im = alpha_fix(box_down(master, size))
    if denoise > 0:
        im = denoise_auto(im, denoise)
    if sharpen:
        im = final_sharp(im, sharpen, radius=1.1, thresh=2)
    if ow:
        im = add_outline(im, width=ow)
    return im


def main():
    skip = 0
    if "--from" in sys.argv:
        skip = int(sys.argv[sys.argv.index("--from") + 1])
    items = collect_items()
    packs = []
    for name in items:
        kw = fix_keyword(keyword_of(name))
        for p in packs:
            if len(p) < 16 and kw not in [x[1] for x in p]:
                p.append((name, kw))
                break
        else:
            packs.append([(name, kw)])
    full = [p for p in packs if len(p) == 16]
    only = int(sys.argv[sys.argv.index("--only") + 1]) if "--only" in sys.argv else None
    print(f"共 {len(items)} 张 -> {len(full)} 个整包"
          + (f" + 剩余 {len(packs[-1])} 张" if packs[-1] and len(packs[-1]) < 16 else "")
          + f"  [从第 {skip+1} 组开始]"
          + (f"  只跑第 {only+1} 组" if only is not None else "")
          + f"\n  GIF/缩略图={GIF_SIZE}px(平台硬要求) 超分母版={MASTER}px "
            f"锐化={FINAL_SHARPEN} 描边={OUTLINE_ON and OUTLINE}px")

    for pi, pack in enumerate(packs):
        if pi < skip or (only is not None and pi != only):
            continue
        tag = f"第{pi+1:02d}包" if len(pack) == 16 else f"剩余{len(pack)}张"
        pdir = os.path.join(OUT, tag)
        os.makedirs(pdir, exist_ok=True)
        print(f"\n== {tag}: {len(pack)} 个 ==")
        t0 = time.time()

        pngs, gifs = [], []
        cover_src = None
        for i, (name, kw) in enumerate(pack, 1):
            src = matte(Image.open(os.path.join(SRC, name)).convert("RGBA"))   # GPU 抠图去白底

            # 超分只做一次到母版（1900），GIF / 缩略图 / 封面全部从这一份派生
            master = sr_polish(src, size=MASTER, ow=0, lq_max=LQ_MAX, denoise=0.0)

            # GIF 与缩略图同源同尺寸：都从母版降到 300px，再在 300px 上锐化 + 描边
            g300 = to_final(master, GIF_SIZE, FINAL_SHARPEN, denoise=0.2, ow=OUTLINE)
            tp = os.path.join(TMPDIR, f"f{pi}_{i}.png")
            n, u = save_png_under(g300, tp, 200 * 1024)
            pngs.append((tp, f"{kw}_{i:02d}.png", n))

            gp = os.path.join(TMPDIR, f"f{pi}_{i}.gif")
            if MOTION:
                # v2 动效链路：语义动作（位移 + 口型），动作贴合表情本身
                from pack_motion import frames_to_gif
                from make_gif_v2 import make_frames_v2
                fr, kind, mouth = make_frames_v2(g300, kw)
                ng, nf = frames_to_gif(fr, gp)
                gifs.append((gp, f"{kw}_{i:02d}.gif", ng))
                mtag = f"+{mouth}" if mouth else ""
                print(f"   {kw}_{i:02d} png{n/1024:.0f}/gif{ng/1024:.0f}({kind}{mtag},{nf}帧) ", end="")
            else:
                ng, gu = to_gif(g300, gp)
                gifs.append((gp, f"{kw}_{i:02d}.gif", ng))
                print(f"   {kw}_{i:02d} png{n/1024:.0f}/gif{ng/1024:.0f}({gu}) ", end="")

            # 封面候选：无文字版打call > 能裁掉文字带的图
            if name == "打call.png" and os.path.exists(DCALL):
                cover_src = Image.open(DCALL).convert("RGBA")
                print(f"[封面=打call无文字版]", end="")
            elif cover_src is None:
                cand = strip_top_text(src)
                if cand is not None:
                    cb = content_bbox(cand)
                    if cb:
                        # cb 是原图坐标系，必须换算到母版坐标系再裁，否则封面会歪
                        k = MASTER / max(src.width, src.height)
                        cb_m = tuple(round(v * k) for v in cb)
                        cw, ch = cb_m[2] - cb_m[0], cb_m[3] - cb_m[1]
                        s = min((MASTER - 8) / cw, (MASTER - 8) / ch)   # 母版内留 8px 边距
                        nw = max(1, round(cw * s)); nh = max(1, round(ch * s))
                        cb_m = (round(cb_m[0] + (cw - nw) / 2), round(cb_m[1] + (ch - nh) / 2),
                                round(cb_m[0] + (cw + nw) / 2), round(cb_m[1] + (ch + nh) / 2))
                        cover_src = to_final(master.crop(cb_m), 200, ow=OUTLINE)
                    if cover_src is not None:
                        print(f"[封面候选={kw}]", end="")
        print(f"\n   耗时 {time.time()-t0:.0f}s")

        # 封面：优先无文字打call，其次裁掉顶部文字带，兜底裁上部 35%
        if cover_src is None:
            base = Image.open(DCALL_FALLBACK).convert("RGBA")
            cand = strip_top_text(base)
            if cand is not None:
                base = cand
                print("   封面: 打call 裁顶部文字带")
            else:
                bb = content_bbox(base)
                crop = base.crop((bb[0], bb[1] + int((bb[3]-bb[1])*0.35), bb[2], bb[3]))
                base = crop if content_bbox(crop) else base
                print("   封面: 打call 裁掉上部35%兜底")
            cb = content_bbox(base)
            cover_src = fin(base.crop(cb), 200) if cb else fin(base, 200)

        cp = os.path.join(pdir, "表情封面图.png")
        n, u = save_png_under(to_final(cover_src, 200, ow=OUTLINE), cp, 100 * 1024)
        print(f"   表情封面图.png {n/1024:.1f}KB({u})")

        # 表情缩略图.zip / 动画 zip（先在内存打包，再整体落盘，避免被 OneDrive 占用打断）
        buf1, buf2 = io.BytesIO(), io.BytesIO()
        with zipfile.ZipFile(buf1, "w", zipfile.ZIP_DEFLATED) as z:
            for tmp, final, n in pngs:
                z.writestr(final, open(tmp, "rb").read())
        with zipfile.ZipFile(buf2, "w", zipfile.ZIP_DEFLATED) as z:
            for tmp, final, n in gifs:
                z.writestr(final, open(tmp, "rb").read())
        robust_write(os.path.join(pdir, "表情缩略图.zip"), buf1.getvalue())
        robust_write(os.path.join(pdir, "表情动画.zip"), buf2.getvalue())
        print(f"   缩略图.zip {len(pngs)}个 {len(buf1.getvalue())//1024}KB | "
              f"动画.zip {len(gifs)}个 {len(buf2.getvalue())//1024}KB")

    print("\nALL DONE")


if __name__ == "__main__":
    main()

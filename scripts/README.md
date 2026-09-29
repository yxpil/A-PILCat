# 生成管线脚本

本目录收录这套表情包从「白底原图」到「平台三件套」的**全部**生成代码，
包括最终采用的方案、中间被否决的实验（完整留痕，方便对比调参思路）。

> 脚本里保留了项目当时的绝对路径（`C:\Users\Admin\OneDrive\Desktop\合格` 等），
> 复现时按需改文件顶部的 `SRC` / `OUT` 常量即可。

## 环境依赖

| 依赖 | 用途 |
|---|---|
| Python 3.11+ / PyTorch (CUDA) | GPU 抠图、扩散模型重绘、动效 |
| diffusers 0.40 | SD1.5 + ControlNet img2img 管线 |
| controlnet_aux | lineart_anime 线稿条件图提取 |
| rembg / onnxruntime | 人物分割 |
| opencv-python-headless==4.10.0.84 | 动漫脸检测（5.x 无 CascadeClassifier） |
| Pillow / numpy / scipy | 图像处理与统计 |

## 终版流程（当前产线）

```
合格/*.png (155 张 1700~2000px 白底)
   │
   ├─ matting_gpu.py      GPU 软 alpha 抠图：白度场×连通先验 + guided filter 细边
   │                      （不是二值白度切割，避免灰晕粘连/削掉白描边）
   │
   ├─ prompt_map.py       154 个贴纸逐条专属重绘提示词（语义级，非通用套话）
   │
   ├─ redraw3.py          SD1.5 + ControlNet lineart_anime img2img 真重绘
   │                      · lineart 锁原线稿结构 -> 构图/姿势不变
   │                      · 每张贴纸 3 个 seed -> 3 张候选
   │                      · --work 1024 --strength 0.45 --sharp 30
   │
   └─ pack3.py            3 张重绘 -> 逐张抠图 -> 统一画布 -> 3 帧 GIF 三件套
                          · 表情动画.zip   300x300 GIF, 3 帧, 白色 3px 描边,
                            透明底, 命名 关键词_01.gif, 单张 <=250KB
                          · 表情缩略图.zip 300x300 PNG(第1帧), <=200KB, 与 GIF 一一对应
                          · 表情封面图.png 200x200, 无文字, <=100KB
```

终版产物对应仓库 `assets/deliver/`（zip 原样）与 `preview/deliver/`（导出 GIF 可直接预览）。

## 上一代方案（已被重绘版取代）

| 脚本 | 思路 | 为何被换掉 |
|---|---|---|
| `pack_sr.py` + `rrdb.py` + `sr_core.py` | RealESRGAN 超分出静态三件套 | 超分对高清源图是负收益（lap_var 比值 0.95~0.99） |
| `simple_down.py` + `pack_final.py` | 8 像素合一 BOX 降采样 + 头部动效 | 线条最干净，但动画是「形变」不是「重绘」 |
| `head_motion.py` | 动漫脸检测 + 头部绕颈部旋转 8 帧动效 | 同上，属几何形变 |
| `pack_hd.py` + `pack_motion.py` | 统一 1900×1900 高清画布 | 用户要求「真重绘，不是缩放变形」 |
| `line_down.py` / `line_down2.py` / `line_tune.py` | 墨量模型降采样 v1~v3 | 直接原图输入下回染黑噪块（`ab_simple` 有 A/B 证据） |
| `svd2.py` / `svd_motion.py` | SVD 出运动场 + 光流反投影 | 分区位移统计证明 ≈ 纯刚体缩放 |
| `mimic.py` | MimicMotion 姿势驱动 | 真人舞蹈模型对 Q 版角色全线崩坏 |
| `ai_motion.py` / `motion.py` / `motion2.py` | 各类动效实验 | 同上 |

## 质检脚本

- `verify.py` / `verify2.py` / `verify3.py` —— 三件套合规校验（尺寸/体积/命名/zip UTF-8/白描边连续性）
- `verify_hd.py` —— 1900 高清版几何自检（不切头/不贴边/封面无字）
- `quality_check.py` / `geom_test.py` / `probe_bbox.py` —— 几何与统计诊断
- `ab_simple.py` / `ab_cmp.py` / `_rd_cmp.py` —— A/B 对比板（调参证据）

## 关键踩坑（留档）

1. RGBA 直接 LANCZOS 会把透明区黑 RGB 混进边缘（premultiplied 污染）→ 先 `extend_colors`
2. numpy 分块必须先 `(nh,k,nw,k)` 再 transpose，直接 reshape 会切竖带
3. diffusers ControlNet img2img 必须显式传 `width/height=init.size`，否则条件图被降到 512 报张量不匹配
4. `pip install controlnet_aux` 会把 torch 换成 CPU 版 → 装完必验 `torch.cuda.is_available()`
5. 3 帧动画绝对不能各自 trim 再 fit（内容位置略异会抖），必须共用同一画布坐标系
6. 关键词重名（`加油/然后呢/笑死` 各有 `_2`）按关键词命名产物会互相覆盖
7. GitHub zip 中文文件名必须带 UTF-8 flag（Python `zipfile` 默认即 UTF-8）
8. 描边：`ImageFilter.MaxFilter` 半径 3 对 alpha 膨胀即得到精确 3px 外扩白描边

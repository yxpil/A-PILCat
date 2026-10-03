# A-PILCat 测试说明
- 测试完成：是（2026-10-04）
- 测试日期：2026-10-04
- 测试内容：单元测试覆盖 `scripts/pack_sr.py` 的 GPU 无关纯函数（content_bbox / fit_from_big / robust_write / save_png_under / keyword_of / fix_keyword / strip_top_text），含主路径与边界（全透明图返回 None、大图等比缩放居中、原子写覆盖、PNG 限体积回读、编号尾缀剥离、CJK 截断、空白图返回 None）。本仓库无 HTTP/subprocess/SQL 注入面、无钩子/插件机制，故无对应注入/钩子用例。
- 运行命令：python -m pytest tests/ -v
- 测试框架：pytest
- 模型：豆包（Doubao）生成

## 仓库性质

本仓库主体是**表情包素材**（PNG/WebP/GIF），`scripts/` 是从白底原图到
「平台三件套」的一次性 GPU 生成管线，依赖 PyTorch(CUDA)、diffusers、
rembg/onnxruntime、opencv-headless、scipy，且脚本内写死了作者机器上的绝对路径
（`C:\Users\Admin\OneDrive\Desktop\合格`）。这些管线无法在 CI / 本机以无 GPU 方式复现。

因此本测试只针对 `scripts/pack_sr.py` 中**与 GPU 无关的纯函数**，在
`tests/conftest.py` 里用桩模块屏蔽 `torch` 与 `rrdb`（模型加载），被测函数
本身不会触达这些桩。

## 运行方式

```powershell
pip install numpy Pillow pytest
python -m pytest tests/ -v
```

- 预期：**14 passed**（Python 3.14 / Windows）。
- 不需要 GPU、不需要下载模型、不需要原始素材目录。

## 覆盖了什么

| 函数 | 测什么 |
| --- | --- |
| `content_bbox(im)` | 不透明内容包围盒；全透明图返回 None |
| `fit_from_big(big,size,margin)` | 大图等比缩到 size 画布并居中、四角透明 |
| `robust_write(path,data)` | 原子写入、返回字节数、覆盖已有文件 |
| `save_png_under(im,path,limit)` | 限体积导出 PNG、回读尺寸正确 |
| `keyword_of(name)` | 文件名去掉 `_2/_3` 编号尾缀 |
| `fix_keyword(k)` | CJK 关键词长度截断规则 |
| `strip_top_text(im)` | 空白图返回 None（文字带裁剪的边界） |

## 注入与钩子说明

本仓库**不存在**不可信输入面：脚本读取的是作者本机素材目录，无网络/无 HTTP、
无 subprocess/shell、无 SQL、无模板拼接；也没有插件/事件/回调（钩子）机制。
因此没有专门的路径穿越 / 命令注入 / 钩子隔离用例——被测函数只处理本地图像
数组与字符串，属纯函数数学逻辑。GPU 超分与 SD 重绘管线不在可构建范围内。

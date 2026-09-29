import os, zipfile, sys
from PIL import Image

base = r"C:\Users\Admin\OneDrive\Desktop\上传"
only = sys.argv[1] if len(sys.argv) > 1 else None

dirs = []
for d in sorted(os.listdir(base)):
    pd = os.path.join(base, d)
    if not os.path.isdir(pd):
        continue
    if d == "Example":      # 样板目录，跳过
        continue
    if only and d != only:
        continue
    dirs.append(pd)

ok = True
for pd in dirs:
    zname = None
    if os.path.exists(os.path.join(pd, "表情缩略图.zip")):
        zname = os.path.join(pd, "表情缩略图.zip")
    cov = os.path.join(pd, "表情封面图.png")

    n = None
    bad = None
    if zname:
        try:
            with zipfile.ZipFile(zname) as z:
                bad = z.testzip()
                n = len(z.namelist())
                sizes = [round(i.file_size / 1024, 1) for i in z.infolist()]
        except Exception as e:
            ok = False
            print(f"{os.path.basename(pd)}: ZIP 打开失败 {e}")
            continue
        if bad is not None or n not in (10, 16):
            ok = False

    line = f"{os.path.basename(pd):<10} zip={n if n is not None else '-':>3} "
    line += f"maximg={max(sizes) if sizes else '-':>7}KB corrupt={bad}  "

    if os.path.exists(cov):
        im = Image.open(cov)
        w, h = im.size
        kb = os.path.getsize(cov) / 1024
        a = im.split()[3]
        bbox = a.getbbox()
        r = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) / (w * h) if bbox else 0
        line += f"封面 {w}x{h} {kb:.1f}KB 内容占比{r*100:.0f}%"
        if w != 200 or h != 200 or kb > 100:
            ok = False
            line += "  <-- 不合规"
    else:
        line += "封面: 缺失"
        ok = False
    print(line)

print("\nALL OK" if ok else "\n有项目不合规")

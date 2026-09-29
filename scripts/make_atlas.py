"""从 合格 文件夹生成图集（contact sheet），带序号标签，方便挑选。"""
import os, hashlib
from PIL import Image, ImageDraw

SRC = r"C:\Users\Admin\OneDrive\Desktop\合格"
OUT = r"C:\Users\Admin\上传_图集"
os.makedirs(OUT, exist_ok=True)

CELL = 200
COLS = 6
ROWS = 8

seen = {}
items = []
for name in sorted(os.listdir(SRC)):
    if not name.lower().endswith(".png"):
        continue
    h = hashlib.md5(open(os.path.join(SRC, name), "rb").read()).hexdigest()
    if h in seen:
        print("dup skip:", name, "==", seen[h])
        continue
    seen[h] = name
    items.append(name)

print("total unique:", len(items))

def make_sheet(batch, idx0, path):
    rows = (len(batch) + COLS - 1) // COLS
    sheet = Image.new("RGB", (CELL * COLS, CELL * rows), (245, 245, 245))
    d = ImageDraw.Draw(sheet)
    for i, name in enumerate(batch):
        im = Image.open(os.path.join(SRC, name)).convert("RGBA")
        im.thumbnail((CELL - 10, CELL - 10), Image.LANCZOS)
        x = (i % COLS) * CELL
        y = (i // COLS) * CELL
        sheet.paste(im, (x + 5, y + 5), im)
        d.rectangle([x + 2, y + 2, x + CELL - 3, y + CELL - 3], outline=(190, 190, 190))
        d.text((x + 8, y + 8), str(idx0 + i), fill=(255, 0, 0))
    sheet.save(path)
    print("sheet:", path, len(batch))

batch = 1
i = 0
while i < len(items):
    chunk = items[i:i + COLS * ROWS]
    make_sheet(chunk, i, os.path.join(OUT, f"sheet_{i // (COLS*ROWS) + 1:02d}.jpg"))
    i += len(chunk)

with open(os.path.join(OUT, "清单.txt"), "w", encoding="utf-8") as f:
    for idx, name in enumerate(items):
        f.write(f"{idx}\t{name}\n")
print("done")

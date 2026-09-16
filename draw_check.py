import sys, os, glob
from PIL import Image, ImageDraw

labels_dir, images_dir, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out_dir, exist_ok=True)
names = open(glob.glob(os.path.join(labels_dir, "**", "obj.names"), recursive=True)[0]).read().split()
labels = {os.path.splitext(os.path.basename(p))[0]: p
          for p in glob.glob(os.path.join(labels_dir, "**", "*.txt"), recursive=True)}

for img_path in glob.glob(os.path.join(images_dir, "*")):
    stem, ext = os.path.splitext(os.path.basename(img_path))
    if ext.lower() not in (".jpg", ".jpeg", ".png") or stem not in labels:
        continue
    img = Image.open(img_path).convert("RGB"); W, H = img.size
    d = ImageDraw.Draw(img); n = 0
    for line in open(labels[stem]):
        p = line.split()
        if len(p) != 5: continue
        c, cx, cy, bw, bh = int(p[0]), *map(float, p[1:])
        x1, y1, x2, y2 = (cx - bw/2) * W, (cy - bh/2) * H, (cx + bw/2) * W, (cy + bh/2) * H
        d.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
        d.text((x1 + 3, y1 + 2), names[c], fill=(255, 255, 0))
        n += 1
    img.save(os.path.join(out_dir, stem + "_bbox.jpg"))
    print(f"{stem}: {n} caixas")

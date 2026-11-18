import json, sys, os, math
from collections import defaultdict
coco = json.load(open(sys.argv[1])); out = sys.argv[2]
os.makedirs(out, exist_ok=True)
cats = sorted(coco["categories"], key=lambda c: c["id"])
idx = {c["id"]: i for i, c in enumerate(cats)}
open(os.path.join(out, "obj.names"), "w").write("\n".join(c["name"] for c in cats) + "\n")
anns = defaultdict(list)
for a in coco["annotations"]:
    anns[a["image_id"]].append(a)
for img in coco["images"]:
    W, H = img["width"], img["height"]
    lines = []
    for a in anns[img["id"]]:
        x, y, w, h = a["bbox"]
        rot = (a.get("attributes") or {}).get("rotation", 0) or 0
        if rot and not isinstance(a.get("segmentation"), dict):
            # retangulo rotacionado do CVAT: caixa alinhada aos eixos que envolve o retangulo girado no centro
            t = math.radians(rot); cx0, cy0 = x + w/2, y + h/2
            w, h = w*abs(math.cos(t)) + h*abs(math.sin(t)), w*abs(math.sin(t)) + h*abs(math.cos(t))
            x, y = cx0 - w/2, cy0 - h/2
        x1, y1 = max(x, 0), max(y, 0)
        x2, y2 = min(x + w, W), min(y + h, H)
        if x2 <= x1 or y2 <= y1: continue
        cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
        lines.append(f"{idx[a['category_id']]} {cx:.6f} {cy:.6f} {(x2-x1)/W:.6f} {(y2-y1)/H:.6f}")
    name = os.path.splitext(os.path.basename(img["file_name"]))[0] + ".txt"
    open(os.path.join(out, name), "w").write("\n".join(lines) + ("\n" if lines else ""))

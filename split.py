import sys, os, glob, shutil, csv, random

# Separa as imagens ja classificadas em train/, val/ e test/, no formato da YOLO de classificacao:
#   <out_dir>/train/<classe>/..., <out_dir>/val/<classe>/..., <out_dir>/test/<classe>/...
# A separacao e estratificada: cada classe e dividida na mesma proporcao.
# Uso: python split.py <classification_dir> <out_dir> [val] [test] [seed]
#   ex.: python split.py classification dataset_cls 0.15 0.15 0
EXTENSOES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
SPLITS = ("train", "val", "test")

src_dir, out_dir = sys.argv[1], sys.argv[2]
val_frac = float(sys.argv[3]) if len(sys.argv) > 3 else 0.15
test_frac = float(sys.argv[4]) if len(sys.argv) > 4 else 0.15
seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0

classes = sorted(d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d)))
rng = random.Random(seed)

def tamanhos(n):
    # Garante ao menos 1 imagem no treino; val e test so recebem imagem se sobrar
    n_val = max(1, round(n * val_frac)) if n >= 2 else 0
    n_test = max(1, round(n * test_frac)) if n >= 3 else 0
    return n - n_val - n_test, n_val, n_test

# Recria a pasta de saida do zero para nao misturar com uma separacao antiga.
# Todas as classes existem em todos os conjuntos, mesmo vazias, para a YOLO ver as mesmas classes.
shutil.rmtree(out_dir, ignore_errors=True)
for s in SPLITS:
    for c in classes:
        os.makedirs(os.path.join(out_dir, s, c))

rows = []
for c in classes:
    imgs = sorted(p for p in glob.glob(os.path.join(src_dir, c, "*")) if p.lower().endswith(EXTENSOES))
    rng.shuffle(imgs)
    n_train, n_val, _ = tamanhos(len(imgs))
    partes = {"train": imgs[:n_train], "val": imgs[n_train:n_train + n_val], "test": imgs[n_train + n_val:]}
    for s, lista in partes.items():
        for p in lista:
            shutil.copy2(p, os.path.join(out_dir, s, c, os.path.basename(p)))
            rows.append((os.path.basename(p), c, s))
        if not lista:
            print(f"AVISO: {c} ficou sem imagens em {s}/ ({len(imgs)} imagens na classe)")

with open(os.path.join(out_dir, "split.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["imagem", "classe", "conjunto"])
    w.writerows(sorted(rows, key=lambda r: (SPLITS.index(r[2]), r[1], r[0])))

print(f"\n{'classe':16s}" + "".join(f"{s:>7s}" for s in SPLITS) + f"{'total':>7s}")
for c in classes + ["total"]:
    cont = [sum(r[2] == s and c in (r[1], "total") for r in rows) for s in SPLITS]
    print(f"{c:16s}" + "".join(f"{n:7d}" for n in cont) + f"{sum(cont):7d}")

import sys, os, glob, shutil, csv

# Classifica cada imagem pela contagem de carros (classe "carro") nas anotacoes YOLO.
# Motos nao contam: ficam em area propria e nao ocupam vaga de carro.
# Uso: python classify.py <labels_dir> <images_dir> <out_dir>
BAIXA_MAX = 24   # <= 24 carros -> baixa-ocupacao
MEIO_MAX = 35    # 25..35 -> meio-termo; >= 36 -> alta-ocupacao
CLASSES = ("baixa-ocupacao", "meio-termo", "alta-ocupacao")

def classe(n):
    if n <= BAIXA_MAX: return "baixa-ocupacao"
    if n <= MEIO_MAX: return "meio-termo"
    return "alta-ocupacao"

labels_dir, images_dir, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
names = open(glob.glob(os.path.join(labels_dir, "**", "obj.names"), recursive=True)[0]).read().split()
carro = names.index("carro")
labels = {os.path.splitext(os.path.basename(p))[0]: p
          for p in glob.glob(os.path.join(labels_dir, "**", "*.txt"), recursive=True)}

# Recria as pastas de classe do zero para nao sobrar imagem de uma classificacao antiga
for c in CLASSES:
    shutil.rmtree(os.path.join(out_dir, c), ignore_errors=True)
    os.makedirs(os.path.join(out_dir, c))

rows = []
for img_path in sorted(glob.glob(os.path.join(images_dir, "*"))):
    stem, ext = os.path.splitext(os.path.basename(img_path))
    if ext.lower() not in (".jpg", ".jpeg", ".png"):
        continue
    if stem not in labels:
        print(f"{stem}: sem anotacao, ignorada")
        continue
    ids = [int(l.split()[0]) for l in open(labels[stem]) if len(l.split()) == 5]
    n_carros = ids.count(carro)
    c = classe(n_carros)
    shutil.copy2(img_path, os.path.join(out_dir, c, os.path.basename(img_path)))
    rows.append((os.path.basename(img_path), n_carros, len(ids) - n_carros, c))
    print(f"{stem}: {n_carros} carros -> {c}")

with open(os.path.join(out_dir, "classificacao.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["imagem", "carros", "outros_veiculos", "classe"])
    w.writerows(sorted(rows, key=lambda r: r[1]))

print()
for c in CLASSES:
    print(f"{c}: {sum(r[3] == c for r in rows)} imagens")

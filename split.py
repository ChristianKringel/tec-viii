import sys, os, re, glob, shutil, csv, random
from datetime import datetime, timedelta

# Separa as imagens ja classificadas em train/, val/ e test/, no formato da YOLO de classificacao:
#   <out_dir>/train/<classe>/..., <out_dir>/val/<classe>/..., <out_dir>/test/<classe>/...
# - Fotos reais sao separadas por sequencia: fotos tiradas com menos de 20 s de diferenca mostram os
#   mesmos carros e ficam sempre no mesmo conjunto (senao a nota de val/test sai inflada).
# - val/ e test/ so tem fotos reais. Imagens sinteticas (prefixo "gerada_") so vao para o treino:
#   as de inpainting ("gerada_<foto de origem>_<baixa|meio>[_v<n>]") so se a foto de origem estiver no treino;
#   as geradas por IA sempre (decisao do grupo); quando a foto de origem (extras/origem_ia.csv) ficou em
#   val/test, o split.csv registra isso na coluna "obs" para a avaliacao separar essas fotos.
# - A divisao das sequencias e sorteada varias vezes e fica a mais proxima das proporcoes pedidas em
#   cada classe. Toda classe tem ao menos 1 foto real no teste; com uma unica foto real, ela vai para o teste.
# Uso: python split.py <classification_dir> <out_dir> [val] [test] [seed]
#   ex.: python split.py classification dataset_cls 0.25 0.25 0
EXTENSOES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
SPLITS = ("train", "val", "test")
GAP = timedelta(seconds=20)
TENTATIVAS = 2000

src_dir, out_dir = sys.argv[1], sys.argv[2]
val_frac = float(sys.argv[3]) if len(sys.argv) > 3 else 0.15
test_frac = float(sys.argv[4]) if len(sys.argv) > 4 else 0.15
seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0

def quando(nome):
    m = re.search(r"(\d{4})-(\d{2})-(\d{2}) at (\d{2})\.(\d{2})\.(\d{2})", nome)
    if m:
        return datetime(*map(int, m.groups()))
    m = re.search(r"(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})", nome)
    return datetime(*map(int, m.groups())) if m else None

classes = sorted(d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d)))
reais, sinteticas = [], []
for c in classes:
    for p in sorted(glob.glob(os.path.join(src_dir, c, "*"))):
        if p.lower().endswith(EXTENSOES):
            (sinteticas if os.path.basename(p).startswith("gerada_") else reais).append((p, c))

# Sequencias de fotos reais; foto sem data no nome vira uma sequencia sozinha
sem_data = [r for r in reais if quando(os.path.basename(r[0])) is None]
com_data = sorted((r for r in reais if r not in sem_data), key=lambda r: quando(os.path.basename(r[0])))
grupos, ultimo = [], None
for r in com_data:
    t = quando(os.path.basename(r[0]))
    if ultimo is None or t - ultimo > GAP:
        grupos.append([])
    grupos[-1].append(r)
    ultimo = t
grupos += [[r] for r in sem_data]

# Sorteia a ordem das sequencias e enche val e test ate a proporcao pedida; fica com o sorteio cujas
# contagens por classe ficam mais perto do alvo
rng = random.Random(seed)
total = {c: sum(r[1] == c for r in reais) for c in classes}
# Toda classe tem pelo menos 1 foto real no teste; classe com uma unica foto real fica sem val
alvo = {"test": {c: max(total[c] * test_frac, 1) for c in classes},
        "val": {c: max(total[c] * val_frac, 1) if total[c] > 1 else 0 for c in classes}}
melhor = None
for _ in range(TENTATIVAS):
    ordem = grupos[:]
    rng.shuffle(ordem)
    cont = {s: {c: 0 for c in classes} for s in SPLITS}
    destino = {}
    for g in ordem:
        s = "train"
        for cand in ("test", "val"):
            if all(cont[cand][c] + sum(r[1] == c for r in g) <= alvo[cand][c] for c in classes):
                s = cand
                break
        for p, c in g:
            destino[p] = s
            cont[s][c] += 1
    erro = sum(abs(cont[s][c] - alvo[s][c]) for s in ("val", "test") for c in classes)
    if melhor is None or erro < melhor[0]:
        melhor = (erro, destino)
destino = melhor[1]

# Reais no conjunto sorteado; sinteticas so no treino
no_treino = {os.path.splitext(os.path.basename(p))[0] for p, s in destino.items() if s == "train"}
shutil.rmtree(out_dir, ignore_errors=True)
for s in SPLITS:
    for c in classes:
        os.makedirs(os.path.join(out_dir, s, c))
rows = []
for p, c in reais:
    shutil.copy2(p, os.path.join(out_dir, destino[p], c, os.path.basename(p)))
    rows.append((os.path.basename(p), c, destino[p], "real", ""))
nomes_reais = {os.path.splitext(os.path.basename(p))[0] for p, _ in reais}
arq_ia = os.path.join(os.path.dirname(os.path.abspath(__file__)), "extras", "origem_ia.csv")
origem_ia = {}
for r in (csv.DictReader(open(arq_ia)) if os.path.exists(arq_ia) else []):
    origem_ia.setdefault(r["imagem"], []).append(os.path.splitext(r["origem"])[0])
for p, c in sinteticas:
    m = re.match(r"gerada_(.*)_(baixa|meio)(_v\d+)?$", os.path.splitext(os.path.basename(p))[0])
    if m and m.group(1) in nomes_reais and m.group(1) not in no_treino:
        rows.append((os.path.basename(p), c, "descartada", "sintetica", ""))   # inpainting com origem em val/test
        continue
    fora = [o for o in origem_ia.get(os.path.basename(p), []) if o not in no_treino]
    shutil.copy2(p, os.path.join(out_dir, "train", c, os.path.basename(p)))
    rows.append((os.path.basename(p), c, "train", "sintetica", "origem fora do treino: " + "; ".join(fora) if fora else ""))

with open(os.path.join(out_dir, "split.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["imagem", "classe", "conjunto", "tipo", "obs"])
    w.writerows(sorted(rows, key=lambda r: (r[2], r[1], r[0])))

print(f"{len(reais)} fotos reais em {len(grupos)} sequencias, {len(sinteticas)} sinteticas")
print(f"\n{'classe':16s}" + "".join(f"{s:>7s}" for s in SPLITS) + f"{'descart.':>9s}")
for c in classes:
    cont = [sum(r[1] == c and r[2] == s for r in rows) for s in SPLITS + ("descartada",)]
    print(f"{c:16s}" + "".join(f"{n:7d}" for n in cont[:3]) + f"{cont[3]:9d}")
    reais_c = [sum(r[1] == c and r[2] == s and r[3] == "real" for r in rows) for s in SPLITS]
    print(f"{'  (reais)':16s}" + "".join(f"{n:7d}" for n in reais_c))
for s in ("val", "test"):
    for c in classes:
        if not any(r[1] == c and r[2] == s for r in rows):
            print(f"AVISO: {c} ficou sem imagens em {s}/")

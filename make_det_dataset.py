import sys, os, re, shutil, csv, random, argparse
from datetime import datetime, timedelta

# Monta o dataset de DETECCAO no formato da YOLO a partir de annotations/ (rotulos .txt do coco2yolo.py
# e imagens em annotations/imagens/):
#   <out>/images/{train,val,test}/...  <out>/labels/{train,val,test}/...  <out>/data.yaml  <out>/split.csv
# - Remove caixas duplicadas (mesma linha repetida no .txt).
# - Exclui imagens da lista EXCLUIR (rotulo vazio numa foto que tem carros = anotacao faltando).
# - Remapeia as classes: so ficam as de --classes, renumeradas a partir de 0; caixas das outras saem.
# - Mesmas regras do split.py: fotos reais separadas por sequencia (< 20 s de diferenca = mesmo grupo),
#   val/ e test/ so com fotos reais, sinteticas ("gerada_") so no treino, e inpainting cuja foto de origem
#   ficou em val/test e descartado.
# Uso: python make_det_dataset.py [--src annotations] [--out dataset_det] [--classes carro,moto]
#                                 [--val 0.2] [--test 0.2] [--seed 0]
#   --classes carro,moto      -> 2 classes (padrao: as que tem exemplos)
#   --classes carro           -> 1 classe; use --merge moto=carro para juntar moto em carro
EXTENSOES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
SPLITS = ("train", "val", "test")
GAP = timedelta(seconds=20)
TENTATIVAS = 2000
EXCLUIR = {
    "WhatsApp Image 2026-09-15 at 08.04.24",   # rotulo vazio, mas a foto tem dezenas de carros
}

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="annotations")
ap.add_argument("--out", default="dataset_det")
ap.add_argument("--classes", default="carro,moto")
ap.add_argument("--merge", default="", help="ex.: moto=carro,suv_picape=carro")
ap.add_argument("--val", type=float, default=0.2)
ap.add_argument("--test", type=float, default=0.2)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

nomes_orig = [l.strip() for l in open(os.path.join(a.src, "obj.names")) if l.strip()]
novas = [c.strip() for c in a.classes.split(",") if c.strip()]
juntar = dict(m.split("=") for m in a.merge.split(",") if m.strip())
for c in novas + list(juntar) + list(juntar.values()):
    if c not in nomes_orig:
        sys.exit(f"classe desconhecida: {c} (existem: {nomes_orig})")
mapa = {}   # id original -> id novo
for i, n in enumerate(nomes_orig):
    alvo = juntar.get(n, n)
    if alvo in novas:
        mapa[i] = novas.index(alvo)

def quando(nome):
    m = re.search(r"(\d{4})-(\d{2})-(\d{2}) at (\d{2})\.(\d{2})\.(\d{2})", nome)
    if m:
        return datetime(*map(int, m.groups()))
    m = re.search(r"(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})", nome)
    return datetime(*map(int, m.groups())) if m else None

# Le imagens e rotulos (ja sem duplicatas e com as classes remapeadas)
pasta_img = os.path.join(a.src, "imagens")
itens, duplicadas, removidas = {}, 0, 0
for f in sorted(os.listdir(pasta_img)):
    nome, ext = os.path.splitext(f)
    if ext.lower() not in EXTENSOES or nome in EXCLUIR:
        continue
    txt = os.path.join(a.src, nome + ".txt")
    if not os.path.exists(txt):
        sys.exit(f"imagem sem rotulo: {f}")
    vistas, linhas = set(), []
    for l in open(txt):
        p = l.split()
        if not p:
            continue
        chave = tuple(p)
        if chave in vistas:
            duplicadas += 1
            continue
        vistas.add(chave)
        c = int(p[0])
        if c not in mapa:
            removidas += 1
            continue
        linhas.append(" ".join([str(mapa[c])] + p[1:5]))
    itens[nome] = (os.path.join(pasta_img, f), linhas)

def caixas(nome):
    cont = [0] * len(novas)
    for l in itens[nome][1]:
        cont[int(l.split()[0])] += 1
    return cont

reais = [n for n in itens if not n.startswith("gerada_")]
sinteticas = [n for n in itens if n.startswith("gerada_")]

# Sequencias de fotos reais; foto sem data no nome vira uma sequencia sozinha
sem_data = [n for n in reais if quando(n) is None]
com_data = sorted((n for n in reais if quando(n) is not None), key=quando)
grupos, ultimo = [], None
for n in com_data:
    t = quando(n)
    if ultimo is None or t - ultimo > GAP:
        grupos.append([])
    grupos[-1].append(n)
    ultimo = t
grupos += [[n] for n in sem_data]

# Sorteia a ordem das sequencias e enche test e val ate a proporcao pedida (em imagens);
# fica com o sorteio mais perto do alvo em imagens e em caixas de cada classe
rng = random.Random(a.seed)
tot_img = len(reais)
tot_cx = [sum(caixas(n)[k] for n in reais) for k in range(len(novas))]
frac = {"val": a.val, "test": a.test}
melhor = None
for _ in range(TENTATIVAS):
    ordem = grupos[:]
    rng.shuffle(ordem)
    img = {s: 0 for s in SPLITS}
    cx = {s: [0] * len(novas) for s in SPLITS}
    destino = {}
    for g in ordem:
        s = "train"
        for cand in ("test", "val"):
            if img[cand] + len(g) <= round(tot_img * frac[cand]):
                s = cand
                break
        for n in g:
            destino[n] = s
            img[s] += 1
            cx[s] = [x + y for x, y in zip(cx[s], caixas(n))]
    erro = 0.0
    for s in ("val", "test"):
        erro += abs(img[s] - tot_img * frac[s]) / max(tot_img * frac[s], 1)
        for k in range(len(novas)):
            if tot_cx[k]:
                alvo_k = tot_cx[k] * frac[s]
                erro += abs(cx[s][k] - alvo_k) / max(alvo_k, 1)
                erro += 5 * (cx[s][k] == 0)          # toda classe com pelo menos 1 caixa em val e test
    if melhor is None or erro < melhor[0]:
        melhor = (erro, destino)
destino = melhor[1]

# Sinteticas: so no treino; inpainting com origem fora do treino e descartado
no_treino = {n for n, s in destino.items() if s == "train"}
for n in sinteticas:
    m = re.match(r"gerada_(.*)_(baixa|meio)(_v\d+)?$", n)
    destino[n] = "descartada" if (m and m.group(1) in itens and m.group(1) not in no_treino) else "train"

shutil.rmtree(a.out, ignore_errors=True)
for s in SPLITS:
    os.makedirs(os.path.join(a.out, "images", s))
    os.makedirs(os.path.join(a.out, "labels", s))
rows = []
for n, s in sorted(destino.items()):
    img, linhas = itens[n]
    tipo = "sintetica" if n.startswith("gerada_") else "real"
    rows.append([os.path.basename(img), s, tipo] + caixas(n))
    if s == "descartada":
        continue
    shutil.copy2(img, os.path.join(a.out, "images", s, os.path.basename(img)))
    with open(os.path.join(a.out, "labels", s, n + ".txt"), "w") as f:
        f.write("\n".join(linhas) + ("\n" if linhas else ""))

with open(os.path.join(a.out, "split.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["imagem", "conjunto", "tipo"] + [f"caixas_{c}" for c in novas])
    w.writerows(sorted(rows, key=lambda r: (r[1], r[0])))

with open(os.path.join(a.out, "data.yaml"), "w") as f:
    # sem "path": a YOLO usa a pasta deste data.yaml, entao o dataset funciona em qualquer PC
    f.write("train: images/train\nval: images/val\ntest: images/test\n\nnames:\n")
    for i, c in enumerate(novas):
        f.write(f"  {i}: {c}\n")

print(f"classes: {novas}" + (f"  (juntadas: {juntar})" if juntar else ""))
print(f"{len(reais)} fotos reais em {len(grupos)} sequencias, {len(sinteticas)} sinteticas")
print(f"excluidas: {sorted(EXCLUIR)}")
print(f"caixas duplicadas removidas: {duplicadas} · caixas de classes fora da lista: {removidas}\n")
print(f"{'conjunto':12s}{'imagens':>9s}{'reais':>7s}" + "".join(f"{c:>10s}" for c in novas))
for s in SPLITS + ("descartada",):
    sel = [r for r in rows if r[1] == s]
    if not sel:
        continue
    print(f"{s:12s}{len(sel):9d}{sum(r[2] == 'real' for r in sel):7d}"
          + "".join(f"{sum(r[3 + k] for r in sel):10d}" for k in range(len(novas))))
for s in ("val", "test"):
    for k, c in enumerate(novas):
        if not any(r[1] == s and r[3 + k] for r in rows):
            print(f"AVISO: {c} ficou sem caixas em {s}/")

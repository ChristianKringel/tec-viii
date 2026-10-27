import sys, os, glob, csv
import cv2
import numpy as np

# Descobre de qual foto real cada imagem gerada por IA (gerada_image*) foi editada: compara pontos SIFT
# do fundo (predio, postes, guarda-corpo) e conta os que batem numa homografia (RANSAC). Com mais de
# LIMIAR pontos, a foto e considerada origem; pode haver mais de uma (fotos quase iguais da mesma sequencia).
# Saida: extras/origem_ia.csv (imagem, origem, pontos), usado pelo split.py e pelo experimentos/cv.py.
# Uso: python experimentos/origem_ia.py [classification_dir] [saida.csv]
LIMIAR = 250
src = sys.argv[1] if len(sys.argv) > 1 else "classification"
saida = sys.argv[2] if len(sys.argv) > 2 else os.path.join("extras", "origem_ia.csv")

sift = cv2.SIFT_create(3000)
bf = cv2.BFMatcher()

def pontos(p):
    im = cv2.imread(p, 0)
    h, w = im.shape
    s = 800 / max(h, w)
    return sift.detectAndCompute(cv2.resize(im, (int(w * s), int(h * s))), None)

def coincidentes(a, b):
    (ka, da), (kb, db) = a, b
    m = [x for x, y in bf.knnMatch(da, db, k=2) if x.distance < 0.7 * y.distance]
    if len(m) < 8:
        return 0
    _, mask = cv2.findHomography(np.float32([ka[x.queryIdx].pt for x in m]),
                                 np.float32([kb[x.trainIdx].pt for x in m]), cv2.RANSAC, 4.0)
    return int(mask.sum()) if mask is not None else 0

arquivos = [p for p in sorted(glob.glob(os.path.join(src, "*", "*"))) if not p.endswith(".csv")]
reais = {os.path.basename(p): pontos(p) for p in arquivos if not os.path.basename(p).startswith("gerada_")}
linhas = []
for p in arquivos:
    nome = os.path.basename(p)
    if not nome.startswith("gerada_image"):
        continue
    f = pontos(p)
    for real, n in sorted(((r, coincidentes(f, fr)) for r, fr in reais.items()), key=lambda x: -x[1]):
        if n < LIMIAR:
            break
        linhas.append((nome, real, n))
        print(f"{nome} <- {real} ({n} pontos)")

with open(saida, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["imagem", "origem", "pontos"])
    w.writerows(linhas)

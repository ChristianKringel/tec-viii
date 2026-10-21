import sys, os, glob, csv, random
import numpy as np
from PIL import Image, ImageDraw
from simple_lama_inpainting import SimpleLama

# Gera versoes com menos carros de fotos alta-ocupacao, apagando carros sorteados (pelas caixas
# das anotacoes YOLO) e preenchendo o lugar com o LaMa (inpainting). Como sabemos quantos carros
# sobraram, a classe nova e exata. Mesmos limites do classify.py.
# Uso: python remove_cars.py <labels_dir> <images_dir> <out_dir> <imagem1> [imagem2 ...]
BAIXA_MAX = 24
MEIO_MAX = 35
ALVOS = {"baixa-ocupacao": (12, 22), "meio-termo": (27, 33)}   # quantos carros devem sobrar
MARGEM = 0.20   # aumenta a caixa apagada em 20% de cada lado, para levar junto a sombra/borda
SEED = 0

labels_dir, images_dir, out_dir, nomes = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
names = open(os.path.join(labels_dir, "obj.names")).read().split()
carro = names.index("carro")
rng = random.Random(SEED)
lama = SimpleLama()

def classe(n):
    if n <= BAIXA_MAX: return "baixa-ocupacao"
    if n <= MEIO_MAX: return "meio-termo"
    return "alta-ocupacao"

def caixa_px(b, w, h, margem=0.0):
    _, cx, cy, bw, bh = b
    mx, my = bw * margem, bh * margem
    return (int((cx - bw / 2 - mx) * w), int((cy - bh / 2 - my) * h),
            int((cx + bw / 2 + mx) * w) + 1, int((cy + bh / 2 + my) * h) + 1)

for d in list(ALVOS) + ["comparacao", "labels"]:
    os.makedirs(os.path.join(out_dir, d), exist_ok=True)

rows = []
for nome in nomes:
    stem = os.path.splitext(nome)[0]
    img = Image.open(os.path.join(images_dir, nome)).convert("RGB")
    w, h = img.size
    boxes = [tuple(float(v) for v in l.split()) for l in open(os.path.join(labels_dir, stem + ".txt")) if len(l.split()) == 5]
    carros = [b for b in boxes if int(b[0]) == carro]
    outros = [b for b in boxes if int(b[0]) != carro]

    for alvo_classe, (lo, hi) in ALVOS.items():
        alvo = rng.randint(lo, hi)
        removidos = rng.sample(carros, len(carros) - alvo)
        mantidos = [b for b in carros if b not in removidos] + outros

        # Mascara: caixas removidas (com margem), menos as caixas mantidas, para nao borrar vizinhos
        mask = Image.new("L", (w, h), 0)
        dm = ImageDraw.Draw(mask)
        for b in removidos:
            dm.rectangle(caixa_px(b, w, h, MARGEM), fill=255)
        for b in mantidos:
            dm.rectangle(caixa_px(b, w, h), fill=0)

        saida = lama(img, mask).crop((0, 0, w, h))
        n_depois = alvo
        assert classe(n_depois) == alvo_classe
        arq = f"gerada_{stem}_{alvo_classe.split('-')[0]}.jpg"
        saida.save(os.path.join(out_dir, alvo_classe, arq), quality=95)

        # Anotacao nova: so os objetos mantidos
        with open(os.path.join(out_dir, "labels", os.path.splitext(arq)[0] + ".txt"), "w") as f:
            for b in mantidos:
                f.write(f"{int(b[0])} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")

        # Comparacao lado a lado: original com os removidos em vermelho | editada
        orig = img.copy()
        do = ImageDraw.Draw(orig)
        for b in removidos:
            do.rectangle(caixa_px(b, w, h), outline=(255, 0, 0), width=4)
        comp = Image.new("RGB", (w * 2 + 10, h), (255, 255, 255))
        comp.paste(orig, (0, 0)); comp.paste(saida, (w + 10, 0))
        comp.save(os.path.join(out_dir, "comparacao", arq), quality=90)

        rows.append((arq, nome, len(carros), len(removidos), n_depois, alvo_classe))
        print(f"{arq}: {len(carros)} -> {n_depois} carros ({alvo_classe})")

with open(os.path.join(out_dir, "relatorio.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["imagem", "origem", "carros_antes", "removidos", "carros_depois", "classe"])
    w.writerows(rows)

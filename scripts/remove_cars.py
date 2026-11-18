import sys, os, glob, csv, random
import numpy as np
import cv2
import torch
from PIL import Image
from simple_lama_inpainting import SimpleLama
from segment_anything import sam_model_registry, SamPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights

# Gera versoes com menos carros de fotos alta-ocupacao, apagando carros (pelas caixas das anotacoes
# YOLO) e preenchendo o lugar com o LaMa (inpainting). Como sabemos quantos carros sobraram, a classe
# nova e exata. Mesmos limites do classify.py.
# O SAM recorta o contorno de cada veiculo dentro da caixa: assim a mascara apaga o carro inteiro mais
# a sombra dele, e protege so o formato real dos vizinhos (e nao o retangulo todo).
# Um detector pronto (Faster R-CNN do COCO) acha veiculos sem anotacao (ex.: no corredor), que sao so
# protegidos: nao entram nas labels nem na contagem.
# Uso: python remove_cars.py <labels_dir> <images_dir> <out_dir> <imagem1> [imagem2 ...]
# Pesos do SAM: variavel SAM_CKPT (padrao ~/.cache/sam/sam_vit_b_01ec64.pth)
BAIXA_MAX = 24
MEIO_MAX = 35
ALVOS = {"baixa-ocupacao": (12, 22), "meio-termo": (27, 33)}   # quantos carros devem sobrar
MARGEM = 0.60        # ate onde a sombra pode ir, alem da caixa (fracao da caixa, de cada lado); o lado
MARGEM_BAIXO = 0.70  # depende do sol, e se a sombra for cortada o LaMa espalha o escuro no buraco
SOMBRA = 0.80        # pixel mais escuro que 80% do piso em volta conta como sombra...
TOM = 0.04           # ...e se o verde (cor sem o brilho) estiver a menos de 0.04 do verde do piso
DILATA = 0.06        # folga em volta do contorno do carro apagado (fracao do lado menor da caixa)
SEED = 0
# Variante (variavel VARIANTE, padrao 0): cada variante sorteia outros carros e outro alvo para a mesma
# foto e ganha o sufixo _v<n> no nome. Nas variantes o sorteio pesa mais, senao saem quase iguais.
VARIANTE = int(os.environ.get("VARIANTE", "0"))
ACASO = 0.05 if VARIANTE == 0 else 0.4
SUFIXO = f"_v{VARIANTE}" if VARIANTE else ""

labels_dir, images_dir, out_dir, nomes = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
names = open(os.path.join(labels_dir, "obj.names")).read().split()
carro = names.index("carro")
rng = random.Random(SEED + VARIANTE)
lama = SimpleLama()
sam_ckpt = os.environ.get("SAM_CKPT", os.path.expanduser("~/.cache/sam/sam_vit_b_01ec64.pth"))
sam = SamPredictor(sam_model_registry["vit_b"](checkpoint=sam_ckpt).eval())
detector = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1).eval()
VEICULOS_COCO = {3, 4, 6, 8}   # car, motorcycle, bus, truck

def classe(n):
    if n <= BAIXA_MAX: return "baixa-ocupacao"
    if n <= MEIO_MAX: return "meio-termo"
    return "alta-ocupacao"

def caixa_px(b, w, h, margem=0.0, margem_baixo=None):
    _, cx, cy, bw, bh = b
    mx, my = bw * margem, bh * margem
    mb = bh * (margem if margem_baixo is None else margem_baixo)
    return (max(int((cx - bw / 2 - mx) * w), 0), max(int((cy - bh / 2 - my) * h), 0),
            min(int((cx + bw / 2 + mx) * w) + 1, w), min(int((cy + bh / 2 + mb) * h) + 1, h))

def cobre(a, b):
    # Fracao da caixa a coberta pela caixa b (coordenadas YOLO normalizadas)
    ax0, ay0, ax1, ay1 = a[1] - a[3] / 2, a[2] - a[4] / 2, a[1] + a[3] / 2, a[2] + a[4] / 2
    bx0, by0, bx1, by1 = b[1] - b[3] / 2, b[2] - b[4] / 2, b[1] + b[3] / 2, b[2] + b[4] / 2
    iw, ih = min(ax1, bx1) - max(ax0, bx0), min(ay1, by1) - max(ay0, by0)
    return iw * ih / (a[3] * a[4]) if iw > 0 and ih > 0 else 0.0

def escolher_removidos(carros, outros, n):
    # Remove um por vez o carro que menos encosta nos que ficam: menos chance de sobrar pedaco dele
    # colado num vizinho. Pior se ele esta na frente do mantido (base mais baixa na foto).
    removidos, mantidos = [], list(carros)
    while len(removidos) < n:
        def custo(c):
            s = 0.0
            for k in mantidos + outros:
                if k is not c:
                    f = cobre(c, k)
                    s += f * (3 if c[2] + c[4] / 2 > k[2] + k[4] / 2 else 1)
            return s + rng.random() * ACASO   # parte aleatoria, para variar entre fotos e variantes
        c = min(mantidos, key=custo)
        mantidos.remove(c)
        removidos.append(c)
    return removidos

def nao_anotados(arr, anotados, w, h):
    # Veiculos que o detector acha e que nao batem com nenhuma caixa anotada, no formato YOLO (classe -1)
    with torch.no_grad():
        p = detector([torch.from_numpy(arr).permute(2, 0, 1).float() / 255])[0]
    out = []
    for (x0, y0, x1, y1), lab, s in zip(p["boxes"].tolist(), p["labels"].tolist(), p["scores"].tolist()):
        if lab not in VEICULOS_COCO or s < 0.5:
            continue
        b = (-1, (x0 + x1) / 2 / w, (y0 + y1) / 2 / h, (x1 - x0) / w, (y1 - y0) / h)
        if all(cobre(b, a) < 0.3 and cobre(a, b) < 0.3 for a in anotados):
            out.append(b)
    return out

def contornos(arr, boxes, w, h):
    # Uma mascara SAM por veiculo, cortada na propria caixa (com 5% de folga) para nao vazar no vizinho
    sam.set_image(arr)
    caixas = torch.tensor([caixa_px(b, w, h) for b in boxes], dtype=torch.float)
    with torch.no_grad():
        ms, _, _ = sam.predict_torch(None, None, boxes=sam.transform.apply_boxes_torch(caixas, (h, w)),
                                     multimask_output=False)
    out = []
    for b, m in zip(boxes, ms[:, 0].numpy()):
        x0, y0, x1, y1 = caixa_px(b, w, h, 0.05)
        corte = np.zeros_like(m)
        corte[y0:y1, x0:x1] = m[y0:y1, x0:x1]
        if corte.sum() < 0.4 * (x1 - x0) * (y1 - y0):   # SAM pegou so um pedaco: usa a caixa inteira
            corte[:] = False
            x0, y0, x1, y1 = caixa_px(b, w, h)
            corte[y0:y1, x0:x1] = True
        out.append(cv2.morphologyEx(corte.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)))
    return out

def mascara_removido(b, m, cinza, tom, todos, w, h):
    # Contorno do carro (dilatado) + sombra: pixels escuros dentro da caixa com margem, ligados ao carro
    x0, y0, x1, y1 = caixa_px(b, w, h, MARGEM, MARGEM_BAIXO)
    lado = min(b[3] * w, b[4] * h)
    k = max(3, int(lado * DILATA)) | 1
    carro_d = cv2.dilate(m, np.ones((k, k), np.uint8))
    # Nivel do piso: mediana de um anel em volta da regiao, sem contar veiculos
    X0, Y0, X1, Y1 = caixa_px(b, w, h, MARGEM + 0.3, MARGEM_BAIXO + 0.3)
    anel = np.zeros((h, w), bool)
    anel[Y0:Y1, X0:X1] = True
    anel[y0:y1, x0:x1] = False
    anel &= todos == 0
    if not anel.any():
        anel[y0:y1, x0:x1] = True
    piso, tom_piso = np.median(cinza[anel]), np.median(tom[anel], axis=0)
    regiao = np.zeros((h, w), np.uint8)
    regiao[y0:y1, x0:x1] = 1
    # Sombra escurece o piso e puxa para o azul (luz do ceu), mas mantem o verde: assim um carro sem
    # anotacao (ex.: parado no corredor) nao e confundido com sombra e apagado junto
    mesmo_tom = ((np.abs(tom[..., 1] - tom_piso[1]) < TOM) & (tom[..., 0] <= tom_piso[0] + 0.02)
                 & (tom[..., 2] >= tom_piso[2] - 0.02))
    # Logo embaixo do carro a sombra e quase preta e a cor fica instavel: ai so exige que nao puxe pro vermelho
    mesmo_tom |= (cinza < 0.35 * piso) & (tom[..., 0] <= tom_piso[0] + 0.03)
    cand = ((cinza < SOMBRA * piso) & mesmo_tom & (regiao > 0)).astype(np.uint8) | carro_d
    _, rot = cv2.connectedComponents(cand)
    ligados = np.unique(rot[carro_d > 0])
    # Alarga um pouco no fim para pegar a penumbra (borda clara da sombra)
    return cv2.dilate(np.isin(rot, ligados[ligados > 0]).astype(np.uint8), np.ones((9, 9), np.uint8))

for d in list(ALVOS) + ["comparacao", "labels"]:
    os.makedirs(os.path.join(out_dir, d), exist_ok=True)

rows = []
for nome in nomes:
    stem = os.path.splitext(nome)[0]
    img = Image.open(os.path.join(images_dir, nome)).convert("RGB")
    w, h = img.size
    arr = np.array(img)
    cinza = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY).astype(np.float32)
    tom = arr.astype(np.float32) / (arr.sum(axis=2, keepdims=True) + 1.0)   # cor sem o brilho
    boxes = [tuple(float(v) for v in l.split()) for l in open(os.path.join(labels_dir, stem + ".txt")) if len(l.split()) == 5]
    carros = [b for b in boxes if int(b[0]) == carro]
    outros = [b for b in boxes if int(b[0]) != carro]   # vao para as labels como estao
    # Veiculos que ficam e precisam de protecao: outras classes anotadas (menos vaga_livre, que e piso)
    # e os que o detector achou sem anotacao
    fixos = [b for b in outros if names[int(b[0])] != "vaga_livre"]
    fixos += nao_anotados(arr, carros + fixos, w, h)
    veiculos = carros + fixos
    sam_m = dict(zip(map(id, veiculos), contornos(arr, veiculos, w, h)))
    todos = np.maximum.reduce(list(sam_m.values()))
    print(f"{nome}: {sum(b[0] == -1 for b in fixos)} veiculos sem anotacao protegidos")

    for alvo_classe, (lo, hi) in ALVOS.items():
        if len(carros) - hi < 3:   # a foto ja esta nessa classe (ou quase): nao ha o que gerar
            continue
        alvo = rng.randint(lo, hi)
        removidos = escolher_removidos(carros, fixos, len(carros) - alvo)
        mantidos = [b for b in carros if b not in removidos] + outros

        # Mascara: contorno + sombra dos removidos, menos o contorno (com 2px de folga) dos que ficam
        mask = np.zeros((h, w), np.uint8)
        for b in removidos:
            mask |= mascara_removido(b, sam_m[id(b)], cinza, tom, todos, w, h)
        for b in [b for b in carros if b not in removidos] + fixos:
            mask[cv2.dilate(sam_m[id(b)], np.ones((5, 5), np.uint8)) > 0] = 0
        mask = Image.fromarray(mask * 255)

        # Cola o resultado do LaMa so dentro da mascara: fora dela a foto fica identica a original
        saida = Image.composite(lama(img, mask).crop((0, 0, w, h)), img, mask)
        n_depois = alvo
        assert classe(n_depois) == alvo_classe
        arq = f"gerada_{stem}_{alvo_classe.split('-')[0]}{SUFIXO}.jpg"
        saida.save(os.path.join(out_dir, alvo_classe, arq), quality=95)

        # Anotacao nova: so os objetos mantidos
        with open(os.path.join(out_dir, "labels", os.path.splitext(arq)[0] + ".txt"), "w") as f:
            for b in mantidos:
                f.write(f"{int(b[0])} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")

        # Comparacao lado a lado: original com a mascara apagada em vermelho | editada
        orig = img.copy()
        verm = Image.new("RGB", (w, h), (255, 0, 0))
        orig = Image.composite(Image.blend(orig, verm, 0.45), orig, mask)
        comp = Image.new("RGB", (w * 2 + 10, h), (255, 255, 255))
        comp.paste(orig, (0, 0)); comp.paste(saida, (w + 10, 0))
        comp.save(os.path.join(out_dir, "comparacao", arq), quality=90)

        rows.append((arq, nome, len(carros), len(removidos), n_depois, alvo_classe))
        print(f"{arq}: {len(carros)} -> {n_depois} carros ({alvo_classe})")

with open(os.path.join(out_dir, "relatorio.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["imagem", "origem", "carros_antes", "removidos", "carros_depois", "classe"])
    w.writerows(rows)

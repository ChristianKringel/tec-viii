#!/usr/bin/env python3
"""
grid_tool.py - Definicao da grade de vagas virtuais.

Projeto: Contagem de Vagas no Estacionamento da UFPel Anglo (TEC VIII)

O estacionamento nao tem demarcacao no chao, portanto as vagas sao criadas
manualmente uma unica vez sobre uma imagem de referencia. O resultado
(grid.json) e consumido pelas etapas seguintes do pipeline
(alinhamento, recorte, anotacao).

Uso:
    python grid_tool.py                       # menu com as imagens de dataset/
    python grid_tool.py --list                # so lista as imagens e sai
    python grid_tool.py --ref 15.13.08        # busca por parte do nome
    python grid_tool.py --ref 3               # busca pelo indice do menu
    python grid_tool.py --review              # revisa grid.json ja salvo

Os nomes exportados do WhatsApp tem espacos e parenteses, o que atrapalha na
linha de comando. Por isso --ref aceita apenas um trecho do nome
("15.13.08", "(2)") ou o indice mostrado no menu. Se nada for encontrado, o
programa lista as imagens disponiveis em vez de abortar. O caminho completo
tambem funciona, desde que entre aspas:
    python grid_tool.py --ref "dataset/WhatsApp Image 2026-09-07 at 15.13.08.jpeg"

Requisitos:
    pip install opencv-python numpy

Controles:
    f           modo FILEIRA  - 4 cliques + digitar N + ENTER
    v           modo VAGA     - 4 cliques (uma vaga isolada)
    r           modo ROI      - N cliques + ENTER para fechar
    u           desfazer (ponto pendente, ou ultimo objeto criado)
    l           liga/desliga rotulos das vagas
    m           liga/desliga a lupa
    s           salvar grid.json
    h           ajuda
    q / ESC     sair

Ordem dos cliques numa FILEIRA (importante):
    1) canto da frente da PRIMEIRA vaga
    2) canto da frente da ULTIMA vaga   <- o lado 1->2 define a direcao da fileira
    3) canto do fundo da ULTIMA vaga
    4) canto do fundo da PRIMEIRA vaga
    Ou seja: percorra o contorno da fileira inteira, sem cruzar as arestas.
    A subdivisao acontece ao longo da aresta 1->2.

Numa VAGA isolada a ordem e a mesma logica: contorno sem cruzar.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# ----------------------------------------------------------------------------
# Aparencia
# ----------------------------------------------------------------------------
COR_ROI = (0, 255, 255)
COR_FILEIRA = (255, 160, 0)
COR_VAGA = (0, 220, 0)
COR_EXTRA = (255, 0, 255)
COR_PENDENTE = (0, 0, 255)
COR_HUD = (255, 255, 255)
FONTE = cv2.FONT_HERSHEY_SIMPLEX

MAX_LADO_JANELA = 1500  # largura maxima da janela de trabalho
LUPA_RAIO = 60          # raio da regiao ampliada, em pixels da imagem original
LUPA_ZOOM = 4


EXTENSOES = {".jpeg", ".jpg", ".png", ".bmp", ".webp"}


# ----------------------------------------------------------------------------
# Descoberta e leitura de imagens
# ----------------------------------------------------------------------------
def ler_imagem(caminho):
    """cv2.imread falha silenciosamente com acentos/caracteres nao-ASCII no
    Windows. Ler os bytes e decodificar evita esse problema."""
    try:
        buf = np.fromfile(str(caminho), dtype=np.uint8)
    except OSError as e:
        print(f"[erro] nao foi possivel abrir {caminho}: {e}")
        return None
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def listar_imagens(dir_dataset):
    """Retorna as imagens de dataset/ ordenadas por nome."""
    d = Path(dir_dataset)
    if not d.is_dir():
        return []
    return sorted(
        (p for p in d.iterdir() if p.is_file() and p.suffix.lower() in EXTENSOES),
        key=lambda p: p.name.lower(),
    )


def _dimensoes(caminho):
    """Le so o cabecalho o suficiente para saber o tamanho. Barato o bastante
    para poucas imagens; devolve None se falhar."""
    img = ler_imagem(caminho)
    if img is None:
        return None
    h, w = img.shape[:2]
    return w, h


def mostrar_menu(imagens, dir_dataset):
    print(f"\nImagens em {dir_dataset}/ ({len(imagens)}):\n")
    infos = []
    for i, p in enumerate(imagens, start=1):
        dim = _dimensoes(p)
        infos.append((i, p, dim))

    # sugere a de maior resolucao: mais pixels por vaga no recorte
    melhor = max(
        (i for i, _, d in infos if d),
        key=lambda i: infos[i - 1][2][0] * infos[i - 1][2][1],
        default=None,
    )
    for i, p, dim in infos:
        tam = f"{dim[0]}x{dim[1]}" if dim else "ILEGIVEL"
        marca = "  <- maior resolucao" if i == melhor else ""
        print(f"  [{i:>2}] {p.name:<52} {tam}{marca}")
    print()
    return melhor


def escolher_interativo(imagens, dir_dataset):
    melhor = mostrar_menu(imagens, dir_dataset)
    if not sys.stdin.isatty():
        sys.exit("[erro] sem terminal interativo. Rode com --ref <trecho do nome>.")
    while True:
        padrao = f" [ENTER = {melhor}]" if melhor else ""
        resp = input(f"Numero da imagem de referencia{padrao} ou 'q' para sair: ").strip()
        if resp.lower() in ("q", "quit", "sair"):
            sys.exit(0)
        if not resp and melhor:
            return imagens[melhor - 1]
        if resp.isdigit() and 1 <= int(resp) <= len(imagens):
            return imagens[int(resp) - 1]
        print("  valor invalido, tente de novo.")


def resolver_ref(ref, dir_dataset):
    """Resolve --ref de varias formas, com menu como ultimo recurso.

    Ordem: caminho exato -> caminho dentro de dataset/ -> indice do menu ->
    trecho do nome (case-insensitive) -> menu interativo.
    """
    imagens = listar_imagens(dir_dataset)

    if not imagens:
        d = Path(dir_dataset)
        if not d.is_dir():
            sys.exit(
                f"[erro] a pasta '{dir_dataset}/' nao existe.\n"
                f"        Rode o script na raiz do projeto (ao lado de dataset/),\n"
                f"        ou informe outra pasta com --dataset <caminho>.\n"
                f"        Pasta atual: {Path.cwd()}"
            )
        sys.exit(
            f"[erro] nenhuma imagem em '{dir_dataset}/'.\n"
            f"        Extensoes aceitas: {', '.join(sorted(EXTENSOES))}"
        )

    if ref:
        p = Path(ref)
        if p.is_file():
            return p
        candidato = Path(dir_dataset) / ref
        if candidato.is_file():
            return candidato

        if ref.isdigit() and 1 <= int(ref) <= len(imagens):
            escolhida = imagens[int(ref) - 1]
            print(f"[info] indice {ref} -> {escolhida.name}")
            return escolhida

        alvo = ref.lower()
        parciais = [q for q in imagens if alvo in q.name.lower()]
        if len(parciais) == 1:
            print(f"[info] '{ref}' -> {parciais[0].name}")
            return parciais[0]
        if len(parciais) > 1:
            print(f"[aviso] '{ref}' casa com {len(parciais)} imagens:")
            for q in parciais:
                print(f"          {q.name}")
            return escolher_interativo(parciais, dir_dataset)

        print(f"[aviso] nada encontrado para '{ref}'.")

    return escolher_interativo(imagens, dir_dataset)


# ----------------------------------------------------------------------------
# Geometria
# ----------------------------------------------------------------------------
def subdividir_fileira(cantos, n):
    """Divide o quadrilatero de uma fileira em n vagas, com correcao de perspectiva.

    cantos: 4 pontos (x, y) na ordem frente-inicio, frente-fim, fundo-fim,
            fundo-inicio. A subdivisao ocorre ao longo da aresta cantos[0]->cantos[1].

    Retorna lista de n poligonos de 4 pontos, em coordenadas da imagem original.

    A divisao NAO e feita interpolando os cantos linearmente. Mapeia-se o
    quadrilatero para um retangulo unitario, divide-se ali (onde as vagas tem
    largura igual de verdade) e volta-se para a imagem. Em fileiras vistas em
    diagonal a diferenca entre os dois metodos e grande nas extremidades.
    """
    src = np.float32(cantos)
    unit = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
    H = cv2.getPerspectiveTransform(unit, src)  # unitario -> imagem

    polys = []
    for i in range(n):
        u0, u1 = i / n, (i + 1) / n
        quad = np.float32([[[u0, 0]], [[u1, 0]], [[u1, 1]], [[u0, 1]]])
        polys.append(cv2.perspectiveTransform(quad, H).reshape(-1, 2))
    return polys


def centroide(poly):
    p = np.asarray(poly, dtype=np.float32)
    return int(p[:, 0].mean()), int(p[:, 1].mean())


def area_poly(poly):
    return abs(cv2.contourArea(np.asarray(poly, dtype=np.float32)))


# ----------------------------------------------------------------------------
# Estado
# ----------------------------------------------------------------------------
class Estado:
    def __init__(self, img, caminho_img, escala):
        self.img = img
        self.caminho_img = caminho_img
        self.escala = escala
        self.h, self.w = img.shape[:2]

        self.modo = "fileira"          # fileira | vaga | roi
        self.pendentes = []            # pontos em coord. ORIGINAL
        self.aguardando_n = False
        self.buffer_n = ""

        self.roi = []                  # lista de pontos
        self.fileiras = []             # {"id", "corners", "n_spaces"}
        self.extras = []               # {"id", "polygon"}

        self.mostrar_rotulos = True
        self.mostrar_lupa = True
        self.mostrar_ajuda = False
        self.cursor = (0, 0)           # coord. ORIGINAL
        self.msg = "modo FILEIRA: clique os 4 cantos da fileira"

    # -- ids ---------------------------------------------------------------
    def proximo_id_fileira(self):
        usados = {f["id"] for f in self.fileiras}
        for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            if c not in usados:
                return c
        return f"F{len(self.fileiras)}"

    def proximo_id_extra(self):
        return f"X{len(self.extras) + 1:02d}"

    # -- vagas geradas -----------------------------------------------------
    def vagas(self):
        """Lista final de vagas: geradas das fileiras + extras manuais."""
        saida = []
        for f in self.fileiras:
            polys = subdividir_fileira(f["corners"], f["n_spaces"])
            for i, p in enumerate(polys, start=1):
                saida.append(
                    {
                        "id": f"{f['id']}{i:02d}",
                        "row": f["id"],
                        "polygon": [[float(x), float(y)] for x, y in p],
                    }
                )
        for e in self.extras:
            saida.append(
                {
                    "id": e["id"],
                    "row": None,
                    "polygon": [[float(x), float(y)] for x, y in e["polygon"]],
                }
            )
        return saida


# ----------------------------------------------------------------------------
# Desenho
# ----------------------------------------------------------------------------
def desenhar(est):
    canvas = cv2.resize(
        est.img, None, fx=est.escala, fy=est.escala, interpolation=cv2.INTER_AREA
    )
    s = est.escala

    def esc(pts):
        return (np.asarray(pts, dtype=np.float32) * s).astype(np.int32)

    # ROI
    if len(est.roi) >= 2:
        cv2.polylines(canvas, [esc(est.roi)], len(est.roi) >= 3, COR_ROI, 2)
    for p in est.roi:
        cv2.circle(canvas, tuple(esc([p])[0]), 4, COR_ROI, -1)

    # fileiras (contorno) + vagas geradas
    for f in est.fileiras:
        cv2.polylines(canvas, [esc(f["corners"])], True, COR_FILEIRA, 2)
        cx, cy = centroide(f["corners"])
        cv2.putText(
            canvas,
            f"{f['id']} ({f['n_spaces']})",
            (int(cx * s) - 30, int(cy * s)),
            FONTE, 0.7, COR_FILEIRA, 2,
        )

    for vg in est.vagas():
        cor = COR_EXTRA if vg["row"] is None else COR_VAGA
        cv2.polylines(canvas, [esc(vg["polygon"])], True, cor, 1)
        if est.mostrar_rotulos:
            cx, cy = centroide(vg["polygon"])
            cv2.putText(
                canvas, vg["id"], (int(cx * s) - 12, int(cy * s) + 4),
                FONTE, 0.35, cor, 1,
            )

    # pontos pendentes
    if est.pendentes:
        cv2.polylines(canvas, [esc(est.pendentes)], False, COR_PENDENTE, 2)
        for p in est.pendentes:
            cv2.circle(canvas, tuple(esc([p])[0]), 5, COR_PENDENTE, -1)
        for i, p in enumerate(est.pendentes, start=1):
            q = esc([p])[0]
            cv2.putText(canvas, str(i), (q[0] + 8, q[1] - 8), FONTE, 0.6, COR_PENDENTE, 2)

    hud(canvas, est)
    if est.mostrar_lupa:
        lupa(canvas, est)
    return canvas


def hud(canvas, est):
    n_vagas = len(est.vagas())
    linhas = [
        f"modo: {est.modo.upper()}   fileiras: {len(est.fileiras)}   "
        f"extras: {len(est.extras)}   vagas: {n_vagas}   ROI: {len(est.roi)} pts",
        est.msg,
    ]
    if est.aguardando_n:
        linhas.append(f"quantas vagas nesta fileira? {est.buffer_n}_  (ENTER confirma)")

    y = 24
    for t in linhas:
        cv2.rectangle(canvas, (8, y - 18), (8 + 11 * len(t), y + 8), (0, 0, 0), -1)
        cv2.putText(canvas, t, (12, y), FONTE, 0.6, COR_HUD, 1, cv2.LINE_AA)
        y += 28

    if est.mostrar_ajuda:
        ajuda = [
            "f fileira | v vaga | r ROI | ENTER fecha ROI",
            "u desfazer | l rotulos | m lupa | s salvar | q sair",
            "fileira: 1 frente-inicio, 2 frente-fim, 3 fundo-fim, 4 fundo-inicio",
        ]
        for t in ajuda:
            cv2.rectangle(canvas, (8, y - 18), (8 + 11 * len(t), y + 8), (0, 0, 0), -1)
            cv2.putText(canvas, t, (12, y), FONTE, 0.55, (120, 255, 120), 1, cv2.LINE_AA)
            y += 26


def lupa(canvas, est):
    """Inset ampliado ao redor do cursor, para clicar cantos com precisao."""
    cx, cy = est.cursor
    r = LUPA_RAIO
    x0, y0 = max(0, cx - r), max(0, cy - r)
    x1, y1 = min(est.w, cx + r), min(est.h, cy + r)
    if x1 - x0 < 10 or y1 - y0 < 10:
        return

    crop = est.img[y0:y1, x0:x1]
    crop = cv2.resize(crop, None, fx=LUPA_ZOOM, fy=LUPA_ZOOM,
                      interpolation=cv2.INTER_NEAREST)
    ch, cw = crop.shape[:2]

    # cruz na posicao exata do cursor dentro do crop
    px, py = (cx - x0) * LUPA_ZOOM, (cy - y0) * LUPA_ZOOM
    cv2.line(crop, (px - 14, py), (px + 14, py), (0, 0, 255), 1)
    cv2.line(crop, (px, py - 14), (px, py + 14), (0, 0, 255), 1)

    H, W = canvas.shape[:2]
    ox, oy = W - cw - 12, 12
    if ox < 0 or oy + ch > H:
        return
    canvas[oy:oy + ch, ox:ox + cw] = crop
    cv2.rectangle(canvas, (ox - 1, oy - 1), (ox + cw, oy + ch), (255, 255, 255), 1)
    cv2.putText(canvas, f"({cx},{cy})", (ox + 6, oy + ch - 8),
                FONTE, 0.45, (255, 255, 255), 1)


# ----------------------------------------------------------------------------
# Mouse
# ----------------------------------------------------------------------------
def on_mouse(evento, x, y, flags, est):
    ox, oy = int(round(x / est.escala)), int(round(y / est.escala))
    ox = max(0, min(est.w - 1, ox))
    oy = max(0, min(est.h - 1, oy))
    est.cursor = (ox, oy)

    if evento != cv2.EVENT_LBUTTONDOWN or est.aguardando_n:
        return

    est.pendentes.append((ox, oy))
    n = len(est.pendentes)

    if est.modo == "fileira":
        if n < 4:
            est.msg = f"fileira: {n}/4 cantos"
        else:
            est.aguardando_n = True
            est.buffer_n = ""
            est.msg = "digite o numero de vagas e pressione ENTER"
    elif est.modo == "vaga":
        if n < 4:
            est.msg = f"vaga: {n}/4 cantos"
        else:
            poly = np.float32(est.pendentes)
            if area_poly(poly) < 50:
                est.msg = "poligono degenerado, descartado"
            else:
                est.extras.append({"id": est.proximo_id_extra(), "polygon": poly})
                est.msg = f"vaga {est.extras[-1]['id']} criada"
            est.pendentes = []
    elif est.modo == "roi":
        est.msg = f"ROI: {n} pontos (ENTER fecha)"


# ----------------------------------------------------------------------------
# IO
# ----------------------------------------------------------------------------
def salvar(est, caminho):
    vagas = est.vagas()
    dados = {
        "reference_image": str(est.caminho_img),
        "image_size": [est.w, est.h],
        "roi": [[float(x), float(y)] for x, y in est.roi],
        "rows": [
            {
                "id": f["id"],
                "n_spaces": f["n_spaces"],
                "corners": [[float(x), float(y)] for x, y in f["corners"]],
            }
            for f in est.fileiras
        ],
        "extra_spaces": [
            {"id": e["id"], "polygon": [[float(x), float(y)] for x, y in e["polygon"]]}
            for e in est.extras
        ],
        "spaces": vagas,
    }
    Path(caminho).write_text(json.dumps(dados, indent=2), encoding="utf-8")
    print(f"[ok] {caminho}: {len(vagas)} vagas, {len(est.fileiras)} fileiras, "
          f"ROI com {len(est.roi)} pontos")


def carregar(est, caminho):
    p = Path(caminho)
    if not p.exists():
        return False
    d = json.loads(p.read_text(encoding="utf-8"))
    est.roi = [tuple(pt) for pt in d.get("roi", [])]
    est.fileiras = [
        {"id": r["id"], "n_spaces": r["n_spaces"],
         "corners": [tuple(c) for c in r["corners"]]}
        for r in d.get("rows", [])
    ]
    est.extras = [
        {"id": e["id"], "polygon": np.float32(e["polygon"])}
        for e in d.get("extra_spaces", [])
    ]
    tam = d.get("image_size")
    if tam and (tam[0] != est.w or tam[1] != est.h):
        print(f"[aviso] grid feito em {tam[0]}x{tam[1]}, imagem atual "
              f"{est.w}x{est.h}. Coordenadas nao vao coincidir sem alinhamento.")
    print(f"[ok] carregado de {caminho}")
    return True


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Define a grade de vagas virtuais.")
    ap.add_argument("--ref", default=None,
                    help="imagem de referencia: caminho, trecho do nome ou indice. "
                         "Sem este argumento, abre um menu.")
    ap.add_argument("--dataset", default="dataset",
                    help="pasta com as imagens (padrao: dataset)")
    ap.add_argument("--grid", default="grid.json", help="arquivo de saida/entrada")
    ap.add_argument("--review", action="store_true",
                    help="apenas visualiza um grid salvo, sem editar")
    ap.add_argument("--list", action="store_true",
                    help="lista as imagens disponiveis e sai")
    args = ap.parse_args()

    if args.list:
        imagens = listar_imagens(args.dataset)
        if not imagens:
            if not Path(args.dataset).is_dir():
                sys.exit(f"[erro] a pasta '{args.dataset}/' nao existe.\n"
                         f"        Pasta atual: {Path.cwd()}\n"
                         f"        Rode na raiz do projeto ou use --dataset <caminho>.")
            sys.exit(f"[erro] nenhuma imagem em '{args.dataset}/'.\n"
                     f"        Extensoes aceitas: {', '.join(sorted(EXTENSOES))}")
        mostrar_menu(imagens, args.dataset)
        return

    caminho = resolver_ref(args.ref, args.dataset)
    img = ler_imagem(caminho)
    if img is None:
        sys.exit(f"[erro] arquivo ilegivel ou corrompido: {caminho}\n"
                 f"        Confira se e realmente uma imagem "
                 f"(extensao pode estar mentindo).")
    print(f"[ok] referencia: {caminho.name} ({img.shape[1]}x{img.shape[0]})")

    h, w = img.shape[:2]
    escala = min(1.0, MAX_LADO_JANELA / w)
    est = Estado(img, caminho, escala)

    if Path(args.grid).exists():
        carregar(est, args.grid)
    elif args.review:
        sys.exit(f"[erro] {args.grid} nao existe, nada para revisar")

    if args.review:
        est.msg = "REVIEW (somente leitura) - q para sair"

    janela = "grid_tool - vagas virtuais"
    cv2.namedWindow(janela, cv2.WINDOW_AUTOSIZE)
    if not args.review:
        cv2.setMouseCallback(janela, on_mouse, est)

    while True:
        cv2.imshow(janela, desenhar(est))
        k = cv2.waitKey(20) & 0xFF
        if k == 255:
            continue

        if k in (ord("q"), 27):
            if not args.review and (est.fileiras or est.extras or est.roi):
                print("[aviso] saindo. Se nao salvou com 's', o trabalho foi perdido.")
            break
        if k == ord("h"):
            est.mostrar_ajuda = not est.mostrar_ajuda
            continue
        if k == ord("l"):
            est.mostrar_rotulos = not est.mostrar_rotulos
            continue
        if k == ord("m"):
            est.mostrar_lupa = not est.mostrar_lupa
            continue
        if args.review:
            continue

        # entrada do numero de vagas da fileira
        if est.aguardando_n:
            if ord("0") <= k <= ord("9"):
                est.buffer_n += chr(k)
            elif k == 8:  # backspace
                est.buffer_n = est.buffer_n[:-1]
            elif k in (13, 10):  # enter
                try:
                    n = int(est.buffer_n)
                except ValueError:
                    n = 0
                if 1 <= n <= 200:
                    est.fileiras.append(
                        {"id": est.proximo_id_fileira(),
                         "n_spaces": n,
                         "corners": list(est.pendentes)}
                    )
                    est.msg = f"fileira {est.fileiras[-1]['id']} com {n} vagas"
                    est.pendentes = []
                    est.aguardando_n = False
                    est.buffer_n = ""
                else:
                    est.msg = "valor invalido (1-200)"
                    est.buffer_n = ""
            elif k == 27:
                est.aguardando_n = False
                est.buffer_n = ""
                est.pendentes = []
                est.msg = "fileira cancelada"
            continue

        if k == ord("f"):
            est.modo, est.pendentes = "fileira", []
            est.msg = "modo FILEIRA: 4 cantos (frente-ini, frente-fim, fundo-fim, fundo-ini)"
        elif k == ord("v"):
            est.modo, est.pendentes = "vaga", []
            est.msg = "modo VAGA: 4 cantos de uma vaga isolada"
        elif k == ord("r"):
            est.modo, est.pendentes = "roi", []
            est.msg = "modo ROI: clique o contorno da area util, ENTER fecha"
        elif k in (13, 10):
            if est.modo == "roi":
                if len(est.pendentes) >= 3:
                    est.roi = list(est.pendentes)
                    est.pendentes = []
                    est.msg = f"ROI definida com {len(est.roi)} pontos"
                else:
                    est.msg = "ROI precisa de ao menos 3 pontos"
        elif k == ord("u"):
            if est.pendentes:
                est.pendentes.pop()
                est.msg = "ponto removido"
            elif est.modo == "roi" and est.roi:
                est.roi = []
                est.msg = "ROI apagada"
            elif est.modo == "vaga" and est.extras:
                est.msg = f"vaga {est.extras.pop()['id']} removida"
            elif est.fileiras:
                est.msg = f"fileira {est.fileiras.pop()['id']} removida"
        elif k == ord("s"):
            salvar(est, args.grid)
            est.msg = f"salvo em {args.grid}"

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
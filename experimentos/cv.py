import sys, os, re, csv, json, glob, shutil
from datetime import datetime, timedelta

# Validacao cruzada (5 folds) para comparar configuracoes de dados e hiperparametros da YOLO-cls.
# - Fotos reais sao agrupadas por sequencia (fotos tiradas com menos de 20 s de diferenca formam um
#   grupo): fotos da mesma sequencia mostram os mesmos carros e nunca ficam em conjuntos diferentes.
# - Cada foto real cai no teste exatamente uma vez. Val e teste so tem fotos reais.
# - Imagens sinteticas (geradas por IA, ou por inpainting a partir de uma foto real) so entram no treino,
#   e so quando a foto de origem esta no treino (origem das de IA: extras/origem_ia.csv).
# - O experimento "ingenuo" reproduz o split.py antigo (sorteio imagem a imagem, sinteticas em todos os
#   conjuntos), para medir quanto esse vazamento infla a nota.
# Uso:
#   python cv.py prep <saida> [split.csv]             monta os folds de cada experimento; com o split.csv
#                                                     do split.py, as fotos do teste final ficam de fora
#   python cv.py train <saida> <exp> <fold> <seed>    treina e avalia um fold
#   python cv.py report <saida>                       junta as previsoes e calcula as metricas
CLASSES = ("alta-ocupacao", "baixa-ocupacao", "meio-termo")
K = 5
GAP = timedelta(seconds=20)
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Experimentos: dados extras no treino + hiperparametros (o resto e igual ao notebook)
EXPS = {
    "real":             dict(extras=[]),
    "real+ia":          dict(extras=["ia"]),
    "real+inp":         dict(extras=["inp"]),
    "real+ia+inp":      dict(extras=["ia", "inp"]),
    "ingenuo":          dict(extras=["ia", "inp"], ingenuo=True),
    # real+ia+inp mais as variantes _v1/_v2 aprovadas do inpainting
    "real+ia+inp_var":  dict(extras=["ia", "inp", "var"]),
    # Hiperparametros, sobre a melhor configuracao de dados
    "todas@img320":     dict(extras=["ia", "inp"], imgsz=320),
    "todas@img448":     dict(extras=["ia", "inp"], imgsz=448),
    "todas@flip":       dict(extras=["ia", "inp"], fliplr=0.5),
    "todas@modelo_s":   dict(extras=["ia", "inp"], model="yolo26s-cls.pt"),
}
PADRAO = dict(imgsz=224, fliplr=0.0, epochs=30, model="yolo26n-cls.pt")

def quando(nome):
    m = re.search(r"(\d{4})-(\d{2})-(\d{2}) at (\d{2})\.(\d{2})\.(\d{2})", nome)
    if m:
        return datetime(*map(int, m.groups()))
    m = re.search(r"(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})", nome)
    return datetime(*map(int, m.groups())) if m else None

def inventario(fora=()):
    # fora: fotos reais que nao entram na validacao cruzada (o teste final), nem as sinteticas feitas delas
    reais, ia, inp = [], [], []
    for c in CLASSES:
        for p in sorted(glob.glob(os.path.join(RAIZ, "classification", c, "*"))):
            nome = os.path.basename(p)
            m = re.match(r"gerada_(WhatsApp.*)_(baixa|meio)\.jpg$", nome)
            if m:   # inpainting aprovado: gerada_<stem da origem>_<baixa|meio>.jpg
                inp.append(dict(nome=nome, caminho=p, classe=c, tipo="inp", stem=m.group(1)))
            elif nome.startswith("gerada_"):
                ia.append(dict(nome=nome, caminho=p, classe=c, tipo="ia"))
            elif nome not in fora:
                reais.append(dict(nome=nome, caminho=p, classe=c, tipo="real"))
    # Grupos por sequencia de fotos
    reais.sort(key=lambda r: quando(r["nome"]))
    g, ultimo = -1, None
    for r in reais:
        t = quando(r["nome"])
        if ultimo is None or t - ultimo > GAP:
            g += 1
        r["grupo"], ultimo = g, t
    # Variantes _v1/_v2 do inpainting (teste-remocao/v4), menos as reprovadas no controle visual
    rejeitadas = set(l.strip() for l in open(os.path.join(RAIZ, "experimentos", "rejeitadas.txt")) if l.strip())
    for c in ("baixa-ocupacao", "meio-termo"):
        for p in sorted(glob.glob(os.path.join(RAIZ, "teste-remocao", "v4", c, "*_v[0-9].jpg"))):
            nome = os.path.basename(p)
            if nome not in rejeitadas:
                stem = re.match(r"gerada_(.*)_(baixa|meio)_v\d+\.jpg$", nome).group(1)
                inp.append(dict(nome=nome, caminho=p, classe=c, tipo="var", stem=stem))
    # Liga cada inpainting a foto de origem; sem origem na validacao cruzada (teste final), fica de fora
    origem = {os.path.splitext(r["nome"])[0]: r["nome"] for r in reais}
    inp = [dict(r, origens=[origem[r["stem"]]]) for r in inp if r["stem"] in origem]
    # IA: origens descobertas por experimentos/origem_ia.py; sem origem listada, entra em todo fold
    origem_ia = {}
    for r in csv.DictReader(open(os.path.join(RAIZ, "extras", "origem_ia.csv"))):
        origem_ia.setdefault(r["imagem"], []).append(r["origem"])
    for r in ia:
        r["origens"] = origem_ia.get(r["nome"], [])
    return reais, ia, inp

def folds(reais):
    # Com poucos grupos, o sorteio do StratifiedGroupKFold muda muito o equilibrio dos folds: testa
    # varias sementes e fica com a de folds mais parecidos (tamanho e quantidade de cada classe)
    import warnings
    import numpy as np
    from sklearn.model_selection import StratifiedGroupKFold
    y = [r["classe"] for r in reais]
    gs = [r["grupo"] for r in reais]
    melhor = None
    for seed in range(300):
        skf = StratifiedGroupKFold(n_splits=K, shuffle=True, random_state=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f = {reais[j]["nome"]: i for i, (_, te) in enumerate(skf.split(reais, y, gs)) for j in te}
        cont = np.array([[sum(f[r["nome"]] == k and r["classe"] == c for r in reais) for c in CLASSES]
                         for k in range(K)])
        custo = cont.sum(1).std() + cont.std(0).sum()
        if melhor is None or custo < melhor[0]:
            melhor = (custo, f)
    return melhor[1]

def folds_ingenuos(todas):
    # Como o split.py antigo: sorteio estratificado imagem a imagem, sem olhar sequencia nem origem
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=0)
    y = [r["classe"] for r in todas]
    return {todas[j]["nome"]: i for i, (_, te) in enumerate(skf.split(todas, y)) for j in te}

def liga(origem, destino):
    os.makedirs(os.path.dirname(destino), exist_ok=True)
    os.symlink(origem, destino)

def prep(saida, split_csv=None):
    fora = set()
    if split_csv:
        fora = {r["imagem"] for r in csv.DictReader(open(split_csv)) if r["conjunto"] == "test" and r["tipo"] == "real"}
    reais, ia, inp = inventario(fora)
    fold = folds(reais)
    shutil.rmtree(os.path.join(saida, "dados"), ignore_errors=True)
    for exp, cfg in EXPS.items():
        for k in range(K):
            base = os.path.join(saida, "dados", exp, str(k))
            for s in ("train", "val", "test"):
                for c in CLASSES:
                    os.makedirs(os.path.join(base, s, c), exist_ok=True)
            if cfg.get("ingenuo"):
                todas = reais + [r for r in ia if "ia" in cfg["extras"]] + [r for r in inp if r["tipo"] in cfg["extras"]]
                fi = folds_ingenuos(todas)
                for r in todas:
                    s = "test" if fi[r["nome"]] == k else "val" if fi[r["nome"]] == (k + 1) % K else "train"
                    liga(r["caminho"], os.path.join(base, s, r["classe"], r["nome"]))
                continue
            no_treino = set()
            for r in reais:
                f = fold[r["nome"]]
                s = "test" if f == k else "val" if f == (k + 1) % K else "train"
                liga(r["caminho"], os.path.join(base, s, r["classe"], r["nome"]))
                if s == "train":
                    no_treino.add(r["nome"])
            for r in ia + inp:
                if r["tipo"] in cfg["extras"] and all(o in no_treino for o in r["origens"]):
                    liga(r["caminho"], os.path.join(base, "train", r["classe"], r["nome"]))
    with open(os.path.join(saida, "inventario.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["imagem", "classe", "tipo", "grupo", "fold", "origem"])
        for r in reais + ia + inp:
            w.writerow([r["nome"], r["classe"], r["tipo"], r.get("grupo", ""), fold.get(r["nome"], ""),
                        ";".join(r.get("origens", []))])
    for exp in EXPS:
        base = os.path.join(saida, "dados", exp, "0")
        cont = {s: [len(os.listdir(os.path.join(base, s, c))) for c in CLASSES] for s in ("train", "val", "test")}
        print(f"{exp:18s} fold 0 (alta, baixa, meio): {json.dumps(cont)}")
    n_var = sum(r["tipo"] == "var" for r in inp)
    print(f"{len(reais)} reais em {len(set(r['grupo'] for r in reais))} grupos, {len(ia)} ia, "
          f"{len(inp) - n_var} inp, {n_var} variantes; fora da validacao cruzada: {len(fora)}")

def train(saida, exp, k, seed):
    from ultralytics import YOLO
    cfg = {**PADRAO, **EXPS[exp]}
    dados = os.path.join(saida, "dados", exp, str(k))
    proj = os.path.join(saida, "runs", exp, f"s{seed}")
    os.chdir(saida)
    model = YOLO(cfg["model"])
    model.train(data=dados, epochs=cfg["epochs"], imgsz=cfg["imgsz"], batch=8, patience=100, seed=seed,
                deterministic=True, fliplr=cfg["fliplr"], project=proj, name=str(k), exist_ok=True,
                device="cpu", workers=2, verbose=False, plots=False)
    pasta = os.path.join(proj, str(k))
    tipo = {}
    with open(os.path.join(saida, "inventario.csv")) as fh:
        for r in csv.DictReader(fh):
            tipo[r["imagem"]] = r["tipo"]
    linhas = []
    for peso in ("best", "last"):
        m = YOLO(os.path.join(pasta, "weights", f"{peso}.pt"))
        for c in CLASSES:
            for p in sorted(glob.glob(os.path.join(dados, "test", c, "*"))):
                r = m.predict(p, imgsz=cfg["imgsz"], verbose=False, device="cpu")[0]
                nome = os.path.basename(p)
                linhas.append([exp, seed, k, peso, nome, tipo[nome], c, r.names[r.probs.top1],
                               float(r.probs.top1conf)])
    with open(os.path.join(pasta, "previsoes.csv"), "w", newline="") as fh:
        csv.writer(fh).writerows(linhas)

def metricas(d):
    # F1 macro so sobre as classes presentes no teste (sem foto real de uma classe, ela nao entra na media)
    from sklearn.metrics import f1_score, recall_score
    presentes = [c for c in CLASSES if (d["real"] == c).any()]
    rec = dict(zip(CLASSES, recall_score(d["real"], d["prev"], labels=list(CLASSES), average=None, zero_division=0)))
    return dict(acuracia=(d["real"] == d["prev"]).mean(),
                f1_macro=f1_score(d["real"], d["prev"], labels=presentes, average="macro", zero_division=0),
                **{f"rec_{c.split('-')[0]}": rec[c] if c in presentes else float("nan") for c in CLASSES})

def report(saida):
    import pandas as pd
    from sklearn.metrics import confusion_matrix
    cols = ["exp", "seed", "fold", "peso", "imagem", "tipo", "real", "prev", "conf"]
    df = pd.concat(pd.read_csv(p, names=cols) for p in glob.glob(os.path.join(saida, "runs", "*", "*", "*", "previsoes.csv")))
    linhas = []
    for (exp, peso), d in df.groupby(["exp", "peso"]):
        # Metricas por semente (cada foto de teste uma vez por semente); depois media e desvio entre sementes
        por_seed = pd.DataFrame([metricas(ds) for _, ds in d.groupby("seed")])
        lin = dict(exp=exp, peso=peso, sementes=d["seed"].nunique(), n_por_semente=len(d) // d["seed"].nunique())
        for c in por_seed.columns:
            lin[c] = por_seed[c].mean()
        lin["acc_dp"] = por_seed["acuracia"].std()
        if (d["tipo"] != "real").any():   # experimento ingenuo: nota so nas fotos reais do teste
            lin["acc_so_reais"] = pd.Series([metricas(ds)["acuracia"] for _, ds in d[d["tipo"] == "real"].groupby("seed")]).mean()
        linhas.append(lin)
        cm = confusion_matrix(d["real"], d["prev"], labels=list(CLASSES))
        print(f"\n{exp} [{peso}] matriz somada nas sementes (linhas = real, colunas = previsto; alta, baixa, meio):\n{cm}")
    res = pd.DataFrame(linhas).sort_values(["peso", "f1_macro"], ascending=[True, False])
    print("\n" + res.round(3).to_string(index=False))
    res.to_csv(os.path.join(saida, "resultados.csv"), index=False)

if __name__ == "__main__":
    cmd, saida = sys.argv[1], os.path.abspath(sys.argv[2])
    if cmd == "prep":
        prep(saida, sys.argv[3] if len(sys.argv) > 3 else None)
    elif cmd == "train":
        train(saida, sys.argv[3], int(sys.argv[4]), int(sys.argv[5]))
    elif cmd == "report":
        report(saida)

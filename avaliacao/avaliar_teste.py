import sys, os, glob, csv, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import precision_recall_fscore_support
from ultralytics import YOLO

# Avaliacao no teste, feita UMA vez depois do treino (topicos 5 a 9 do guia de apresentacao):
# - model.val(split="test") para o best.pt e o last.pt (acuracia top-1 e matrizes de confusao do teste);
# - previsoes_teste.csv com o best.pt (caminho, classe real, classe prevista, confianca);
# - precisao, recall e F1 por classe, F1 macro, referencia "sempre a classe mais frequente";
# - galeria de erros (ordenada pela confianca) e resumo.json com todos os numeros;
# - acuracia tambem sem as fotos do teste que tem uma versao editada (IA) no treino (coluna obs do split.csv).
# Uso: python avaliacao/avaliar_teste.py <dataset_cls> <pasta_do_treino>
#   ex.: python avaliacao/avaliar_teste.py dataset_cls avaliacao/runs/treino
EXTENSOES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

dados, treino = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
saida = os.path.dirname(treino)

# Top-1 de validacao por epoca. O best.pt e regravado a cada epoca que empata com a melhor nota,
# entao ele corresponde a ULTIMA epoca com a maior top-1 de validacao
res = pd.read_csv(os.path.join(treino, "results.csv"))
res.columns = [c.strip() for c in res.columns]
top1 = res["metrics/accuracy_top1"]
ep_best = int(res.loc[top1[top1 == top1.max()].index[-1], "epoch"])
ep_last = int(res["epoch"].iloc[-1])

def contar(conj):
    return {c: len([p for p in glob.glob(os.path.join(dados, conj, c, "*")) if p.lower().endswith(EXTENSOES)])
            for c in sorted(os.listdir(os.path.join(dados, conj)))}

resumo = {"imagens": {s: contar(s) for s in ("train", "val", "test")}, "pesos": {}}
for peso, ep in (("best", ep_best), ("last", ep_last)):
    m = YOLO(os.path.join(treino, "weights", f"{peso}.pt"))
    r = m.val(data=dados, split="test", imgsz=224, batch=8, device="cpu", project=saida, name=f"teste_{peso}",
              exist_ok=True)
    n = sum(resumo["imagens"]["test"].values())
    resumo["pesos"][peso] = dict(epoca=ep, top1_val=float(top1[res["epoch"] == ep].iloc[0]),
                                 top1_teste=float(r.top1), acertos=int(round(r.top1 * n)), imagens=n,
                                 pasta=os.path.join(saida, f"teste_{peso}"))

# CSV de previsoes do best.pt
m = YOLO(os.path.join(treino, "weights", "best.pt"))
nomes = [m.names[i] for i in range(len(m.names))]
linhas = []
for c in sorted(os.listdir(os.path.join(dados, "test"))):
    for p in sorted(glob.glob(os.path.join(dados, "test", c, "*"))):
        if not p.lower().endswith(EXTENSOES):
            continue
        r = m.predict(p, imgsz=224, verbose=False, device="cpu")[0]
        probs = r.probs.data.tolist()
        linhas.append(dict(caminho=os.path.relpath(p, dados), classe_real=c, classe_prevista=nomes[r.probs.top1],
                           confianca=float(r.probs.top1conf), **{f"p_{n}": probs[i] for i, n in enumerate(nomes)}))
prev = pd.DataFrame(linhas)
prev.to_csv(os.path.join(saida, "previsoes_teste.csv"), index=False)

# Metricas por classe; a verificacao do guia: a acuracia do CSV deve bater com a do model.val
acc_csv = float((prev["classe_real"] == prev["classe_prevista"]).mean())
p, r, f, s = precision_recall_fscore_support(prev["classe_real"], prev["classe_prevista"], labels=nomes,
                                             zero_division=0)
presentes = [i for i, n in enumerate(nomes) if s[i] > 0]
resumo["por_classe"] = {n: dict(precisao=float(p[i]), recall=float(r[i]), f1=float(f[i]), imagens=int(s[i]))
                        for i, n in enumerate(nomes)}
resumo["f1_macro"] = float(np.mean([f[i] for i in presentes]))
resumo["precisao_macro"] = float(np.mean([p[i] for i in presentes]))
resumo["recall_macro"] = float(np.mean([r[i] for i in presentes]))
resumo["acuracia_csv"] = acc_csv
arq_split = os.path.join(dados, "split.csv")
editadas = set()
for x in (csv.DictReader(open(arq_split)) if os.path.exists(arq_split) else []):
    if x.get("obs", "").startswith("origem fora do treino: "):
        editadas |= {o.strip() for o in x["obs"].split(": ", 1)[1].split(";")}
prev["versao_editada_no_treino"] = prev["caminho"].map(lambda c: os.path.splitext(os.path.basename(c))[0] in editadas)
limpas = prev[~prev["versao_editada_no_treino"]]
resumo["sem_versao_editada_no_treino"] = dict(
    imagens=len(limpas), acertos=int((limpas["classe_real"] == limpas["classe_prevista"]).sum()),
    excluidas=sorted(prev.loc[prev["versao_editada_no_treino"], "caminho"]))
prev.to_csv(os.path.join(saida, "previsoes_teste.csv"), index=False)
mais_freq = prev["classe_real"].value_counts()
resumo["referencia_mais_frequente"] = dict(classe=mais_freq.index[0], acuracia=float(mais_freq.iloc[0] / len(prev)))
with open(os.path.join(saida, "resumo.json"), "w") as fh:
    json.dump(resumo, fh, indent=2, ensure_ascii=False)

# Painel com todas as previsoes do teste, erros primeiro e por confianca (galeria de erros)
prev["erro"] = prev["classe_real"] != prev["classe_prevista"]
ordem = prev.sort_values(["erro", "confianca"], ascending=[False, False]).reset_index(drop=True)
cols = 5
lin = int(np.ceil(len(ordem) / cols))
fig, axs = plt.subplots(lin, cols, figsize=(4 * cols, 3.6 * lin))
for ax in np.ravel(axs):
    ax.axis("off")
for ax, (_, x) in zip(np.ravel(axs), ordem.iterrows()):
    ax.imshow(Image.open(os.path.join(dados, x["caminho"])).convert("RGB"))
    cor = "#c62828" if x["erro"] else "#2e7d32"
    ax.set_title(f"{'ERRO' if x['erro'] else 'ok'}: real {x['classe_real']}\nprevisto {x['classe_prevista']} "
                 f"({x['confianca']:.0%})", color=cor, fontsize=10)
fig.suptitle("Teste: previsoes do best.pt (erros primeiro, por confianca)", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(saida, "galeria_teste.jpg"), dpi=110)

print(json.dumps(resumo, indent=2, ensure_ascii=False))
dif = abs(acc_csv - resumo["pesos"]["best"]["top1_teste"])
print(f"\nVerificacao: acuracia pelo CSV {acc_csv:.1%} x model.val {resumo['pesos']['best']['top1_teste']:.1%}"
      f" ({'ok' if dif < 0.01 else 'DIFERENTE, conferir'})")

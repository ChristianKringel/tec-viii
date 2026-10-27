# Experimentos: dados sintéticos (YOLO26n-cls)

Validação cruzada com 5 folds sobre as **27 fotos reais que não estão no teste final** (as 10 do teste,
definidas pelo `split.py`, ficam de fora para não participarem de nenhuma decisão). Fotos com menos de
20 s de diferença formam um grupo e ficam sempre no mesmo conjunto. As imagens sintéticas entram **só
no treino**, e só quando a foto de origem também está no treino: a origem do inpainting está no nome do
arquivo, e a das imagens editadas por IA foi descoberta por `origem_ia.py` (`extras/origem_ia.csv`).
Cada configuração foi rodada com 3 sementes (CPU, ~2 min por treino).

Configuração base = a do notebook: `yolo26n-cls.pt`, `imgsz=224`, `epochs=30`, `batch=8`, `fliplr=0.0`.

## Dados de treino (peso `best.pt`, média de 3 sementes)

| treino | acurácia | F1 macro (alta, meio) | recall alta | recall meio | meio previsto como baixa |
| --- | --- | --- | --- | --- | --- |
| só reais | 72,8% | 0,64 | 93% | 33% | 0 |
| + editadas por IA | 75,3% | 0,65 | 98% | 30% | 3 |
| + inpainting | 75,3% | 0,69 | 91% | 44% | 0 |
| **+ IA + inpainting (escolhido)** | 72,8% | 0,65 | 93% | 33% | 3 |
| + IA + inpainting + variantes `_v1`/`_v2` | 75,3% | 0,66 | 96% | 33% | 1 |

A única foto real de baixa está no teste final, então a validação cruzada mede só alta × meio (a
contagem de "meio previsto como baixa" soma as 3 sementes). As diferenças são menores que uma foto
(3,7 pontos) e o desvio entre sementes vai de 2 a 9 pontos: **nenhuma configuração é melhor além do
ruído**. Os dados sintéticos não atrapalham alta × meio e são a única fonte de baixa no treino, por isso
o treino oficial usa IA + inpainting (sem as variantes, que são quase cópias das mesmas cenas).

### Histórico

Uma rodada anterior indicava 68,5% (só reais) → 82,9% (+ IA + inpainting). Ela tinha dois problemas,
corrigidos aqui: a foto 08.33.12 estava rotulada como baixa por falta de caixas na anotação (corrigida
para meio-termo em `rotulos_corrigidos.csv`), e as imagens de IA entravam em todos os folds, inclusive
quando a foto de origem estava no teste do fold. Hiperparâmetros (`imgsz` 320/448, `fliplr=0.5`,
`yolo26s-cls`) só foram testados nessa rodada anterior e não superaram a base; ficamos com o baseline.

## Arquivos

- `cv.py`: monta os folds, treina e gera o relatório (`prep`, `train`, `report`).
- `origem_ia.py`: descobre de qual foto real cada imagem editada por IA saiu (SIFT + RANSAC).
- `rejeitadas.txt`: imagens de inpainting reprovadas no controle visual (globos de poste borrados, manchas).
- `resultados_cv.csv`: todas as métricas; `inventario_cv.csv`: cada imagem com tipo, sequência, fold e origem.

Para reproduzir (precisa de `ultralytics` e `scikit-learn`; `origem_ia.py` precisa de `opencv`):

```bash
python split.py classification dataset_cls 0.25 0.25 0
python experimentos/cv.py prep cv_saida dataset_cls/split.csv
python experimentos/cv.py train cv_saida real+ia+inp 0 0     # exp, fold, semente
python experimentos/cv.py report cv_saida
```

O treino oficial e a avaliação no teste ficam em `avaliacao/` (`treinar.py` e `avaliar_teste.py`).

# Teste de remoção de carros (inpainting)

Instruções para um agente (Claude Code) rodar o teste de geração de imagens com menos carros.

## Contexto

O dataset classifica fotos do estacionamento pela quantidade de carros (classe `carro` nas anotações YOLO):

| classe | carros |
| --- | --- |
| `baixa-ocupacao` | até 24 |
| `meio-termo` | 25 a 35 |
| `alta-ocupacao` | 36 ou mais |

Há poucas fotos reais de `baixa-ocupacao` e `meio-termo`. O `remove_cars.py` pega fotos `alta-ocupacao` anotadas,
sorteia carros pelas caixas de `annotations/` e apaga com o LaMa (inpainting). Como o script sabe quantos carros sobraram,
a classe nova é exata.

## Objetivo do teste

Gerar, a partir de **10 fotos alta-ocupacao**, 10 imagens `baixa-ocupacao` e 10 `meio-termo`, numa pasta separada na raiz
do projeto, **`teste-remocao/`**, para o usuário avaliar a qualidade visual.

**Não** copiar nada para `classification/`, `extras/` ou `dataset_cls/`. Só o usuário decide se as imagens entram no dataset.

## Ambiente

Feito para rodar na CPU (máquina com 32 GB de RAM). A tentativa anterior numa máquina com ~2 GB livres morreu por falta
de memória (exit 137): o LaMa na foto inteira (1600x1200) precisa de alguns GB. GPU não é necessária; a RX 6600 (AMD)
exigiria ROCm e não compensa para 20 imagens.

```bash
python3 -m venv .venv-lama
./.venv-lama/bin/pip install --upgrade pip
./.venv-lama/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
./.venv-lama/bin/pip install simple-lama-inpainting
```

Na primeira execução, o `simple-lama-inpainting` baixa o modelo `big-lama.pt` (~200 MB) do GitHub.
Não commitar `.venv-lama/` nem `teste-remocao/` sem o usuário pedir.

## Execução

Testar primeiro com uma imagem e conferir o resultado:

```bash
./.venv-lama/bin/python remove_cars.py annotations dataset teste-remocao "WhatsApp Image 2026-09-15 at 08.04.10.jpeg"
```

Se der certo, apagar `teste-remocao/` e rodar com as 10 fotos (escolhidas sem pares quase idênticos):

```bash
rm -rf teste-remocao
./.venv-lama/bin/python remove_cars.py annotations dataset teste-remocao \
  "WhatsApp Image 2026-09-15 at 08.28.27.jpeg" \
  "WhatsApp Image 2026-09-15 at 08.04.10.jpeg" \
  "WhatsApp Image 2026-09-07 at 15.13.08.jpeg" \
  "WhatsApp Image 2026-09-15 at 08.27.30.jpeg" \
  "WhatsApp Image 2026-09-15 at 08.04.23.jpeg" \
  "WhatsApp Image 2026-09-15 at 08.25.41.jpeg" \
  "WhatsApp Image 2026-09-15 at 08.28.34.jpeg" \
  "WhatsApp Image 2026-09-15 at 08.25.40 (1).jpeg" \
  "WhatsApp Image 2026-09-07 at 14.21.36.jpeg" \
  "WhatsApp Image 2026-09-15 at 08.04.25.jpeg"
```

Se ainda faltar memória, trocar no script a chamada `lama(img, mask)` por um inpainting em recortes ao redor de cada
carro removido (crop com margem de contexto, inpaint, colar de volta).

## Saída esperada

```
teste-remocao/
├── baixa-ocupacao/   10 imagens  gerada_<foto>_baixa.jpg  (12 a 22 carros)
├── meio-termo/       10 imagens  gerada_<foto>_meio.jpg   (27 a 33 carros)
├── comparacao/       20 imagens  original com os removidos em vermelho | editada
├── labels/           20 anotações YOLO só com os objetos mantidos
└── relatorio.csv     imagem, origem, carros_antes, removidos, carros_depois, classe
```

## Verificação antes de entregar

1. Conferir que há 10 imagens em cada classe e 20 linhas no `relatorio.csv`.
2. Abrir algumas imagens de `comparacao/` e procurar:
   - manchas ou sombras escuras onde havia carro (se for comum, aumentar `MARGEM` no script, ex.: 0.3);
   - carros mantidos cortados ou borrados quando estavam encostados num removido;
   - borrões nos carros pequenos do fundo.
3. Relatar ao usuário quantas imagens ficaram boas, quais têm defeito e qual defeito. Não esconder os problemas.

## Ressalvas para o relatório do trabalho

- Imagens editadas equilibram as classes, mas o modelo pode aprender marcas da edição. `val/` e `test/` devem ter
  principalmente fotos reais.
- Uma foto editada e a original vêm da mesma cena: manter as duas no mesmo conjunto (train, val ou test) no `split.py`.

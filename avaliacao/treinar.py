import sys, os
from ultralytics import YOLO

# Treino oficial, com a mesma configuracao do yolo_mnist.ipynb (baseline).
# Uso: python avaliacao/treinar.py <dataset_cls> [saida]
#   ex.: python avaliacao/treinar.py dataset_cls avaliacao/runs
dados = os.path.abspath(sys.argv[1])
saida = os.path.abspath(sys.argv[2] if len(sys.argv) > 2 else "avaliacao/runs")

model = YOLO("yolo26n-cls.pt")    # pre-treinado no ImageNet
model.train(
    data=dados,
    epochs=30,
    imgsz=224,
    batch=8,
    patience=30,
    seed=0,
    deterministic=True,
    fliplr=0.0,
    project=saida,
    name="treino",
    exist_ok=True,
    device="cpu",
)
print("Treino salvo em", model.trainer.save_dir)

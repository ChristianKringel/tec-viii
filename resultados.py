from ultralytics import YOLO
def main():
 modelo = YOLO("resultados/treino1/weights/best.pt")
 metricas = modelo.val(data="/home/aluno/meu_dataset/data.yaml")
 print(f"precisão média : {metricas.box.mp:.3f}")
 print(f"recall médio : {metricas.box.mr:.3f}")
 print(f"mAP50 : {metricas.box.map50:.3f}")
 print(f"mAP50-95 : {metricas.box.map:.3f}")
 # mAP50-95 de cada classe
 for i, valor in enumerate(metricas.box.maps):
 print(f"{modelo.names[i]:15s} mAP50-95 {valor:.3f}")
if __name__ == "__main__": # obrigatório em scripts (veja a seção 5)
 main()

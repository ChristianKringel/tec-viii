from ultralytics import YOLO
def main():
 # modelo pequeno, pré-treinado no COCO (80 classes)
 modelo = YOLO("yolo26n.pt")
 modelo.train(
 data="/home/aluno/meu_dataset/data.yaml", # o arquivo data.yaml
 epochs=100,
 imgsz=640,
 batch=16,
 patience=20,
 seed=0,
 project="resultados",
 name="treino1",
 )
if __name__ == "__main__":
 main()

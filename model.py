from ultralytics import YOLO

def train_model():
    model = YOLO("yolo26n.pt")

    model.train(data="data/data.yaml", epochs=20)

train_model()


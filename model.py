from ultralytics import YOLO

def train_model():
    model = YOLO("yolo26n.pt")

    model.train(data="data/data.yaml", epochs=100)

train_model()


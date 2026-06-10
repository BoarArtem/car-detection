from ultralytics import YOLO

def train_model():
    model = YOLO("yolo26n.pt")

    model.train(data="data/data.yaml", epochs=20)

if __name__ == "__main__":
    train_model()


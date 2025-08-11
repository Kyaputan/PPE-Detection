import os
from ultralytics import YOLO
from config import WEIGHTS_DIR, MODEL_NAME, MODEL_CONF

def load_model():
    model_path = os.path.join(WEIGHTS_DIR, MODEL_NAME)
    model = YOLO(model_path)
    class_names = {i: str(n).strip() for i, n in model.names.items()}
    return model, class_names

def infer(model, frame):
    return model(frame, conf=MODEL_CONF)[0]
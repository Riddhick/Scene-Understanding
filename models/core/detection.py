import cv2
from collections import defaultdict
import torch
from ultralytics import YOLO
from utils import get_class_color

def load_model(model_name="D:\Work\RCI\Code\models\weights\yolov11_trained.pt"):
    return YOLO(model_name)

def run_detection(model, img_path):
    results = model(img_path)
    detections = results[0].boxes

    class_id_counters = defaultdict(int)
    detected_objects = []

    img = cv2.imread(img_path)

    for box in detections:
        cls_id = int(box.cls.cpu().numpy()[0])
        class_name = model.names[cls_id]
        conf = float(box.conf.cpu().numpy()[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

        if conf < 0.60:
            continue

        # Assign unique ID per class
        class_id_counters[class_name] += 1
        obj_id = class_id_counters[class_name]
        unique_name = f"{class_name} {obj_id}"

        # Add object
        detected_objects.append({
            "name": unique_name,
            "class": class_name,
            "confidence": conf,
            "bbox": (x1, y1, x2, y2),
            "center": ((x1 + x2) // 2, (y1 + y2) // 2),
        })

        # Draw bbox
        color = get_class_color(class_name)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, unique_name, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return img, detected_objects

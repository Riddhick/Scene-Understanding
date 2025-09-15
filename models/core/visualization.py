import cv2
import matplotlib.pyplot as plt
from config import RELATION_COLORS

def draw_scene_graph(image, scene_graph):
    img = image.copy()
    for rel in scene_graph:
        subject, obj, predicate = rel["subject"], rel["object"], rel["predicate"]
        color = RELATION_COLORS.get(predicate, (255, 255, 255))
        cv2.line(img, subject["center"], obj["center"], color, 2)
        mid = ((subject["center"][0] + obj["center"][0]) // 2,
               (subject["center"][1] + obj["center"][1]) // 2)
        cv2.putText(img, predicate, mid, cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 2, cv2.LINE_AA)
    return img

def show_image(img, title="Output"):
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()

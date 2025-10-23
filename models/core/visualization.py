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

def draw_object_angles(image, detected_objects, scene_graph):
    """
    Draws detected objects and visualizes all spatial angles between each pair.
    Each direction is drawn separately (A→B and B→A if both exist).
    Only angles are displayed as labels.
    """
    img = image.copy()

    # Optional: Draw boxes & labels
    for obj in detected_objects:
        x1, y1, x2, y2 = obj["bbox"]
        label = obj["name"]
        color = (0, 255, 0)  # Green
        # Uncomment if you want bounding boxes visible
        # cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        # cv2.putText(img, label, (x1, y1 - 10),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Draw every relationship (including both directions)
    drawn_pairs = set()
    for rel in scene_graph:
        subj = rel["subject"]
        obj = rel["object"]
        angle = rel["angle"]

        sx, sy = subj["center"]
        ox, oy = obj["center"]

        # Allow both directions, but ensure no duplicate same-direction draw
        pair_key = (subj["name"], obj["name"], angle)
        if pair_key in drawn_pairs:
            continue
        drawn_pairs.add(pair_key)

        # Draw arrowed line (subject → object)
        cv2.arrowedLine(img, (sx, sy), (ox, oy), (255, 0, 0), 2, tipLength=0.03)

        # Midpoint for angle annotation
        mid_x = int((sx + ox) / 2)
        mid_y = int((sy + oy) / 2)

        # Label only the angle
        cv2.putText(img, f"{angle} deg", (mid_x + 5, mid_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return img


def show_image(img, title="Output"):
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()

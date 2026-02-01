import cv2
import matplotlib.pyplot as plt
import numpy as np
from config import RELATION_COLORS

# ---------------------- Drawing Utilities ----------------------

def draw_text(img, text, pos, font_scale=0.6, color=(0, 0, 0), thickness=2):
    """Text with outline for better readability in thesis images."""
    # Text outline in white
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), thickness + 2, cv2.LINE_AA)
    # Main text
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness, cv2.LINE_AA)


def draw_scene_graph(image, scene_graph):
    img = image.copy()
    for rel in scene_graph:
        subject, obj, predicate = rel["subject"], rel["object"], rel["predicate"]
        color = RELATION_COLORS.get(predicate, (255, 255, 255))

       # cv2.line(img, subject["center"], obj["center"], color, 3, cv2.LINE_AA)
        cv2.arrowedLine(img, subject["center"], obj["center"],
                        color, 3, cv2.LINE_AA, tipLength=0.04)

        mid = (
            (subject["center"][0] + obj["center"][0]) // 2,
            (subject["center"][1] + obj["center"][1]) // 2
        )
        draw_text(img, predicate, (mid[0] + 5, mid[1] - 5),
                  font_scale=0.65, color=(0, 0, 0), thickness=2)
    return img


def draw_object_angles(image, detected_objects, scene_graph):
    """
    Draws objects + directional spatial angles (A→B).
    Suitable for thesis-quality visualization.
    """
    img = image.copy()

    # Optional: professional bounding boxes
    for obj in detected_objects:
        x1, y1, x2, y2 = obj["bbox"]
        label = obj["name"]
        color = (40, 200, 40)  # pleasant green

        # Uncomment only when needed
        # cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        # draw_text(img, label, (x1, y1 - 8), font_scale=0.6, color=color)

    drawn_pairs = set()

    for rel in scene_graph:
        subj, obj, angle = rel["subject"], rel["object"], rel["angle"]
        sx, sy = subj["center"]
        ox, oy = obj["center"]

        pair_key = (subj["name"], obj["name"], angle)
        if pair_key in drawn_pairs:
            continue
        drawn_pairs.add(pair_key)

        # Arrow with anti-aliasing and smoother tip
        cv2.arrowedLine(img, (sx, sy), (ox, oy),
                        (230, 40, 40), 3, cv2.LINE_AA, tipLength=0.04)

        mid_x = int((sx + ox) / 2)
        mid_y = int((sy + oy) / 2)

        draw_text(img, f"{angle} deg", (mid_x + 6, mid_y - 6),
                  font_scale=0.65, color=(20, 20, 255))

    return img


# ---------------------- Display Utility ----------------------

def show_image(img, title="Output", dpi=180):
    """
    Display thesis-quality output.
    dpi=180-300 recommended when saving images for LaTeX/PDF.
    """
    plt.figure(figsize=(10, 7), dpi=dpi)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title, fontsize=16)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def save_image(img, filename, dpi=300):
    """
    Save publication-quality image, suitable for LaTeX thesis.
    """
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imsave(filename, rgb, dpi=dpi)

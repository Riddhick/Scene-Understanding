# relationships.py
import numpy as np


def determine_spatial_relationship(box1, box2):
    """
    Compute spatial relation of box2 w.r.t box1.
    Angle is measured from horizontal axis through box1 center (0° = right).
    """
    x1a, y1a, x2a, y2a = box1
    x1b, y1b, x2b, y2b = box2

    # centers
    cx1, cy1 = (x1a + x2a) / 2, (y1a + y2a) / 2   # subject center
    cx2, cy2 = (x1b + x2b) / 2, (y1b + y2b) / 2   # object center

    w1, h1 = x2a - x1a, y2a - y1a
    w2, h2 = x2b - x1b, y2b - y1b

    # vector from subject → object
    dx = cx2 - cx1
    dy = cy2 - cy1

    # angle in degrees: 0 = right, 90 = up, 180 = left, 270 = down
    angle = (np.degrees(np.arctan2(-dy, dx)) + 360) % 360

    # y-overlap to judge horizontal vs vertical
    y_overlap = max(0, min(y2a, y2b) - max(y1a, y1b))
    y_overlap_ratio = (y_overlap / min(h1, h2)) if min(h1, h2) > 0 else 0

    # distance threshold for "near"
    dist = np.hypot(dx, dy)
    near_threshold = ((w1 + w2 + h1 + h2) / 4) * 1.5

    if y_overlap_ratio > 0.4:
        return ("left of" if dx < 0 else "right of"), angle

    if abs(dx) < max(w1, w2) * 0.5:
        return ("up" if dy < 0 else "down"), angle

    if dist < near_threshold:
        return "near_to", angle

    return None, angle


def build_scene_graph(detected_objects):
    """
    Build scene graph from detected_objects.

    Parameters
    ----------
    detected_objects : list of dict
        Each dict must contain:
            - name   : unique string (e.g., "car 1")
            - class  : class label
            - bbox   : (x1, y1, x2, y2)
            - center : (cx, cy)

    Returns
    -------
    list of dict
        [
            {
                "subject": {...},
                "predicate": "left of",
                "object": {...},
                "angle": 33.2,
                "description": "truck 1 is at 33.2° left of car 1"
            },
            ...
        ]
    """
    scene_graph = []

    for i, subj in enumerate(detected_objects):
        for j, obj in enumerate(detected_objects):
            if i == j:
                continue

            pred, ang = determine_spatial_relationship(subj["bbox"], obj["bbox"])
            if pred:
                scene_graph.append({
                    "subject": subj,
                    "predicate": pred,
                    "object": obj,
                    "angle": round(ang, 2),
                    "description": (
                        f'{obj.get("name", "obj")} is at '
                        f'{round(ang,1)}° {pred} {subj.get("name","obj")}'
                    )
                })
    return scene_graph

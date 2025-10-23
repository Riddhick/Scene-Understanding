import numpy as np


def determine_spatial_relationship(box1, box2):
    """
    Compute full angular spatial relation of box2 w.r.t box1.
    Returns both angle and a descriptive direction (e.g., 'upper-right').
    """
    x1a, y1a, x2a, y2a = box1
    x1b, y1b, x2b, y2b = box2

    # centers
    cx1, cy1 = (x1a + x2a) / 2, (y1a + y2a) / 2
    cx2, cy2 = (x1b + x2b) / 2, (y1b + y2b) / 2

    # vector from subject → object
    dx = cx2 - cx1
    dy = cy2 - cy1

    # angle in degrees: 0 = right, 90 = up, 180 = left, 270 = down
    angle = (np.degrees(np.arctan2(-dy, dx)) + 360) % 360

    # descriptive direction (8-way)
    if 337.5 <= angle or angle < 22.5:
        direction = "right"
    elif 22.5 <= angle < 67.5:
        direction = "upper-right"
    elif 67.5 <= angle < 112.5:
        direction = "up"
    elif 112.5 <= angle < 157.5:
        direction = "upper-left"
    elif 157.5 <= angle < 202.5:
        direction = "left"
    elif 202.5 <= angle < 247.5:
        direction = "lower-left"
    elif 247.5 <= angle < 292.5:
        direction = "down"
    elif 292.5 <= angle < 337.5:
        direction = "lower-right"
    else:
        direction = "unknown"

    return direction, round(angle, 2)


def build_scene_graph(detected_objects):
    """
    Build a scene graph with full angle and direction,
    storing only one entry per unique object pair (no redundant inverses).
    """
    scene_graph = []
    n = len(detected_objects)

    for i in range(n):
        for j in range(i + 1, n):  # ensures each pair appears only once
            subj = detected_objects[i]
            obj = detected_objects[j]

            # Forward relation: subj → obj
            direction, angle = determine_spatial_relationship(subj["bbox"], obj["bbox"])

            # Optional: also compute inverse for reference
            inv_direction, inv_angle = determine_spatial_relationship(obj["bbox"], subj["bbox"])

            scene_graph.append({
                "subject": subj,
                "object": obj,
                "predicate": direction,
                "angle": angle,
                "inverse_predicate": inv_direction,
                "inverse_angle": inv_angle,
                "description": (
                    f'{obj.get("name", "obj")} is {direction} ({angle}°) relative to '
                    f'{subj.get("name","obj")}'
                )
            })

    return scene_graph

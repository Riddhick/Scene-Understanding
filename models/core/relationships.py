import numpy as np

def determine_spatial_relationship(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    center_x1, center_y1 = (x1_1 + x2_1) / 2, (y1_1 + y2_1) / 2
    center_x2, center_y2 = (x1_2 + x2_2) / 2, (y1_2 + y2_2) / 2

    width1, height1 = x2_1 - x1_1, y2_1 - y1_1
    width2, height2 = x2_2 - x1_2, y2_2 - y1_2

    if x1_1 > x1_2 and y1_1 > y1_2 and x2_1 < x2_2 and y2_1 < y2_2:
        return "inside"

    y_overlap = max(0, min(y2_1, y2_2) - max(y1_1, y1_2))
    if (y_overlap / min(height1, height2)) if min(height1, height2) > 0 else 0 > 0.4:
        return "left of" if center_x1 < center_x2 else "right of"

    distance = np.sqrt((center_x1 - center_x2)**2 + (center_y1 - center_y2)**2)
    nearness_threshold = ((width1 + width2 + height1 + height2) / 4) * 1.5
    if distance < nearness_threshold:
        return "near_to"

    return None

def build_scene_graph(detected_objects):
    scene_graph = []
    for i, subject in enumerate(detected_objects):
        for j, obj in enumerate(detected_objects):
            if i == j:
                continue
            relation = determine_spatial_relationship(subject["bbox"], obj["bbox"])
            if relation:
                scene_graph.append({
                    "subject": subject,
                    "predicate": relation,
                    "object": obj
                })
    return scene_graph
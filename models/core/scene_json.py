import json
from distances import calculate_pixel_distances

def build_scene_json(detected_objects, scene_graph):
    objects = [{
        "id": obj["name"],
        "class": obj["class"],
        "bbox": obj["bbox"],
        "center": obj["center"]
    } for obj in detected_objects]

    distances = calculate_pixel_distances(detected_objects)

    relationships = [{
        "subject": rel["subject"]["name"],
        "predicate": rel["predicate"],
        "object": rel["object"]["name"],
        "angle": rel["angle"],
        "description" : rel["description"]
    } for rel in scene_graph]

    return {
        "objects": objects,
        "distances": distances,
        "relationships": relationships
    }

def save_scene_json(scene_json, filename="scene_output.json"):
    with open(filename, "w") as f:
        json.dump(scene_json, f, indent=4)
import re
import json
import os

COLORS = ["blue", "green", "red", "white", "gray", "black", "yellow"]

DIRECTION_MAP = {
    "up": "up",
    "down": "down",
    "left": "left",
    "right": "right",
    "upper-right": "upper-right",
    "upper left": "upper-left",
    "upper-left": "upper-left",
    "upper right": "upper-right",
    "uppre-right": "upper-right",   # common typo
    "upperright": "upper-right",
    "lowerright": "lower-right",
    "lower-right": "lower-right",
    "lowerleft": "lower-left",
    "lower-left": "lower-left",
}

def normalize_direction(raw):
    raw = raw.lower().strip()
    return DIRECTION_MAP.get(raw, raw)


def extract_spatial_query(text: str):
    text = text.lower().strip()

    result = {
        "target": {"class": infer_target(text)},
        "constraints": []
    }

    clauses = [c.strip() for c in text.split(",")]

    obj_pattern = r"(blue|green|red|white|gray|black|yellow)?\s*(pedestrian|people|bicycle|car|van|truck|tricycle|awning-tricycle|bus|motor)\s*(\d+)?"

    angle_pattern = r"(\d+)\s*degrees?\s*([a-z\-]+)"
    dist_pattern = r"(\d+)\s*(pixels?|pixles|px)\s*([a-z\-]+)"

    for clause in clauses:
        obj_match = re.search(obj_pattern, clause)
        if not obj_match:
            continue

        color = obj_match.group(1)
        obj_class = obj_match.group(2)
        obj_id = obj_match.group(3)

        constraint = {
            "ref": {
                "class": obj_class,
                "id": obj_id if obj_id else None,
                "color": color if color else None
            }
        }

        parts = re.split(r"\band\b", clause)

        for part in parts:
            angle_match = re.search(angle_pattern, part)
            if angle_match:
                constraint["angle"] = {
                    "relation": normalize_direction(angle_match.group(2)),
                    "value_deg": int(angle_match.group(1))
                }

            dist_match = re.search(dist_pattern, part)
            if dist_match:
                constraint["distance"] = {
                    "relation": normalize_direction(dist_match.group(3)),
                    "value_px": int(dist_match.group(1))
                }

        result["constraints"].append(constraint)

    return result


def infer_target(text: str):
    m = re.match(r"(a|an)\s+(\w+)", text)
    if m:
        return m.group(2)
    return "object"

def save_json(data: dict, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"✅ JSON saved to: {save_path}")

text = "A bunker is at 90 degrees left from truck 1, 30 pixles right and 20 degrees uppre-right from pedestrian 2, 10 pixels up and 45 degrees right from pedestrian 1"

result = extract_spatial_query(text)
print(result)
#save_json(result, "./extracted.json")
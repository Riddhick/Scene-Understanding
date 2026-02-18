import re
import json
import os
from typing import Dict

# -----------------------------
# Constants
# -----------------------------

COLORS = ["blue", "green", "red", "white", "gray", "black", "yellow"]

REF_CLASSES = [
    "pedestrian", "people", "bicycle", "car", "van", "truck",
    "tricycle", "awning-tricycle", "bus", "motor"
]

DIRECTION_MAP = {
    "up": "up",
    "down": "down",
    "left": "left",
    "right": "right",
    "upper-right": "upper-right",
    "upper left": "upper-left",
    "upper-left": "upper-left",
    "upper right": "upper-right",
    "uppre-right": "upper-right",
    "upperright": "upper-right",
    "lowerright": "lower-right",
    "lower-right": "lower-right",
    "lowerleft": "lower-left",
    "lower-left": "lower-left",
}

CMP_MAP = {
    "more than": "gt",
    "greater than": "gt",
    "above": "gt",
    "less than": "lt",
    "below": "lt",
    "under": "lt",
    "almost": "approx",
    "around": "approx",
    "approximately": "approx",
    "about": "approx",
    "exactly": "eq",
    None: "eq"
}

UNIT_MAP = {
    "px": "px",
    "pixel": "px",
    "pixels": "px",
    "pixles": "px",   # typo handling
    "cm": "cm",
    "meter": "m",
    "meters": "m",
    "m": "m",
    "km": "km",
    "kilometer": "km",
    "kilometers": "km"
}

# -----------------------------
# Helpers
# -----------------------------

def normalize_direction(raw: str) -> str:
    raw = raw.lower().strip()
    return DIRECTION_MAP.get(raw, raw)

def infer_target(text: str) -> str:
    m = re.search(r"\b(a|an|the)\s+([a-zA-Z0-9_-]+)", text.lower())
    if m:
        return m.group(2)
    return "object"

# -----------------------------
# Main Extractor
# -----------------------------

def extract_spatial_query(text: str) -> Dict:
    text = text.lower().strip()

    result = {
        "target": {"class": infer_target(text)},
        "constraints": []
    }

    clauses = [c.strip() for c in text.split(",")]

    obj_pattern = (
        r"(blue|green|red|white|gray|black|yellow)?\s*"
        r"(" + "|".join(REF_CLASSES) + r")\s*(\d+)?"
    )

    angle_pattern = r"(\d+)\s*degrees?\s*([a-z\-]+)"

    dist_pattern = (
        r"(more than|greater than|less than|below|above|under|almost|around|approximately|about|exactly)?\s*"
        r"(\d+(?:\.\d+)?)\s*(px|pixels?|pixles|cm|m|meters?|km|kilometers?)\s*([a-z\-]+)"
    )

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
                raw_cmp = dist_match.group(1)
                value = float(dist_match.group(2))
                raw_unit = dist_match.group(3)
                direction = normalize_direction(dist_match.group(4))

                unit = UNIT_MAP.get(raw_unit, raw_unit)
                cmp = CMP_MAP.get(raw_cmp, "eq")

                constraint["distance"] = {
                    "cmp": cmp,           # gt, lt, eq, approx
                    "value": value,       # numeric distance
                    "unit": unit,         # px / cm / m / km
                    "direction": direction
                }

        result["constraints"].append(constraint)

    return result

# -----------------------------
# Save JSON Utility
# -----------------------------

def save_json(data: dict, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✅ JSON saved to: {save_path}")

# -----------------------------
# Example
# -----------------------------

if __name__ == "__main__":
    text = (
        "A bunker is at 2 meters right and 17 degrees from the pedestrian 2, "
        "at more than 300 px left from truck 1, "
        "at 0.5 km up from bus 1"
    )

    result = extract_spatial_query(text)
    print(json.dumps(result, indent=2))
    save_json(result, "./extracted.json")

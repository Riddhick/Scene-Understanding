import json
import cv2
import numpy as np

# ==============================
# Base direction angles
# ==============================

BASE_DIR_ANGLES = {
    "right": 0,
    "upper-right": 45,
    "up": 90,
    "upper-left": 135,
    "left": 180,
    "lower-left": 225,
    "down": 270,
    "lower-right": 315,
}

# ==============================
# Spatial Region Generator
# ==============================

class SpatialRegionGenerator:

    def __init__(self, scene_json_path: str, query_json_path: str, image_path: str):
        self.scene = self.load_json(scene_json_path)
        self.query = self.load_json(query_json_path)
        self.image = cv2.imread(image_path)

        if self.image is None:
            raise ValueError("Image not found.")

        self.H, self.W = self.image.shape[:2]

    # --------------------------
    def load_json(self, path):
        with open(path, "r") as f:
            return json.load(f)

    # --------------------------
    def find_reference_object(self, ref):
        for obj in self.scene["objects"]:
            if obj.get("class") != ref.get("class"):
                continue
            if ref.get("id") is not None and str(ref["id"]) not in str(obj.get("id")):
                continue
            return obj
        return None

    # --------------------------
    def resolve_final_angle(self, angle_relation, angle_value_deg, dist_direction):
        if angle_relation in BASE_DIR_ANGLES:
            base_angle = BASE_DIR_ANGLES[angle_relation]
            return (base_angle + (angle_value_deg or 0)) % 360

        if angle_relation == "from":
            if dist_direction in BASE_DIR_ANGLES:
                base_angle = BASE_DIR_ANGLES[dist_direction]
                return (base_angle + (angle_value_deg or 0)) % 360
            else:
                return (angle_value_deg or 0) % 360

        if dist_direction in BASE_DIR_ANGLES:
            return BASE_DIR_ANGLES[dist_direction]

        return None

    # --------------------------
    def draw_angular_sector(self, img, center, abs_angle, spread_deg=40, color=(0,255,255)):
        cx, cy = center

        theta_left = np.deg2rad(abs_angle - spread_deg / 2)
        theta_right = np.deg2rad(abs_angle + spread_deg / 2)

        max_radius = int(np.sqrt(self.W**2 + self.H**2))

        pts = [(cx, cy)]
        for t in np.linspace(theta_left, theta_right, 80):
            pts.append((int(cx + max_radius * np.cos(t)),
                        int(cy - max_radius * np.sin(t))))

        overlay = img.copy()
        cv2.fillPoly(overlay, [np.array(pts, np.int32)], color)
        cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)

    # --------------------------
    def draw_distance_region(self, img, center, abs_angle, cmp, value_px,
                             tolerance_px=15, spread_deg=40, color=(0,255,255)):
        cx, cy = center
        theta_left = np.deg2rad(abs_angle - spread_deg / 2)
        theta_right = np.deg2rad(abs_angle + spread_deg / 2)
        max_radius = int(np.sqrt(self.W**2 + self.H**2))

        overlay = img.copy()

        def sector_polygon(r1, r2):
            pts = []
            for t in np.linspace(theta_left, theta_right, 80):
                pts.append((int(cx + r2 * np.cos(t)), int(cy - r2 * np.sin(t))))
            for t in np.linspace(theta_right, theta_left, 80):
                pts.append((int(cx + r1 * np.cos(t)), int(cy - r1 * np.sin(t))))
            return np.array(pts, np.int32)

        if cmp == "eq":
            theta = np.deg2rad(abs_angle)
            tx = int(cx + np.cos(theta) * value_px)
            ty = int(cy - np.sin(theta) * value_px)
            cv2.circle(img, (tx, ty), 15, color, 2)
            return

        elif cmp == "lt":
            poly = sector_polygon(0, value_px)
            cv2.fillPoly(overlay, [poly], color)

        elif cmp == "gt":
            poly = sector_polygon(value_px, max_radius)
            cv2.fillPoly(overlay, [poly], color)

        elif cmp == "approx":
            r1 = max(0, value_px - tolerance_px)
            r2 = value_px + tolerance_px
            poly = sector_polygon(r1, r2)
            cv2.fillPoly(overlay, [poly], color)

        cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)

    # --------------------------
    def draw_references(self, img):
        for obj in self.scene["objects"]:
            x1, y1, x2, y2 = map(int, obj["bbox"])
            cx, cy = map(int, obj["center"])
            cv2.rectangle(img, (x1, y1), (x2, y2), (255,255,255), 2)
            cv2.circle(img, (cx, cy), 4, (0,255,255), -1)
            cv2.putText(img, obj["id"], (x1, max(0, y1-8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

    # --------------------------
    def visualize_simple(self):
        output = self.image.copy()
        self.draw_references(output)

        colors = [(0,0,255),(0,255,0),(255,0,0),(0,255,255),(255,0,255)]

        for i, constraint in enumerate(self.query.get("constraints", [])):
            ref = constraint.get("ref")
            obj = self.find_reference_object(ref)
            if obj is None:
                continue

            angle_rel = constraint.get("angle", {}).get("relation")
            angle_val = constraint.get("angle", {}).get("value_deg", 0)

            dist_info = constraint.get("distance", {})
            dist_dir  = dist_info.get("direction")
            cmp       = dist_info.get("cmp", "eq")
            val_px    = dist_info.get("value")   # 🔥 updated
            tol_px    = dist_info.get("tolerance_px", 15)

            abs_angle = self.resolve_final_angle(angle_rel, angle_val, dist_dir)

            color = colors[i % len(colors)]

            if abs_angle is not None and val_px is None:
                self.draw_angular_sector(output, obj["center"], abs_angle, 40, color)

            elif cmp == "eq" and val_px is not None:
                self.draw_distance_region(output, obj["center"], abs_angle, cmp, val_px, tol_px, 40, color)

            elif abs_angle is not None and val_px is not None:
                self.draw_angular_sector(output, obj["center"], abs_angle, 40, color)
                self.draw_distance_region(output, obj["center"], abs_angle, cmp, val_px, tol_px, 40, color)

        return output

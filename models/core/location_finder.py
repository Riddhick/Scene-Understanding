import json
import cv2
import numpy as np


# ==============================
# Direction Mapping
# ==============================

DIRECTION_CENTERS = {
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

    def __init__(self, scene_json_path, query_json_path, image_path):
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
        """
        Match object in scene_json using:
        - class (mandatory)
        - id (optional)
        - color (optional)
        """

        for obj in self.scene["objects"]:

            # 1️⃣ Class must match
            if obj.get("class") != ref.get("class"):
                continue

            # 2️⃣ If ID specified in query → enforce numeric match
            if ref.get("id") is not None:
                query_id = str(ref.get("id"))
                obj_id = str(obj.get("id"))

                if query_id not in obj_id:
                    continue

            # 3️⃣ If color specified in query → enforce match
            if ref.get("color") is not None:
                if obj.get("color") != ref.get("color"):
                    continue

            return obj

        return None

    # --------------------------
    def generate_distance_mask(self, ref_center, relation, value_px):

        if value_px is None:
            return np.ones((self.H, self.W), dtype=bool)

        Y, X = np.ogrid[:self.H, :self.W]

        dx = X - ref_center[0]
        dy = Y - ref_center[1]

        distance = np.sqrt(dx**2 + dy**2)

        if relation == "away":
            return distance > value_px

        elif relation == "near":
            return distance < value_px

        return np.ones((self.H, self.W), dtype=bool)

    # --------------------------
    def generate_direction_mask(self, ref_center, relation, value_deg):

        if relation is None:
            return np.ones((self.H, self.W), dtype=bool)

        if relation not in DIRECTION_CENTERS:
            return np.ones((self.H, self.W), dtype=bool)

        center_angle = DIRECTION_CENTERS[relation]

        tolerance = value_deg / 2 if value_deg else 22.5

        Y, X = np.ogrid[:self.H, :self.W]

        dx = X - ref_center[0]
        dy = -(Y - ref_center[1])  # Image → Cartesian correction

        angle = np.degrees(np.arctan2(dy, dx))
        angle = (angle + 360) % 360

        lower = (center_angle - tolerance) % 360
        upper = (center_angle + tolerance) % 360

        if lower < upper:
            mask = (angle >= lower) & (angle <= upper)
        else:
            mask = (angle >= lower) | (angle <= upper)

        return mask

    # --------------------------
    def get_final_region(self):

        masks = []
        missing_refs = []

        for constraint in self.query.get("constraints", []):

            ref = constraint.get("ref")
            obj = self.find_reference_object(ref)

            if obj is None:
                missing_refs.append(ref)
                continue

            ref_center = obj["center"]

            # Default masks (all True)
            direction_mask = np.ones((self.H, self.W), dtype=bool)
            distance_mask = np.ones((self.H, self.W), dtype=bool)

            # Safe angle handling
            if "angle" in constraint:
                angle_info = constraint["angle"]
                direction_mask = self.generate_direction_mask(
                    ref_center,
                    angle_info.get("relation"),
                    angle_info.get("value_deg"),
                )

            # Safe distance handling
            if "distance" in constraint:
                dist_info = constraint["distance"]
                distance_mask = self.generate_distance_mask(
                    ref_center,
                    dist_info.get("relation"),
                    dist_info.get("value_px"),
                )

            combined_mask = direction_mask & distance_mask
            masks.append(combined_mask)

        if len(masks) == 0:
            raise ValueError("No valid references found in scene.")

        final_mask = masks[0]
        for m in masks[1:]:
            final_mask = final_mask & m

        return final_mask, missing_refs
        
    def visualize_constraints(self, final_mask, radius=8):

        output = self.image.copy()

    # =============================
    # Compute Final Target Point
    # =============================
        ys, xs = np.where(final_mask)

        if len(xs) == 0:
            print("⚠ No intersection region found.")
            return output

        fx = int(np.mean(xs))
        fy = int(np.mean(ys))

    # Draw final target
        cv2.circle(output, (fx, fy), radius, (0, 0, 255), -1)
        cv2.putText(output, "TARGET", (fx + 5, fy - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # =============================
    # Draw constraints + distances
    # =============================
        y_offset = 30

        for constraint in self.query.get("constraints", []):

            ref = constraint.get("ref")
            obj = self.find_reference_object(ref)

            if obj is None:
                continue

            cx, cy = obj["center"]

        # Draw reference center
            cv2.circle(output, (cx, cy), 6, (255, 0, 0), -1)
            cv2.putText(output, obj["id"], (cx + 5, cy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Draw line from reference to target
            cv2.line(output, (cx, cy), (fx, fy), (200, 200, 200), 1)

        # =============================
        # Compute Actual Distance
        # =============================
            distance = np.sqrt((fx - cx) ** 2 + (fy - cy) ** 2)

        # Print in console
            print(f"Distance from {obj['id']} to TARGET: {distance:.2f} px")

        # Print on image (top-left stacked)
            text = f"{obj['id']} -> Target: {distance:.1f}px"
            cv2.putText(output, text, (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            y_offset += 25

        # -------------------
        # Angle constraint
        # -------------------
            if "angle" in constraint:

                relation = constraint["angle"].get("relation")
                value_deg = constraint["angle"].get("value_deg")

                if relation in DIRECTION_CENTERS:

                    center_angle = DIRECTION_CENTERS[relation]
                    tolerance = value_deg / 2 if value_deg else 22.5

                    for offset in [-tolerance, tolerance]:
                        angle = np.deg2rad(center_angle + offset)

                        x2 = int(cx + 300 * np.cos(angle))
                        y2 = int(cy - 300 * np.sin(angle))

                        cv2.line(output, (cx, cy), (x2, y2),
                             (0, 255, 0), 2)

        # -------------------
        # Distance constraint
        # -------------------
            if "distance" in constraint:

                value_px = constraint["distance"].get("value_px")

                if value_px is not None:
                    cv2.circle(output, (cx, cy),
                           int(value_px),
                           (0, 255, 255), 2)

        return output
    # --------------------------
    def highlight_region(self, final_mask, radius=8):

    # Get all valid pixel coordinates
        ys, xs = np.where(final_mask)

        if len(xs) == 0:
            print("⚠ No intersection region found.")
            return self.image

    # Compute centroid of intersection
        cx = int(np.mean(xs))
        cy = int(np.mean(ys))

        output = self.image.copy()

    # Draw circle at centroid
        cv2.circle(output, (cx, cy), radius, (0, 0, 255), -1)

        return output


# ==============================
# MAIN EXECUTION
# ==============================

if __name__ == "__main__":

    scene_json_path = r"D:\Work\RCI\Code\scene_output.json"
    query_json_path = r"D:\Work\RCI\Code\models\extracted.json"
    image_path = r"D:\Work\RCI\Code\Sample\0000103_03738_d_0000032.jpg"

    generator = SpatialRegionGenerator(
        scene_json_path,
        query_json_path,
        image_path
    )

    final_mask, missing_refs = generator.get_final_region()

    if missing_refs:
        print("⚠ WARNING: Some references not found in scene:")
        for ref in missing_refs:
            print(ref)

    result_image = generator.visualize_constraints(final_mask)

    cv2.imshow("Possible Target Region", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
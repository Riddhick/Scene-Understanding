import math
import cv2

def compute_object_metrics(image, detected_objects):
    """
    For each detected object, compute:
        - distance from the image center (in pixels)
        - angle (0–360°) with respect to the horizontal axis
          (0° = right, 90° = up, 180° = left, 270° = down)

    Returns:
        List of dicts with object id, distance, and angle.
    """
    h, w = image.shape[:2]
    cx, cy = w // 2, h // 2

    metrics = []
    for obj in detected_objects:
        ox, oy = obj["center"]

        dx = ox - cx
        dy = cy - oy  # invert y to match conventional axis (up = +)

        distance = math.sqrt(dx ** 2 + dy ** 2)
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)
        if angle_deg < 0:
            angle_deg += 360

        metrics.append({
            "object": obj["name"],
            "distance_px": round(distance, 2),
            "angle_deg": round(angle_deg, 2),
        })
    return metrics


def draw_object_metrics(image, metrics, detected_objects):
    """
    Annotate the image with distance and angle for each object.
    """
    annotated = image.copy()
    h, w = annotated.shape[:2]
    cx, cy = w // 2, h // 2

    # Draw center point
    cv2.circle(annotated, (cx, cy), 5, (0, 0, 255), -1)

    for m, obj in zip(metrics, detected_objects):
        ox, oy = obj["center"]
        text = f"{m['distance_px']}px, {m['angle_deg']}deg"
        cv2.line(annotated, (cx, cy), (ox, oy), (255, 0, 0), 1)
        cv2.putText(annotated, text, (ox + 10, oy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
    return annotated

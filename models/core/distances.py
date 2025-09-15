import math
import cv2

def calculate_pixel_distances(detected_objects):
    distances = []
    for i in range(len(detected_objects)):
        for j in range(i + 1, len(detected_objects)):
            obj1, obj2 = detected_objects[i], detected_objects[j]
            (x1, y1), (x2, y2) = obj1["center"], obj2["center"]
            dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            distances.append({
                "object1": obj1["name"],
                "object2": obj2["name"],
                "distance_px": round(dist, 2)
            })
    return distances

def draw_pixel_distances(image, detected_objects):
    annotated = image.copy()
    for i in range(len(detected_objects)):
        for j in range(i + 1, len(detected_objects)):
            obj1, obj2 = detected_objects[i], detected_objects[j]
            (x1, y1), (x2, y2) = obj1["center"], obj2["center"]
            dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.line(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, f"{dist:.1f}px", (mid_x, mid_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return annotated

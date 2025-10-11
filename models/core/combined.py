# -----------------------------------------------------------------------------
#
#       Combined Computer Vision Pipeline
#
# This script combines all modules for object detection, tracking,
# spatial relationship analysis, and scene graph generation for both
# single images and video files.
#
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------
import cv2
import json
import math
import random
import numpy as np
from collections import defaultdict
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# CONFIG (from config.py)
# -----------------------------------------------------------------------------
# Colors for relations
RELATION_COLORS = {
    "left of": (255, 0, 0),    # Blue
    "right of": (0, 255, 0),   # Green
    "near_to": (0, 0, 255),    # Red
    "inside": (255, 255, 0),   # Cyan
    "up": (255, 0, 255),       # Magenta
    "down": (0, 255, 255),     # Yellow
}

# -----------------------------------------------------------------------------
# UTILS (from utils.py)
# -----------------------------------------------------------------------------
class_colors = {}

def get_class_color(class_name: str):
    """Assigns a unique and consistent color for each object class."""
    if class_name not in class_colors:
        class_colors[class_name] = (
            random.randint(50, 255),
            random.randint(50, 255),
            random.randint(50, 255),
        )
    return class_colors[class_name]

# -----------------------------------------------------------------------------
# DISTANCES (from distances.py)
# -----------------------------------------------------------------------------
def calculate_pixel_distances(detected_objects):
    """Calculates the Euclidean distance in pixels between the centers of all detected objects."""
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
    """Draws lines and distance labels between all detected objects on an image."""
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

# -----------------------------------------------------------------------------
# SPATIALREL (from spatialrel.py)
# -----------------------------------------------------------------------------
def determine_spatial_relationship(box1, box2):
    """
    Computes the spatial relation of box2 with respect to box1.
    The angle is measured from the horizontal axis through the center of box1 (0° = right).
    """
    x1a, y1a, x2a, y2a = box1
    x1b, y1b, x2b, y2b = box2

    cx1, cy1 = (x1a + x2a) / 2, (y1a + y2a) / 2
    cx2, cy2 = (x1b + x2b) / 2, (y1b + y2b) / 2

    w1, h1 = x2a - x1a, y2a - y1a
    w2, h2 = x2b - x1b, y2b - y1b

    dx = cx2 - cx1
    dy = cy2 - cy1

    angle = (np.degrees(np.arctan2(-dy, dx)) + 360) % 360

    y_overlap = max(0, min(y2a, y2b) - max(y1a, y1b))
    y_overlap_ratio = (y_overlap / min(h1, h2)) if min(h1, h2) > 0 else 0

    dist = np.hypot(dx, dy)
    near_threshold = ((w1 + w2 + h1 + h2) / 4) * 1.5

    if x1b > x1a and y1b > y1a and x2b < x2a and y2b < y2a:
        return "inside", angle
        
    if y_overlap_ratio > 0.4:
        return ("left of" if dx < 0 else "right of"), angle

    if abs(dx) < max(w1, w2) * 0.5:
        return ("up" if dy < 0 else "down"), angle

    if dist < near_threshold:
        return "near_to", angle

    return None, angle

def build_scene_graph(detected_objects):
    """Builds a scene graph from a list of detected objects."""
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

# -----------------------------------------------------------------------------
# SCENE_JSON (from scene_json.py)
# -----------------------------------------------------------------------------
def build_scene_json(detected_objects, scene_graph):
    """Constructs a JSON-compatible dictionary representing the scene."""
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
    """Saves the scene data to a JSON file."""
    with open(filename, "w") as f:
        json.dump(scene_json, f, indent=4)

# -----------------------------------------------------------------------------
# VISUALIZATION (from visualization.py)
# -----------------------------------------------------------------------------
def draw_scene_graph(image, scene_graph):
    """Draws the relationships from the scene graph onto an image."""
    img = image.copy()
    for rel in scene_graph:
        subject, obj, predicate = rel["subject"], rel["object"], rel["predicate"]
        color = RELATION_COLORS.get(predicate, (255, 255, 255))
        cv2.line(img, subject["center"], obj["center"], color, 2)
        mid = ((subject["center"][0] + obj["center"][0]) // 2,
               (subject["center"][1] + obj["center"][1]) // 2)
        cv2.putText(img, predicate, mid, cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 2, cv2.LINE_AA)
    return img

def show_image(img, title="Output"):
    """Displays an image using Matplotlib."""
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()

# -----------------------------------------------------------------------------
# DETECTION (from detection.py)
# -----------------------------------------------------------------------------
def load_model(model_name="D:\Work\RCI\Code\models\weights\yolov11_trained.pt"):
    """Loads the YOLO object detection and tracking model."""
    print(f"Loading model: {model_name}")
    return YOLO(model_name)

def run_detection(model, img_path):
    """Runs object detection on a single image file."""
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not read image from {img_path}")
        return None, []
    return run_detection_on_frame(model, img, persist_tracking=False)

id_remap = defaultdict(dict)             # maps YOLO orig_id -> per-class remap
next_id_counter = defaultdict(int)       # per-class incremental ID counter
id_class_history = defaultdict(list)     # recent class predictions for each orig_id
last_confidence = defaultdict(float)     # last confidence score for class smoothing
HISTORY_LENGTH = 5                       # number of frames to keep in history

def run_detection_on_frame(model, frame, persist_tracking=True):
    """
    Runs object detection and tracking on a single frame.
    Maintains:
        - Persistent IDs across frames
        - Class smoothing via history majority voting
        - Per-class ID numbering starting from 1
    """
    results = model.track(frame, persist=persist_tracking, verbose=False)
    
    if results[0].boxes is None or results[0].boxes.id is None:
        return frame.copy(), []

    detections = results[0].boxes
    detected_objects = []
    img = frame.copy()

    for box in detections:
        cls_id = int(box.cls.cpu().numpy()[0])
        raw_class = model.names[cls_id]
        conf = float(box.conf.cpu().numpy()[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

        if conf < 0.40:
            continue

        orig_id = int(box.id.cpu().numpy()[0])

        # --- CLASS HISTORY SMOOTHING ---
        # Maintain a rolling history of recent class predictions for this tracker ID
        id_class_history[orig_id].append(raw_class)
        if len(id_class_history[orig_id]) > HISTORY_LENGTH:
            id_class_history[orig_id].pop(0)

        # Compute majority-vote class
        stable_class = max(set(id_class_history[orig_id]),
                           key=id_class_history[orig_id].count)

        # Confidence-based override:
        # If current confidence is significantly higher, allow class switch
        prev_conf = last_confidence.get(orig_id, 0)
        if conf > prev_conf + 0.1:  # tolerate small fluctuations
            stable_class = raw_class
            id_class_history[orig_id] = [stable_class]  # reset history
            last_confidence[orig_id] = conf
        else:
            last_confidence[orig_id] = max(prev_conf, conf)

        # --- CLASS-SPECIFIC ID REMAPPING ---
        if orig_id not in id_remap[stable_class]:
            next_id_counter[stable_class] += 1
            id_remap[stable_class][orig_id] = next_id_counter[stable_class]

        new_id = id_remap[stable_class][orig_id]
        unique_name = f"{stable_class} {new_id}"

        # --- Append detected object ---
        detected_objects.append({
            "name": unique_name,
            "class": stable_class,
            "confidence": conf,
            "bbox": (x1, y1, x2, y2),
            "center": ((x1 + x2) // 2, (y1 + y2) // 2),
        })

        # --- Visualization ---
        color = get_class_color(stable_class)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, unique_name, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return img, detected_objects
# -----------------------------------------------------------------------------
# VIDEO PROCESSING (from video_processing.py)
# -----------------------------------------------------------------------------
def process_video(video_path, model, output_json_path="video_output.json",
                  save_output=True, store_every_n_frames=30):
    """Processes a video with tracking on all frames, stores JSON every N frames,
    and fits display window to screen while maintaining aspect ratio."""
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # --- Video writer setup ---
    out_writer = None
    if save_output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_writer = cv2.VideoWriter("annotated_output.mp4", fourcc, fps, (width, height))

    # --- Get screen size for scaling ---
    screen_w = 1280
    screen_h = 720

    all_frames_data = []
    frame_count = 0

    print("\n--- Starting video processing ---")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection + tracking
        annotated_frame, detected_objects = run_detection_on_frame(model, frame)

        # Build scene graph
        scene_graph = build_scene_graph(detected_objects)
        frame_with_relations = draw_scene_graph(annotated_frame, scene_graph)

        # --- Resize for display (fit-to-screen) ---
        h, w = frame_with_relations.shape[:2]
        scale = min(screen_w / w, screen_h / h)
        disp_w, disp_h = int(w * scale), int(h * scale)
        frame_display = cv2.resize(frame_with_relations, (disp_w, disp_h))

        # --- Show live video ---
        cv2.imshow("Scene Understanding - Video", frame_display)

        # --- Save annotated video ---
        if save_output:
            out_writer.write(frame_with_relations)

        # --- Store JSON every N frames ---
        if frame_count % store_every_n_frames == 0:
            scene_json = build_scene_json(detected_objects, scene_graph)
            scene_json['frame_number'] = frame_count
            all_frames_data.append(scene_json)

        frame_count += 1
        print(f"Processed frame {frame_count}", end="\r")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- Cleanup ---
    cap.release()
    if save_output:
        out_writer.release()
    cv2.destroyAllWindows()

    # --- Save JSON ---
    with open(output_json_path, "w") as f:
        json.dump(all_frames_data, f, indent=4)

    print(f"\n✅ Video processing complete.")
    print(f"➡ Annotated video saved as 'annotated_output.mp4'")
    print(f"➡ JSON saved as '{output_json_path}'")

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main_image():
    """Defines and runs the pipeline for a single image."""
    # !!! IMPORTANT: Update this path to your image file !!!
    img_path = "D:\Work\VisDrone2019-VID-val\\VisDrone2019-VID-val\\sequences\\uav0000137_00458_v\0000001.jpg"
    model = load_model()

    img, detected_objects = run_detection(model, img_path)
    if img is None:
        return

    scene_graph = build_scene_graph(detected_objects)
    img_with_graph = draw_scene_graph(img, scene_graph)
    show_image(img_with_graph, "Scene Graph Visualization")

    scene_json = build_scene_json(detected_objects, scene_graph)
    print("--- Single Image JSON Output ---")
    print(json.dumps(scene_json, indent=4))
    save_scene_json(scene_json, "single_image_output.json")

def main_video():
    """Defines and runs the pipeline for a video file."""
    # !!! IMPORTANT: Update this path to your video file !!!
    video_path = "D:\Work\RCI\Code\output_video.mp4" 
    model = load_model()
    
    process_video(video_path, model, "video_scene_output.json")

if __name__ == "__main__":
    # --- CHOOSE WHICH PIPELINE TO RUN ---
    # To process a single image, uncomment the line below
    #main_image() 
    
    # To process a video, uncomment the line below and update the video_path
    main_video()
    
# -----------------------------------------------------------------------------
#
#       Focused Video Tracking Pipeline with Camera Compensation
#
# This script runs object tracking on a video file and generates a
# temporal JSON summary of all tracked objects, compensating for
# camera movement (ego-motion).
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
# DETECTION (from detection.py)
# -----------------------------------------------------------------------------
def load_model(model_name="D:\Work\RCI\Code\models\weights\yolov11_trained.pt"):
    """Loads the YOLO object detection and tracking model."""
    print(f"Loading model: {model_name}")
    return YOLO(model_name)

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
    results = model.track(frame, persist=True, tracker="botsort.yaml", verbose=False)
    
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

# --- CAM COMP START ---
def transform_point(point, matrix):
    """Applies a 2x3 affine matrix to a (x, y) point."""
    p = np.array([point[0], point[1], 1], dtype=np.float32)
    transformed_p = matrix @ p
    # Return as integer tuple
    return (int(transformed_p[0]), int(transformed_p[1]))
# --- CAM COMP END ---

# -----------------------------------------------------------------------------
# VIDEO PROCESSING (from video_processing.py)
# -----------------------------------------------------------------------------
def compute_direction(start, end):
    """
    Computes the dominant 8-way direction between two compensated points.
    Returns one of:
        'stationary', 'right', 'upper-right', 'up', 'upper-left',
        'left', 'lower-left', 'down', 'lower-right'
    """
    dx = end[0] - start[0]
    dy = end[1] - start[1]

    dist = math.hypot(dx, dy)

    # If movement is too small → stationary
    if dist < 15:
        return "stationary"

    # Angle convention:
    # 0° = right, 90° = up, 180° = left, 270° = down
    angle = (math.degrees(math.atan2(-dy, dx)) + 360) % 360

    # 8-way classification
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

    return direction

def draw_direction_arrow(frame, start, end, color=(0, 255, 255), length_scale=1.0):
    """
    Draws a line + arrowhead from start → end on the frame.
    start, end = (x, y) in compensated coordinate space.
    """
    x1, y1 = start
    x2, y2 = end

    # Draw main direction line
    cv2.line(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

    # Compute arrowhead
    dx = x2 - x1
    dy = y2 - y1
    angle = math.atan2(dy, dx)

    arrow_length = 15 * length_scale
    arrow_angle = 0.5  # radians (~30°)

    # Two arrowhead points
    x3 = int(x2 - arrow_length * math.cos(angle - arrow_angle))
    y3 = int(y2 - arrow_length * math.sin(angle - arrow_angle))

    x4 = int(x2 - arrow_length * math.cos(angle + arrow_angle))
    y4 = int(y2 - arrow_length * math.sin(angle + arrow_angle))

    # Draw the arrowhead
    cv2.line(frame, (x2, y2), (x3, y3), color, 2, cv2.LINE_AA)
    cv2.line(frame, (x2, y2), (x4, y4), color, 2, cv2.LINE_AA)


def compute_displacement_and_angle(start, end):
    dx = end[0] - start[0]
    dy = start[1] - end[1]  # invert screen Y-axis

    dist = math.hypot(dx, dy)
    angle = math.degrees(math.atan2(dy, dx))
    if angle < 0:
        angle += 360
    return dist, angle

def process_video(video_path, model, output_json_path="video_temporal_output2.json",
                  save_output=True, trajectory_sample_n=10):
    """Processes a video, maintaining compact temporal info for each tracked object."""
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps

    # --- Video writer ---
    out_writer = None
    if save_output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter("annotated_output2.mp4", fourcc, fps, (width, height))

    screen_w, screen_h = 1280, 720
    frame_count = 0

    # --- Track object histories ---
    object_tracks = {}  # key: unique_name, list of (frame, compensated_center)
    class_map = {}      # key: unique_name -> class
    print("\n--- Starting video processing (temporal mode) ---")

    # --- CAM COMP START ---
    lk_params = dict(winSize=(21, 21),
                     maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    feature_params = dict(maxCorners=200,
                          qualityLevel=0.01,
                          minDistance=10,
                          blockSize=7)

    prev_gray = None
    M_global = np.eye(2, 3, dtype=np.float32)
    # --- CAM COMP END ---

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --- CAMERA COMPENSATION ---
        M_frame = np.eye(2, 3, dtype=np.float32)

        if prev_gray is not None:
            p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

            if p0 is not None:
                p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, p0, None, **lk_params)
                good_new = p1[st == 1]
                good_old = p0[st == 1]

                if len(good_new) > 6:
                    M, mask = cv2.estimateAffinePartial2D(good_old, good_new, method=cv2.RANSAC)
                    if M is not None:
                        M_frame = M

        # Update global transform
        M_global_3x3 = np.vstack([M_global, [0, 0, 1]])
        M_frame_3x3 = np.vstack([M_frame, [0, 0, 1]])
        M_global_updated = M_frame_3x3 @ M_global_3x3
        M_global = M_global_updated[0:2, :]

        # Inverse for compensation
        M_compensate = cv2.invertAffineTransform(M_global)
        prev_gray = frame_gray.copy()

        # --- DETECTION ---
        annotated_frame, detected_objects = run_detection_on_frame(model, frame)

        # --- TRACKING WITH ARROWS ---
        for obj in detected_objects:
            name = obj["name"]
            center = obj["center"]

            # apply compensation
            compensated_center = transform_point(center, M_compensate)

            # init track if not present
            if name not in object_tracks:
                object_tracks[name] = []
                class_map[name] = obj["class"]

            # --- DRAW ARROW + DISPLACEMENT + ANGLE ---
            if len(object_tracks[name]) > 0:
                prev_center = object_tracks[name][-1][1]
                first_center = object_tracks[name][0][1]

                dist_px, angle_deg = compute_displacement_and_angle(first_center, compensated_center)

                x1, y1, x2, y2 = obj["bbox"]
                #cv2.putText(
                    #annotated_frame,
                    #f"D:{dist_px:.1f}px  A:{angle_deg:.1f} deg",
                    #(x1, y2 + 20),
                   # cv2.FONT_HERSHEY_SIMPLEX,
                   # 0.55,
                    #(0, 255, 255),
                   # 2
                #)

                # draw motion arrow if movement is non-trivial
                #if math.hypot(compensated_center[0] - prev_center[0],
                             # compensated_center[1] - prev_center[1]) > 4:
                    #draw_direction_arrow(annotated_frame, prev_center, compensated_center)

            else:
                # first appearance
                x1, y1, x2, y2 = obj["bbox"]
                #cv2.putText(
                  #  annotated_frame,
                  #  "D:0px A:0°",
                  #  (x1, y2 + 20),
                  #  cv2.FONT_HERSHEY_SIMPLEX,
                  #  0.55,
                  #  (0, 255, 255),
                 #   2
                #)

            # sample trajectory
            if frame_count % trajectory_sample_n == 0:
                object_tracks[name].append((frame_count, compensated_center))

        # --- Visualization ---
        h, w = annotated_frame.shape[:2]
        scale = min(screen_w / w, screen_h / h)
        disp_w, disp_h = int(w * scale), int(h * scale)
        frame_display = cv2.resize(annotated_frame, (disp_w, disp_h))

        cv2.imshow("Scene Understanding - Video", frame_display)
        if save_output:
            out_writer.write(annotated_frame)

        frame_count += 1
        print(f"Processed frame {frame_count}/{total_frames}", end="\r")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- SUMMARIZATION ---
    summarized_objects = []
    for name, trajectory in object_tracks.items():
        if len(trajectory) < 2:
            continue

        entry_frame = trajectory[0][0]
        exit_frame = trajectory[-1][0]
        duration_sec_obj = (exit_frame - entry_frame) / fps

        total_dist = 0
        for i in range(1, len(trajectory)):
            p1, p2 = trajectory[i - 1][1], trajectory[i][1]
            total_dist += math.hypot(p2[0] - p1[0], p2[1] - p1[1])

        avg_speed = total_dist / duration_sec_obj if duration_sec_obj > 0 else 0
        direction = compute_direction(trajectory[0][1], trajectory[-1][1])
        stationary = direction == "stationary"

        summarized_objects.append({
            "id": name,
            "class": class_map[name],
            "entry_frame": entry_frame,
            "exit_frame": exit_frame,
            "entry_time_sec": round(entry_frame / fps, 2),
            "exit_time_sec": round(exit_frame / fps, 2),
            "duration_sec": round(duration_sec_obj, 2),
            "trajectory": [
                {"frame": f, "x": c[0], "y": c[1]} for f, c in trajectory
            ],
            "avg_speed_px_per_sec": round(avg_speed, 2),
            "direction": direction,
            "stationary": stationary
        })

    video_summary = {
        "video_metadata": {
            "video_name": video_path.split("\\")[-1],
            "fps": fps,
            "frame_width": width,
            "frame_height": height,
            "total_frames": total_frames,
            "duration_sec": round(duration_sec, 2)
        },
        "objects": summarized_objects
    }

    with open(output_json_path, "w") as f:
        json.dump(video_summary, f, indent=4)

    cap.release()
    if save_output:
        out_writer.release()
    cv2.destroyAllWindows()

    print(f"\n✅ Video processing complete.")
    print(f"➡ Annotated video saved as 'annotated_output.mp4'")
    print(f"➡ JSON summary saved as '{output_json_path}'")


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main_video():
    """Defines and runs the pipeline for a video file."""
    # !!! IMPORTANT: Update this path to your video file !!!
    video_path = "D:\\Work\\RCI\\Presentation\\video2.mp4" 
    
    # Model path from the original load_model() function default
    model_path = "D:\Work\RCI\Code\models\weights\yolov11_trained.pt"
    
    model = load_model(model_path)
    
    process_video(video_path, model, "video_scene_output2.json")

if __name__ == "__main__":
    # This script is now focused only on video processing.
    main_video()
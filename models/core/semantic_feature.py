import cv2
import numpy as np
import json
from skimage.feature import local_binary_pattern
from sklearn.cluster import MiniBatchKMeans
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

# --- CORE FEATURE EXTRACTION (CPU) ---
    
def compute_sobel_mag(gray):
    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(sx, sy)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return mag.astype(np.uint8)

def extract_semantic_features_cpu(crop, img_shape, debug=False):
    h, w = crop.shape[:2]
    img_h, img_w = img_shape[:2]

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # Geometry logic - Cast to float for JSON safety
    rel_area = float((h * w) / (img_h * img_w))
    aspect_ratio = float(w / (h + 1e-6))

    # Color stats
    brightness = float(np.mean(hsv[..., 2]))
    saturation = float(np.mean(hsv[..., 1]))

    # Texture logic
    edges = cv2.Canny(gray, 100, 200)
    edge_density = float(np.sum(edges > 0) / (h * w))
    texture_var = float(np.var(gray))

    # LBP Texture
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10))
    lbp_hist = (hist / (hist.sum() + 1e-6)).astype(float).round(3).tolist()

    out = {
        "rel_area": round(rel_area, 4),
        "aspect_ratio": round(aspect_ratio, 3),
        "brightness": round(brightness, 2),
        "saturation": round(saturation, 2),
        "edge_density": round(edge_density, 4),
        "texture_var": round(texture_var, 2),
        "lbp": lbp_hist,
        "_bgr_pixels": crop.reshape(-1, 3) 
    }

    if debug:
        out["_crop"] = crop
        out["_sobel"] = compute_sobel_mag(gray)

    return out

# --- COLOR PALETTE LOGIC (LAB SPACE) ---

def sklearn_batched_kmeans_palettes_lab(bgr_batches, num_colors=3, max_pixels=5000, min_percent=0.02):
    palettes = {}

    for idx, pixels in bgr_batches.items():
        if len(pixels) > max_pixels:
            pixels = pixels[np.random.choice(len(pixels), max_pixels, replace=False)]

        pixels_bgr_img = pixels.reshape(-1, 1, 3).astype(np.uint8)
        pixels_lab = cv2.cvtColor(pixels_bgr_img, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)

        kmeans = MiniBatchKMeans(
            n_clusters=num_colors,
            batch_size=2048,
            max_iter=100,
            n_init=3,
            reassignment_ratio=0.01,
            random_state=42
        )

        labels = kmeans.fit_predict(pixels_lab)
        centers = kmeans.cluster_centers_

        palette = []
        total = len(labels)
        counts = np.bincount(labels)

        for c in range(len(centers)):
            percent = counts[c] / total
            if percent < min_percent:
                continue

            lab_color = np.uint8([[centers[c]]])
            bgr_color = cv2.cvtColor(lab_color, cv2.COLOR_LAB2BGR)
            hsv_color = cv2.cvtColor(bgr_color, cv2.COLOR_BGR2HSV)[0, 0]

            # FIX: Convert hsv_color array to standard Python list
            # and ensure each color component is a standard int
            palette.append({
                "hsv": [int(x) for x in hsv_color.tolist()],
                "percent": round(float(percent), 3)
            })

        palette.sort(key=lambda x: x["percent"], reverse=True)
        palettes[idx] = palette

    return palettes

# --- ORCHESTRATION ---

def cpu_parallel_stage(img, detected_objects, debug=False, max_workers=None):
    if max_workers is None:
        max_workers = min(8, os.cpu_count())

    img_shape = img.shape
    results = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(extract_semantic_features_cpu, 
                            img[obj["bbox"][1]:obj["bbox"][3], obj["bbox"][0]:obj["bbox"][2]], 
                            img_shape, debug): i 
            for i, obj in enumerate(detected_objects)
        }

        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()
    return results

def add_semantic_features_hybrid(img, detected_objects, debug=False):
    cpu_features = cpu_parallel_stage(img, detected_objects, debug=debug)
    bgr_batches = {idx: feats.pop("_bgr_pixels") for idx, feats in cpu_features.items()}
    color_palettes = sklearn_batched_kmeans_palettes_lab(bgr_batches)

    for i, obj in enumerate(detected_objects):
        semantic = cpu_features.get(i, {})
        semantic["color_palette"] = color_palettes.get(i, [])
        
        if not debug:
            semantic.pop("_crop", None)
            semantic.pop("_sobel", None)
        
        obj["semantic"] = semantic

    return detected_objects

def visualize_semantic_results(img, detected_objects):
    viz_img = img.copy()
    
    for obj in detected_objects:
        x1, y1, x2, y2 = obj["bbox"]
        semantic = obj.get("semantic", {})
        palette = semantic.get("color_palette", [])
        
        # 1. Draw Bounding Box
        cv2.rectangle(viz_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 2. Draw Color Palette (Visual Verification of LAB Clustering)
        # We draw small squares next to the object to show its dominant colors
        for i, color_info in enumerate(palette):
            hsv = np.uint8([[color_info["hsv"]]])
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0].tolist()
            
            start_point = (x2 + 5, y1 + (i * 25))
            end_point = (x2 + 30, y1 + (i * 25) + 20)
            
            # Draw color swatch
            cv2.rectangle(viz_img, start_point, end_point, bgr, -1)
            cv2.rectangle(viz_img, start_point, end_point, (255, 255, 255), 1)
            
            # Add percentage text
            label = f"{int(color_info['percent']*100)}%"
            cv2.putText(viz_img, label, (x2 + 35, y1 + (i * 25) + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # 3. Add Semantic Label (Brightness/Texture Var)
        label = f"Var: {int(semantic.get('texture_var', 0))}"
        cv2.putText(viz_img, label, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return viz_img

# Example Usage:
# results = add_semantic_features_hybrid(img, detected_objects, debug=False)
# final_view = visualize_semantic_results(img, results)
# cv2.imshow("Semantic Scene Understanding", final_view)
# cv2.waitKey(0)
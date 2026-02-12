import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.cluster import MiniBatchKMeans
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

def extract_semantic_features_cpu(crop, img_shape, debug=False):
    h, w = crop.shape[:2]
    img_h, img_w = img_shape[:2]

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # Geometry
    rel_area = (h * w) / (img_h * img_w)
    aspect_ratio = w / (h + 1e-6)

    # Color stats
    brightness = float(np.mean(hsv[..., 2]))
    saturation = float(np.mean(hsv[..., 1]))

    # Texture
    edges = cv2.Canny(gray, 100, 200)
    edge_density = float(np.sum(edges > 0) / (h * w))
    texture_var = float(np.var(gray))

    # LBP
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10))
    lbp_hist = (hist / (hist.sum() + 1e-6)).round(3).tolist()

    out = {
        "rel_area": round(rel_area, 4),
        "aspect_ratio": round(aspect_ratio, 3),
        "brightness": round(brightness, 2),
        "saturation": round(saturation, 2),
        "edge_density": round(edge_density, 4),
        "texture_var": round(texture_var, 2),
        "lbp": lbp_hist,
        
        "_hsv_pixels": hsv.reshape(-1, 3)
    }

    if debug:
        out["_crop"] = crop
        out["_sobel"] = compute_sobel_mag(gray)

    return out

def cpu_parallel_stage(img, detected_objects, debug=False, max_workers=None):
    if max_workers is None:
        max_workers = min(8, os.cpu_count())

    img_shape = img.shape
    results = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}

        for i, obj in enumerate(detected_objects):
            x1, y1, x2, y2 = obj["bbox"]
            crop = img[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            futures[
                executor.submit(
                    extract_semantic_features_cpu,
                    crop,
                    img_shape,
                    debug      # ✅ PASS DEBUG
                )
            ] = i

        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()

    return results


def build_debug_frame(cpu_features, tile_size=256):
    tiles = []

    for feats in cpu_features.values():
        if "_crop" not in feats:
            continue

        crop = feats["_crop"]
        sobel = feats["_sobel"]

        crop_resized = cv2.resize(crop, (tile_size, tile_size))
        sobel_resized = cv2.resize(sobel, (tile_size, tile_size))
        sobel_rgb = cv2.cvtColor(sobel_resized, cv2.COLOR_GRAY2BGR)

        tile = np.hstack([crop_resized, sobel_rgb])
        tiles.append(tile)

    if not tiles:
        return None

    # grid size
    cols = int(np.ceil(np.sqrt(len(tiles))))
    rows = int(np.ceil(len(tiles) / cols))

    blank = np.zeros_like(tiles[0])
    grid = []

    for r in range(rows):
        row = tiles[r * cols:(r + 1) * cols]
        if len(row) < cols:
            row += [blank] * (cols - len(row))
        grid.append(np.hstack(row))

    return np.vstack(grid)


def compute_sobel_mag(gray):
    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(sx, sy)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return mag.astype(np.uint8)

def sklearn_batched_kmeans_palettes(
    hsv_batches,
    num_colors=3,
    max_pixels=5000,
    min_percent=0.02
):
    palettes = {}

    for idx, pixels in hsv_batches.items():
        if len(pixels) > max_pixels:
            pixels = pixels[np.random.choice(len(pixels), max_pixels, replace=False)]

        pixels = pixels.astype(np.float32)

        kmeans = MiniBatchKMeans(
            n_clusters=num_colors,
            batch_size=2048,
            max_iter=100,
            n_init=3,
            reassignment_ratio=0.01,
            random_state=42
        )


        labels = kmeans.fit_predict(pixels)
        centers = kmeans.cluster_centers_

        palette = []
        total = len(labels)

        for c in range(num_colors):
            count = np.sum(labels == c)
            percent = count / total

            if percent < min_percent:
                continue

            palette.append({
                "hsv": centers[c].astype(int).tolist(),
                "percent": round(float(percent), 3)
            })

        palette.sort(key=lambda x: x["percent"], reverse=True)
        palettes[idx] = palette

    return palettes

def _show_debug_frame(cpu_features, window_name="Semantic Debug View"):
    debug_frame = build_debug_frame(cpu_features)
    if debug_frame is None:
        return

    cv2.imshow(window_name, debug_frame)
    cv2.waitKey(1)
    #cv2.destroyWindow(window_name)   # non-blocking; change to 0 if you want pause


def add_semantic_features_hybrid(img, detected_objects, debug=False):
    # ---------- CPU parallel stage ----------
    cpu_features = cpu_parallel_stage(
        img, 
        detected_objects, 
        debug=debug
    )

    # ---------- Collect HSV pixels ----------
    hsv_batches = {
        idx: feats.pop("_hsv_pixels") 
        for idx, feats in cpu_features.items()
    }

    # ---------- sklearn KMeans ----------
    color_palettes = sklearn_batched_kmeans_palettes(hsv_batches)

    # ✅ MOVE DEBUG CALL HERE (Before data is popped)
    if debug:
        _show_debug_frame(cpu_features)

    # ---------- Merge semantic output ----------
    for i, obj in enumerate(detected_objects):
        semantic = cpu_features.get(i, {})
        semantic["color_palette"] = color_palettes.get(i, [])
        
        # These lines remove the images from the dictionary to keep the JSON output clean
        semantic.pop("_crop", None) 
        semantic.pop("_sobel", None)
        obj["semantic"] = semantic

    return detected_objects
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.cluster import KMeans


def extract_color_palette_hsv(crop,num_colors=3,sample_pixels=5000):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape(-1, 3)

    # Optional subsampling (important for speed)
    if len(pixels) > sample_pixels:
        idx = np.random.choice(len(pixels), sample_pixels, replace=False)
        pixels = pixels[idx]

    # KMeans in HSV space
    kmeans = KMeans(
        n_clusters=num_colors,
        n_init=10,
        random_state=42
    ).fit(pixels)

    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    palette = []
    total = len(labels)

    for i in range(num_colors):
        count = np.sum(labels == i)
        percent = count / total

        if percent < 0.02:
            continue  # drop negligible colors

        hsv_center = centers[i].astype(int).tolist()

        palette.append({
            "hsv": hsv_center,
            "percent": round(float(percent), 3)
        })

    # Sort by dominance
    palette.sort(key=lambda x: x["percent"], reverse=True)

    return palette


def extract_semantic_features(crop, img_shape):
    h, w = crop.shape[:2]
    img_h, img_w = img_shape[:2]

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # ---------------- Geometry ----------------
    rel_area = (h * w) / (img_h * img_w)
    aspect_ratio = w / (h + 1e-6)

    # ---------------- HSV Histograms ----------------
    h_channel, s_channel, v_channel = cv2.split(hsv)

    color_palette = extract_color_palette_hsv(crop, num_colors=3)

    brightness = float(np.mean(v_channel))
    saturation = float(np.mean(s_channel))

    # ---------------- Texture ----------------
    edges = cv2.Canny(gray, 100, 200)
    edge_density = float(np.sum(edges > 0) / (h * w))

    texture_var = float(np.var(gray))

    # ---------------- LBP ----------------
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10))
    lbp_hist = (hist / (hist.sum() + 1e-6)).round(3).tolist()

    return {
        "rel_area": round(rel_area, 4),
        "aspect_ratio": round(aspect_ratio, 3),

       "color_palette": color_palette,

        "brightness": round(brightness, 2),
        "saturation": round(saturation, 2),

        "edge_density": round(edge_density, 4),
        "texture_var": round(texture_var, 2),
        "lbp": lbp_hist
    }

def show_crop_and_sobel(crop, window_prefix="obj"):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Sobel gradients
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_mag = cv2.normalize(sobel_mag, None, 0, 255, cv2.NORM_MINMAX)
    sobel_mag = sobel_mag.astype(np.uint8)

    cv2.imshow(f"{window_prefix}_crop", crop)
    cv2.imshow(f"{window_prefix}_sobel", sobel_mag)

    # Press any key to move to next object
    cv2.waitKey(0)
    cv2.destroyWindow(f"{window_prefix}_crop")
    cv2.destroyWindow(f"{window_prefix}_sobel")


def add_semantic_features(img, detected_objects, debug=False):
    img_shape = img.shape

    for i, obj in enumerate(detected_objects):
        x1, y1, x2, y2 = obj["bbox"]
        crop = img[y1:y2, x1:x2]

        if crop.size == 0:
            obj["semantic"] = {}
            continue

        if debug:
            show_crop_and_sobel(crop, window_prefix=f"obj_{i}")

        obj["semantic"] = extract_semantic_features(crop, img_shape)

    return detected_objects

import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import json

from detection import load_model, run_detection
from spatialrel import build_scene_graph
from visualization import draw_scene_graph, draw_object_angles
from object_metrics import compute_object_metrics, draw_object_metrics
from scene_json import build_scene_json, save_scene_json
from semantic_feature import add_semantic_features_hybrid, visualize_semantic_results
from query_extractor_unit import extract_spatial_query
from area_finder_unit import SpatialRegionGenerator


# ----------------------------
# Utils
# ----------------------------

def compute_gsd_cm_per_pixel(
    altitude_m,
    sensor_width_mm,
    sensor_height_mm,
    focal_length_mm,
    image_width_px,
    image_height_px
):
    altitude_cm = altitude_m * 100.0
    sensor_width_cm = sensor_width_mm / 10.0
    sensor_height_cm = sensor_height_mm / 10.0
    focal_length_cm = focal_length_mm / 10.0

    gsd_x = (altitude_cm * sensor_width_cm) / (focal_length_cm * image_width_px)
    gsd_y = (altitude_cm * sensor_height_cm) / (focal_length_cm * image_height_px)

    return (gsd_x + gsd_y) / 2.0


def real_distance_to_pixels(value, unit, gsd_cm_per_pixel):
    if unit == "px":
        return float(value)

    if unit == "cm":
        value_cm = value
    elif unit == "m":
        value_cm = value * 100.0
    elif unit == "km":
        value_cm = value * 100000.0
    else:
        raise ValueError(f"Unsupported unit: {unit}")

    return value_cm / gsd_cm_per_pixel


def save_temp_image(uploaded_file):
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    tfile.write(uploaded_file.read())
    return tfile.name


def cv2_to_streamlit(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ----------------------------
# Streamlit Setup
# ----------------------------
st.set_page_config(page_title="Spatial Query Visualizer", layout="wide")
st.title("🛰️ UAV Spatial Scene Understanding Demo")

# ----------------------------
# Persistent State
# ----------------------------
if "detected_objects" not in st.session_state:
    st.session_state.detected_objects = None

if "scene_json_path" not in st.session_state:
    st.session_state.scene_json_path = None

if "img_path" not in st.session_state:
    st.session_state.img_path = None

if "img_det" not in st.session_state:
    st.session_state.img_det = None


# ----------------------------
# Load Model (cached)
# ----------------------------
@st.cache_resource
def load_detector():
    return load_model()

model = load_detector()


# ----------------------------
# Upload Image
# ----------------------------
uploaded_file = st.file_uploader("📤 Upload / Drag & Drop an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img_path = save_temp_image(uploaded_file)
    image = cv2.imread(img_path)

    img_h, img_w = image.shape[:2]

    st.subheader("📸 Input Image")
    st.image(cv2_to_streamlit(image), use_container_width=True)

    st.sidebar.markdown("### 🖼️ Image Properties (Auto-filled)")
    st.sidebar.text(f"Width : {img_w} px")
    st.sidebar.text(f"Height: {img_h} px")

    # ----------------------------
    # Run Detection
    # ----------------------------
    if st.button("🚀 Run Detection"):
        img_det, detected_objects = run_detection(model, img_path)
        detected_objects = add_semantic_features_hybrid(img_det, detected_objects, debug=False)

        st.session_state.detected_objects = detected_objects
        st.session_state.img_det = img_det
        st.session_state.img_path = img_path

        # Scene graph + JSON
        scene_graph = build_scene_graph(detected_objects)
        scene_json = build_scene_json(detected_objects, scene_graph)

        scene_json_path = os.path.join(tempfile.gettempdir(), "scene_output.json")
        save_scene_json(scene_json, scene_json_path)
        st.session_state.scene_json_path = scene_json_path

        st.success("✅ Detection and scene graph generated!")

# ----------------------------
# Visualization After Detection
# ----------------------------
if st.session_state.detected_objects is not None:
    detected_objects = st.session_state.detected_objects
    img_det = st.session_state.img_det

    st.subheader("🧠 Detections + Semantic Features")
    semantic_vis = visualize_semantic_results(img_det.copy(), detected_objects)
    st.image(cv2_to_streamlit(semantic_vis), use_container_width=True)

    scene_graph = build_scene_graph(detected_objects)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🔗 Scene Graph")
        graph_vis = draw_scene_graph(img_det.copy(), scene_graph)
        st.image(cv2_to_streamlit(graph_vis), use_container_width=True)

    with col2:
        st.subheader("📐 Object Angles")
        angle_vis = draw_object_angles(img_det.copy(), detected_objects, scene_graph)
        st.image(cv2_to_streamlit(angle_vis), use_container_width=True)

    metrics = compute_object_metrics(img_det, detected_objects)
    metric_vis = draw_object_metrics(img_det.copy(), metrics, detected_objects)

    st.subheader("📏 Object Metrics")
    st.image(cv2_to_streamlit(metric_vis), use_container_width=True)

    # ----------------------------
    # Query Input
    # ----------------------------
    st.subheader("✍️ Enter Spatial Query")
    text_query = st.text_input(
        "Example:",
        "A person is at 10 meters left and 45 degrees from the truck 1"
    )

    st.sidebar.subheader("📐 Camera Geometry (Optional)")
    altitude_m = st.sidebar.number_input("UAV Altitude (meters)", min_value=1.0, value=50.0, step=1.0)
    focal_length_mm = st.sidebar.number_input("Camera Focal Length (mm)", min_value=1.0, value=8.8, step=0.1)
    sensor_width_mm = st.sidebar.number_input("Sensor Width (mm)", min_value=1.0, value=13.2, step=0.1)
    sensor_height_mm = st.sidebar.number_input("Sensor Height (mm)", min_value=1.0, value=8.8, step=0.1)

    if st.button("🎯 Run Spatial Query"):
        if st.session_state.scene_json_path is None:
            st.error("Please run detection first.")
        else:
            # 🔥 Compute GSD
            gsd_cm_per_pixel = compute_gsd_cm_per_pixel(
                altitude_m=altitude_m,
                sensor_width_mm=sensor_width_mm,
                sensor_height_mm=sensor_height_mm,
                focal_length_mm=focal_length_mm,
                image_width_px=img_w,
                image_height_px=img_h
            )

            st.sidebar.success(f"📏 Computed GSD: {gsd_cm_per_pixel:.3f} cm/pixel")

            query_json = extract_spatial_query(text_query)

            # 🔥 Convert all distances to pixels if unit != px
            for c in query_json.get("constraints", []):
                if "distance" in c:
                    dist = c["distance"]
                    unit = dist.get("unit", "px")
                    value = dist.get("value", None)

                    if value is not None:
                        if unit != "px":
                            value_px = real_distance_to_pixels(value, unit, gsd_cm_per_pixel)
                            dist["value"] = float(value_px)
                            dist["unit"] = "px"
                        else:
                            dist["value"] = float(value)
                            dist["unit"] = "px"

            query_json_path = os.path.join(tempfile.gettempdir(), "query_output.json")
            with open(query_json_path, "w") as f:
                json.dump(query_json, f, indent=4)

            generator = SpatialRegionGenerator(
                st.session_state.scene_json_path,
                query_json_path,
                st.session_state.img_path
            )

            region_vis = generator.visualize_simple()

            st.subheader("🗺️ Spatial Constraint Region")
            st.image(cv2_to_streamlit(region_vis), use_container_width=True)

else:
    st.info("⬆️ Upload an image and run detection to begin.")

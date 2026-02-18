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
from query_extractor import extract_spatial_query
from area_finder import SpatialRegionGenerator


# ----------------------------
# Utils
# ----------------------------
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

    st.subheader("📸 Input Image")
    st.image(cv2_to_streamlit(image), use_container_width=True)

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
        "A person is at 956 pixels left and 45 degrees from the truck 1"
    )

    if st.button("🎯 Run Spatial Query"):
        if st.session_state.scene_json_path is None:
            st.error("Please run detection first.")
        else:
            query_json = extract_spatial_query(text_query)

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

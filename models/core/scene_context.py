import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
import joblib
import hdbscan
from transformers import AutoImageProcessor, AutoModel


class SceneContextClassifier:

    def __init__(
        self,
        dino_model_path,
        pipeline_path,
        yolo_class_names,
        device=None
    ):

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # ---- Load DINO ----
        self.processor = AutoImageProcessor.from_pretrained(
            dino_model_path,
            local_files_only=True,
            use_fast=True
        )

        self.dino_model = AutoModel.from_pretrained(
            dino_model_path,
            local_files_only=True
        ).to(self.device)

        self.dino_model.eval()

        # ---- Load clustering pipeline ----
        pipeline = joblib.load(pipeline_path)
        self.scaler = pipeline["scaler"]
        self.reducer = pipeline["umap"]
        self.clusterer = pipeline["hdbscan"]

        # ---- YOLO class mapping ----
        self.class_name_to_id = {
            name: idx for idx, name in yolo_class_names.items()
        }
        self.num_classes = len(yolo_class_names)

        # ---- Cluster label map ----
        self.cluster_label_map = {
            -1: "Visually ambiguous or structurally transitional urban scenes",
            0: "Large organized recreational or athletic ground areas",
            1: "Low-light urban street scenes with artificial illumination",
            2: "Open pedestrian-dominant civic or plaza spaces",
            3: "High-density multi-lane commercial traffic corridors",
            4: "Wide roadways with prominent sky and sparse surroundings",
            5: "Complex elevated highway interchanges and flyover systems",
            6: "Mid-rise residential apartment block clusters with parking",
            7: "Organized institutional or campus-style building compounds",
            8: "Tree-lined residential urban streets with moderate activity",
            9: "Dense mixed-use commercial zones with irregular layouts",
            10: "Linear roadside commercial buildings with adjacent parking",
            11: "Top-down geometric urban road intersection layouts",
            12: "Active construction or transitional urban infrastructure areas",
            13: "Divided highways running through green suburban corridors",
            14: "Signal-controlled busy urban daytime intersections",
            15: "Large open parking lots with clustered vehicles",
            16: "Broad symmetric urban boulevards with median vegetation",
            17: "Commercial urban streets with dense roadside tree canopy"
        }

    # ---------------------------------------------------------
    # Extract object-structure features from detected_objects
    # ---------------------------------------------------------
    def _extract_detection_features(self, detected_objects):

        if len(detected_objects) == 0:
            return np.zeros(self.num_classes + 4)

        hist = np.zeros(self.num_classes)
        areas = []
        aspect_ratios = []

        for obj in detected_objects:
            class_name = obj["class"]

            if class_name not in self.class_name_to_id:
                continue

            cls_id = self.class_name_to_id[class_name]
            hist[cls_id] += 1

            x1, y1, x2, y2 = obj["bbox"]
            width = x2 - x1
            height = y2 - y1

            areas.append(width * height)
            aspect_ratios.append(width / (height + 1e-6))

        if len(areas) == 0:
            return np.zeros(self.num_classes + 4)

        hist = hist / (hist.sum() + 1e-6)

        mean_area = np.mean(areas)
        std_area = np.std(areas)
        mean_aspect = np.mean(aspect_ratios)
        object_count = len(areas)

        stats = np.array([mean_area, std_area, mean_aspect, object_count])

        return np.concatenate([hist, stats])

    # ---------------------------------------------------------
    # Main inference function
    # ---------------------------------------------------------
    def predict(self, image_path, detected_objects):

        # ---- DINO features ----
        image = Image.open(image_path).convert("RGB")

        inputs = self.processor(images=[image], return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.dino_model(**inputs)

        cls = outputs.last_hidden_state[:, 0]
        cls = F.normalize(cls, p=2, dim=1)
        dino_features = cls.cpu().numpy()[0]

        # ---- Detection-based structural features ----
        detection_features = self._extract_detection_features(detected_objects)

        # ---- Combine ----
        combined = np.concatenate([dino_features, detection_features])
        combined = combined.reshape(1, -1)

        # ---- Transform through pipeline ----
        combined_scaled = self.scaler.transform(combined)
        combined_umap = self.reducer.transform(combined_scaled)

        cluster_id, strength = hdbscan.approximate_predict(
            self.clusterer,
            combined_umap
        )

        cluster_id = int(cluster_id[0])
        confidence = float(strength[0])

        scene_label = self.cluster_label_map.get(
            cluster_id,
            "Unknown or undefined scene category"
        )

        return {
            "cluster_id": cluster_id,
            "scene_label": scene_label,
            "confidence": confidence
        }
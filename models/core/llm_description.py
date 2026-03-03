# =========================================
# Q-TOON + GGUF LLaMA INFERENCE MODULE
# (Returns ONLY description)
# =========================================

import json
from typing import Dict, Any, Union
from collections import defaultdict
from llama_cpp import Llama


# =========================================
# Configuration
# =========================================

class QTOONConfig:
    K_NEAREST = 3
    FLOAT_PRECISION = 4
    N_CTX = 4096
    N_THREADS = 8


# =========================================
# Q-TOON Generator
# =========================================

class QTOONGenerator:

    def __init__(self, scene_json: Dict[str, Any]):
        self.scene = scene_json

    def _round(self, value):
        if isinstance(value, float):
            return round(value, QTOONConfig.FLOAT_PRECISION)
        return value

    def _select_knn_relations(self):

        objects = self.scene.get("objects", [])
        distances = self.scene.get("distances", [])
        relationships = self.scene.get("relationships", [])

        adjacency = defaultdict(list)
        distance_map = {}

        # Build adjacency and fast lookup map
        for d in distances:
            obj1 = d["object1"]
            obj2 = d["object2"]
            dist = d["distance_px"]

            adjacency[obj1].append((obj2, dist))
            adjacency[obj2].append((obj1, dist))

            key = tuple(sorted([obj1, obj2]))
            distance_map[key] = dist

        selected_pairs = set()

        # Select K nearest neighbors
        for obj in objects:
            obj_id = obj["id"]
            neighbors = sorted(adjacency[obj_id], key=lambda x: x[1])

            for neighbor_id, _ in neighbors[:QTOONConfig.K_NEAREST]:
                pair = tuple(sorted([obj_id, neighbor_id]))
                selected_pairs.add(pair)

        filtered_relations = []

        for rel in relationships:
            pair = tuple(sorted([rel["subject"], rel["object"]]))
            if pair in selected_pairs:
                rel_copy = rel.copy()
                rel_copy["distance_px"] = distance_map.get(pair)
                filtered_relations.append(rel_copy)

        return filtered_relations

    def generate(self) -> str:

        objects = self.scene.get("objects", [])
        context = self.scene.get("image_context", {})
        relations = self._select_knn_relations()

        lines = []

        # Scene Context
        lines.append("=== SCENE CONTEXT ===")
        lines.append(f"cluster_id: {context.get('cluster_id')}")
        lines.append(f"scene_label: {context.get('scene_label')}")
        lines.append(f"confidence: {self._round(context.get('confidence'))}")
        lines.append("")

        # Objects
        lines.append("=== OBJECTS ===")

        for obj in objects:
            sem = obj["semantic"]

            lines.append(f"[{obj['id']}]")
            lines.append(f"class: {obj['class']}")
            lines.append(f"bbox_xyxy: {obj['bbox']}")
            lines.append(f"center_xy: {obj['center']}")
            lines.append(f"rel_area: {self._round(sem['rel_area'])}")
            lines.append(f"aspect_ratio: {self._round(sem['aspect_ratio'])}")
            lines.append(f"brightness: {self._round(sem['brightness'])}")
            lines.append(f"saturation: {self._round(sem['saturation'])}")
            lines.append(f"edge_density: {self._round(sem['edge_density'])}")
            lines.append(f"texture_var: {self._round(sem['texture_var'])}")
            lines.append(f"lbp_histogram: {[self._round(x) for x in sem['lbp']]}")
            lines.append("color_palette:")

            for color in sem["color_palette"]:
                lines.append(
                    f"  - hsv: {[self._round(v) for v in color['hsv']]}, "
                    f"percent: {self._round(color['percent'])}"
                )

            lines.append("")

        # Relations
        lines.append("=== k-NEAREST RELATIONS ===")

        for rel in relations:
            lines.append(f"{rel['subject']} → {rel['object']}")
            lines.append(f"  predicate: {rel['predicate']}")
            lines.append(f"  angle_deg: {self._round(rel['angle'])}")
            lines.append(f"  distance_px: {self._round(rel['distance_px'])}")
            lines.append("")

        # Global Summary
        lines.append("=== GLOBAL SUMMARY ===")
        lines.append(f"total_objects: {len(objects)}")
        lines.append(f"k_per_object: {QTOONConfig.K_NEAREST}")
        lines.append(f"total_selected_relations: {len(relations)}")

        return "\n".join(lines)


# =========================================
# LLaMA Wrapper
# =========================================

class GGUFLLM:

    def __init__(self, model_path: str):
        self.llm = Llama(
            model_path=model_path,
            n_ctx=QTOONConfig.N_CTX,
            n_threads=QTOONConfig.N_THREADS,
            verbose=False
        )

    def generate(self, prompt: str, max_tokens: int = 600) -> str:
        output = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=0.3,
            top_p=0.9,
            stop=["</s>"]
        )
        return output["choices"][0]["text"].strip()


# =========================================
# Main Pipeline Function
# =========================================

def run_qtoon_pipeline(
    json_input: Union[str, Dict[str, Any]],
    gguf_model_path: str
) -> str:

    # Load JSON
    if isinstance(json_input, str):
        with open(json_input, "r") as f:
            scene_json = json.load(f)
    else:
        scene_json = json_input

    # Generate Q-TOON
    generator = QTOONGenerator(scene_json)
    qtoon_text = generator.generate()

    # ----- ORIGINAL PROMPT (UNCHANGED) -----
    prompt = f"""
You are an expert visual scene interpreter.

Below is a structured quantitative representation of a scene.

Your task is to transform this structured data into a fluent,
coherent, and richly detailed natural language description.

Guidelines:
- Write in smooth, natural prose.
- Do NOT list raw numeric values explicitly.
- Use the values internally to guide spatial and lighting reasoning.
- Describe spatial relationships naturally (e.g., "to the left of", "slightly above").
- Mention object sizes comparatively (small, large, prominent).
- Capture the overall atmosphere of the scene.
- Maintain logical consistency with the data.

Structured Scene Data:

{qtoon_text}

Now generate a detailed natural language description of the scene in one paragraph."""
    # --------------------------------------

    # Load model
    llm = GGUFLLM(gguf_model_path)

    # Generate description
    description = llm.generate(prompt)

    return description


# =========================================
# Example Usage
# =========================================

if __name__ == "__main__":

    model_path = r"D:\Work\RCI\Code\models\core\model\qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"
    json_path = r"D:\Work\RCI\Code\models\core\extracted.json"

    description = run_qtoon_pipeline(json_path, model_path)

    print("\nGenerated Description:\n")
    print(description)
    
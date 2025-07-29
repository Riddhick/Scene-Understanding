import os
import json
import cv2
from tqdm import tqdm

# Get base paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DOTA_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', 'DOTA'))

# Select only these 8 categories
SELECTED_CATEGORIES = [
    "plane", "ship", "storage-tank",
    "harbor", "bridge", "large-vehicle",
    "small-vehicle", "helicopter"
]
category2id = {name: i + 1 for i, name in enumerate(SELECTED_CATEGORIES)}

def parse_dota_label(label_file):
    objects = []
    with open(label_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            try:
                x1, y1, x2, y2, x3, y3, x4, y4 = map(float, parts[:8])
                category = parts[8]
            except ValueError:
                continue

            if category not in category2id:
                continue

            segmentation = [x1, y1, x2, y2, x3, y3, x4, y4]
            xs = [x1, x2, x3, x4]
            ys = [y1, y2, y3, y4]
            xmin, ymin = min(xs), min(ys)
            xmax, ymax = max(xs), max(ys)
            width, height = xmax - xmin, ymax - ymin

            bbox = [xmin, ymin, width, height]
            area = width * height

            obj = {
                "bbox": bbox,
                "segmentation": [segmentation],
                "category_id": category2id[category],
                "area": area,
                "iscrowd": 0
            }
            objects.append(obj)
    return objects

def generate_filtered_coco(image_dir, label_dir, output_json_path):
    print(f"\n[INFO] Processing: {image_dir}")
    coco_dict = {
        "images": [],
        "annotations": [],
        "categories": [{"id": cid, "name": name} for name, cid in category2id.items()]
    }

    annotation_id = 1
    image_id = 1

    for img_file in tqdm(sorted(os.listdir(image_dir))):
        if not img_file.lower().endswith(('.png', '.jpg')):
            continue

        img_path = os.path.join(image_dir, img_file)
        label_file = os.path.join(label_dir, os.path.splitext(img_file)[0] + '.txt')

        if not os.path.exists(label_file):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        height, width = img.shape[:2]

        coco_dict["images"].append({
            "id": image_id,
            "file_name": img_file,
            "width": width,
            "height": height
        })

        objects = parse_dota_label(label_file)
        for obj in objects:
            obj["id"] = annotation_id
            obj["image_id"] = image_id
            coco_dict["annotations"].append(obj)
            annotation_id += 1

        image_id += 1

    with open(output_json_path, 'w') as f:
        json.dump(coco_dict, f, indent=2)
    print(f"[SAVED] {output_json_path}")

if __name__ == "__main__":
    # Training
    train_image_dir = os.path.join(DOTA_ROOT, 'training')
    train_label_dir = os.path.join(DOTA_ROOT, 'label_train')
    train_output_json = os.path.join(SCRIPT_DIR, 'dota_train_filtered_coco.json')

    # Testing
    test_image_dir = os.path.join(DOTA_ROOT, 'testing')
    test_label_dir = os.path.join(DOTA_ROOT, 'label_test')
    test_output_json = os.path.join(SCRIPT_DIR, 'dota_test_filtered_coco.json')

    generate_filtered_coco(train_image_dir, train_label_dir, train_output_json)
    generate_filtered_coco(test_image_dir, test_label_dir, test_output_json)

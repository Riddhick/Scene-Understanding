import json
import os
import shutil
from pathlib import Path

# Your target classes
SELECTED_CLASSES = {
    "plane", "ship", "storage-tank",
    "harbor", "bridge", "large-vehicle",
    "small-vehicle", "helicopter"
}

def filter_coco_dataset(input_json, image_dir, output_json, output_img_dir):
    # Load the COCO dataset
    with open(input_json, 'r') as f:
        data = json.load(f)

    # Build category name-to-ID and ID-to-name mapping
    cat_id_to_name = {cat['id']: cat['name'] for cat in data['categories']}
    cat_name_to_id = {v: k for k, v in cat_id_to_name.items()}

    # Get the category IDs for the selected classes
    selected_cat_ids = {cat_name_to_id[name] for name in SELECTED_CLASSES if name in cat_name_to_id}

    # Filter annotations for selected categories
    filtered_annotations = [ann for ann in data['annotations'] if ann['category_id'] in selected_cat_ids]

    # Keep only images that have at least one valid annotation
    valid_image_ids = {ann['image_id'] for ann in filtered_annotations}
    filtered_images = [img for img in data['images'] if img['id'] in valid_image_ids]

    # Filter categories to selected ones
    filtered_categories = [cat for cat in data['categories'] if cat['id'] in selected_cat_ids]

    # Prepare output folder
    os.makedirs(output_img_dir, exist_ok=True)

    # Copy only filtered images
    copied = 0
    for img in filtered_images:
        filename = img['file_name']
        src = os.path.join(image_dir, filename)
        dst = os.path.join(output_img_dir, filename)
        if os.path.exists(src):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1
        else:
            print(f"⚠️ Missing image file: {filename}")

    # Save filtered COCO JSON
    filtered_data = {
        "images": filtered_images,
        "annotations": filtered_annotations,
        "categories": filtered_categories
    }

    with open(output_json, 'w') as f:
        json.dump(filtered_data, f)

    print(f"✅ {len(filtered_images)} images and {len(filtered_annotations)} annotations saved to:")
    print(f"   JSON: {output_json}")
    print(f"   Images: {output_img_dir} ({copied} files copied)")


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DOTA_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', 'DOTA'))
CODE_ROOT=os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', 'Code'))
TRAIN_IMAGE_DIR = os.path.join(DOTA_ROOT, 'training')  # or 'testing'
TRAIN_ANNOTATION_FILE = os.path.join(CODE_ROOT, 'annotations\dota_train_coco.json')
TRAIN_ANNOTATION_FILE_FILTERED = os.path.join(CODE_ROOT, 'annotations\dota_train_filtered_coco.json')
TEST_IMAGE_DIR = os.path.join(DOTA_ROOT, 'testing')  # or 'testing'
TEST_ANNOTATION_FILE = os.path.join(CODE_ROOT, 'annotations\dota_test_coco.json')
TRAIN_FILTERED_DIR = os.path.join(DOTA_ROOT, 'train_filtered')  
TEST_FILTERED_DIR = os.path.join(DOTA_ROOT, 'test_filtered')  
TEST_ANNOTATION_FILE_FILTERED = os.path.join(CODE_ROOT, 'annotations\dota_test_filtered_coco.json')



# ===== USAGE SECTION =====

# Path to COCO-style DOTA JSONs
train_json = TRAIN_ANNOTATION_FILE
test_json = TEST_ANNOTATION_FILE

# Path to raw image folders
train_img_dir = TRAIN_IMAGE_DIR
test_img_dir = TEST_IMAGE_DIR

# Output locations
train_filtered_json = TRAIN_ANNOTATION_FILE_FILTERED
test_filtered_json = TEST_ANNOTATION_FILE_FILTERED

train_filtered_img_dir = TRAIN_FILTERED_DIR
test_filtered_img_dir = TEST_FILTERED_DIR

# Run
#filter_coco_dataset(train_json, train_img_dir, train_filtered_json, train_filtered_img_dir)
filter_coco_dataset(test_json, test_img_dir, test_filtered_json, test_filtered_img_dir)

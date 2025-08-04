import os
import json
from PIL import Image

def convert_dota_to_coco(dota_image_dir, dota_label_dir, output_json_path):
    """
    Converts DOTA dataset annotations to COCO format.

    Args:
        dota_image_dir (str): Path to the DOTA images directory.
        dota_label_dir (str): Path to the DOTA labelTxt directory.
        output_json_path (str): Path to save the output COCO JSON file.
    """
    # Create the main COCO format dictionary
    coco_data = {
        "info": {
            "description": "COCO-style dataset generated from DOTA annotations.",
            "version": "1.0",
            "year": 2024,
            "contributor": "Automated Script",
            "date_created": "2024/01/01"
        },
        "licenses": [], # DOTA does not provide licenses in the annotations
        "images": [],
        "annotations": [],
        "categories": []
    }

    # Dictionaries to manage unique IDs and categories
    category_map = {}
    image_id = 0
    annotation_id = 0
    category_id = 1

    # Get a list of all DOTA annotation files
    label_files = [f for f in os.listdir(dota_label_dir) if f.endswith('.txt')]

    print(f"Found {len(label_files)} annotation files in {dota_label_dir}. Starting conversion...")

    for label_file in label_files:
        try:
            image_filename = label_file.replace('.txt', '.png')  # Assuming DOTA images are PNG
            image_path = os.path.join(dota_image_dir, image_filename)

            # Check if image file exists
            if not os.path.exists(image_path):
                print(f"Warning: Image file '{image_filename}' not found. Skipping '{label_file}'.")
                continue

            # Open image to get dimensions
            with Image.open(image_path) as img:
                width, height = img.size

            # Create an image entry for the COCO format
            image_entry = {
                "id": image_id,
                "width": width,
                "height": height,
                "file_name": image_filename,
                "license": 1,
            }
            coco_data["images"].append(image_entry)

            # Read the DOTA annotation file
            with open(os.path.join(dota_label_dir, label_file), 'r') as f:
                lines = f.readlines()

            for line in lines:
                # Skip metadata lines (e.g., gsd, imagesource)
                if not line or line.startswith('gsd') or line.startswith('imagesource'):
                    continue

                parts = line.strip().split()
                if len(parts) < 10:
                    print(f"Warning: Skipping malformed line in {label_file}: {line.strip()}")
                    continue

                # DOTA format: x1 y1 x2 y2 x3 y3 x4 y4 category difficult
                points = [float(p) for p in parts[:8]]
                category_name = parts[8]
                difficult_flag = int(parts[9])
                
                # Dynamically create category IDs
                if category_name not in category_map:
                    category_map[category_name] = category_id
                    coco_data["categories"].append({
                        "id": category_id,
                        "name": category_name,
                        "supercategory": "None" # DOTA doesn't have supercategories
                    })
                    category_id += 1

                current_category_id = category_map[category_name]

                # Convert OBB points to an axis-aligned bounding box (HBB)
                # COCO bbox format: [x, y, width, height]
                x_coords = points[0::2]
                y_coords = points[1::2]
                x_min = min(x_coords)
                y_min = min(y_coords)
                x_max = max(x_coords)
                y_max = max(y_coords)
                bbox_width = x_max - x_min
                bbox_height = y_max - y_min
                bbox = [x_min, y_min, bbox_width, bbox_height]

                # Create segmentation polygon from OBB points
                # COCO segmentation format is a flat list of coordinates
                segmentation = [points]
                area = bbox_width * bbox_height

                # Create the annotation entry for the COCO format
                annotation_entry = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": current_category_id,
                    "segmentation": segmentation,
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": difficult_flag # Use the DOTA 'difficult' flag as 'iscrowd'
                }
                coco_data["annotations"].append(annotation_entry)

                annotation_id += 1
            
            image_id += 1

        except Exception as e:
            print(f"Error processing file {label_file}: {e}")

    # Write the final COCO JSON to a file
    with open(output_json_path, 'w') as f:
        json.dump(coco_data, f, indent=4)

    print("\nConversion complete!")
    print(f"Saved COCO-formatted JSON file to: {output_json_path}")
    print(f"Total images converted: {len(coco_data['images'])}")
    print(f"Total annotations converted: {len(coco_data['annotations'])}")
    print(f"Total categories found: {len(coco_data['categories'])}")




if __name__ == "__main__":

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    CODE_ROOT=os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', 'Code'))
    DOTA_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', 'DOTA'))
    IMAGE_DIR = os.path.join(DOTA_ROOT, 'testing')  # or 'testing'
    ANNOTATION_FILE = os.path.join(CODE_ROOT, 'annotations/test_ann.json') 
    # Define your dataset paths
    # IMPORTANT: Replace these with your actual DOTA dataset paths
    dota_image_dir = IMAGE_DIR
    dota_label_dir = 'D:/Work/RCI/DOTA/label_test'
    output_json_path = ANNOTATION_FILE

    # Check if the example paths exist, if not, provide a user-friendly message
    if not os.path.exists(dota_image_dir) or not os.path.exists(dota_label_dir):
        print("DOTA dataset directories not found.")
        print("Please ensure the paths are correct and that you have downloaded the DOTA devkit and placed it in the script's directory.")
        print(f"Expected image directory: {os.path.abspath(dota_image_dir)}")
        print(f"Expected label directory: {os.path.abspath(dota_label_dir)}")
    else:
        convert_dota_to_coco(dota_image_dir, dota_label_dir, output_json_path)
import os
import glob
from PIL import Image

def convert_dota_to_yolo(dota_image_dir, dota_label_dir, output_dir, yaml_path):
    """
    Convert DOTA annotations to YOLOv8 format (HBB).
    Args:
        dota_image_dir: Path to DOTA images.
        dota_label_dir: Path to DOTA labelTxt files.
        output_dir: Where to save YOLO labels.
        yaml_path: Path to save YOLO dataset YAML file.
    """
    os.makedirs(output_dir, exist_ok=True)
    labels_out_dir = os.path.join(output_dir, "labels")
    images_out_dir = os.path.join(output_dir, "images")
    os.makedirs(labels_out_dir, exist_ok=True)
    os.makedirs(images_out_dir, exist_ok=True)

    category_map = {}
    category_id = 0

    label_files = [f for f in os.listdir(dota_label_dir) if f.endswith('.txt')]
    print(f"Found {len(label_files)} label files.")

    for label_file in label_files:
        image_basename = label_file.replace('.txt', '')

        # Find the actual image file with any common extension
        image_path = None
        for ext in ['.png', '.jpg', '.tif', '.bmp']:
            test_path = os.path.join(dota_image_dir, image_basename + ext)
            if os.path.exists(test_path):
                image_path = test_path
                break
        
        if image_path is None:
            print(f"⚠ Image for {label_file} not found, skipping.")
            continue

        # Copy image to output/images
        out_image_path = os.path.join(images_out_dir, os.path.basename(image_path))
        if not os.path.exists(out_image_path):
            from shutil import copyfile
            copyfile(image_path, out_image_path)

        # Get image size
        with Image.open(image_path) as img:
            img_w, img_h = img.size

        yolo_lines = []

        with open(os.path.join(dota_label_dir, label_file), 'r') as f:
            for line in f:
                if not line.strip() or line.startswith('gsd') or line.startswith('imagesource'):
                    continue

                parts = line.strip().split()
                if len(parts) < 10:
                    continue

                # Extract points and category
                points = [float(p) for p in parts[:8]]
                category_name = parts[8]
                # difficult_flag = int(parts[9])  # not used in YOLO

                if category_name not in category_map:
                    category_map[category_name] = category_id
                    category_id += 1

                cls_id = category_map[category_name]

                # Convert OBB → HBB
                x_coords = points[0::2]
                y_coords = points[1::2]
                x_min = min(x_coords)
                y_min = min(y_coords)
                x_max = max(x_coords)
                y_max = max(y_coords)

                # Normalize to YOLO format
                x_center = ((x_min + x_max) / 2) / img_w
                y_center = ((y_min + y_max) / 2) / img_h
                w_norm = (x_max - x_min) / img_w
                h_norm = (y_max - y_min) / img_h

                # Avoid zero-size boxes
                if w_norm <= 0 or h_norm <= 0:
                    continue

                yolo_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

        # Save YOLO label file
        out_label_path = os.path.join(labels_out_dir, image_basename + ".txt")
        with open(out_label_path, 'w') as out_f:
            out_f.write("\n".join(yolo_lines))

    # Save YAML
    with open(yaml_path, 'w') as f:
        f.write("path: " + os.path.abspath(output_dir) + "\n")
        f.write("train: images\n")
        f.write("val: images\n")  # You may split train/val manually
        f.write("names:\n")
        for name, idx in sorted(category_map.items(), key=lambda x: x[1]):
            f.write(f"  {idx}: {name}\n")

    print(f"\n✅ Conversion complete.")
    print(f"Classes: {len(category_map)}")
    print(f"YAML saved to: {yaml_path}")
    print(f"Labels saved in: {labels_out_dir}")
    print(f"Images saved in: {images_out_dir}")


if __name__ == "__main__":
    DOTA_IMAGE_DIR = r"D:\Work\DOTA\images"
    DOTA_LABEL_DIR = r"D:\Work\DOTA\labelTxt"
    OUTPUT_DIR = r"D:\Work\DOTA_YOLOv8"
    YAML_PATH = os.path.join(OUTPUT_DIR, "dota.yaml")

    convert_dota_to_yolo(DOTA_IMAGE_DIR, DOTA_LABEL_DIR, OUTPUT_DIR, YAML_PATH)
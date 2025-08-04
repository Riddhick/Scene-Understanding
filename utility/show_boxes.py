import os
import json
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Path to annotation and image directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT=os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', 'Code'))
DOTA_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', 'DOTA'))
IMAGE_DIR = os.path.join(DOTA_ROOT, 'training')  # or 'testing'
ANNOTATION_FILE = os.path.join(CODE_ROOT, 'annotations\dota_train_filtered_coco.json') 
#ANNOTATION_FILE = os.path.join(SCRIPT_DIR, 'dota_train_filtered_coco.json')  # or 'dota_test_coco.json'

# Load COCO-style annotation
with open(ANNOTATION_FILE, 'r') as f:
    coco = json.load(f)

# Build index for quick lookup
images = {img['id']: img for img in coco['images']}
annotations_per_image = {}
for ann in coco['annotations']:
    annotations_per_image.setdefault(ann['image_id'], []).append(ann)

categories = {cat['id']: cat['name'] for cat in coco['categories']}

# Choose an image ID to visualize
image_id = list(images.keys())[300]  # change index for other images
img_info = images[image_id]
img_path = os.path.join(IMAGE_DIR, img_info['file_name'])

# Load image
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Plot
fig, ax = plt.subplots(1, figsize=(12, 12))
ax.imshow(img)
ax.set_title(f"Image ID: {image_id} - {img_info['file_name']}")

# Draw boxes
for ann in annotations_per_image.get(image_id, []):
    bbox = ann['bbox']
    category_name = categories[ann['category_id']]
    rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
                             linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    ax.text(bbox[0], bbox[1] - 2, category_name,
            fontsize=10, color='white', bbox=dict(facecolor='red', alpha=0.5))

plt.axis('off')
plt.show()

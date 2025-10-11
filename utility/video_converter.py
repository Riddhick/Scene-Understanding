import cv2
import os

# === Configuration ===
image_folder = 'D:\Work\VisDrone2019\VisDrone2019-MOT-train\sequences\\uav0000218_00001_v'   # Folder containing images
output_video = 'output_video.mp4'              # Output video file name
fps = 30                                       # Frames per second

# === Step 1: Get images ===
images = [img for img in os.listdir(image_folder) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
images.sort()  # Sorts alphabetically (keeps your sequence order)

# === Step 2: Read first image for frame size ===
first_frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, _ = first_frame.shape
size = (width, height)

# === Step 3: Create video writer ===
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, size)

# === Step 4: Write frames ===
for img_name in images:
    frame = cv2.imread(os.path.join(image_folder, img_name))
    out.write(frame)

out.release()
print(f"✅ Video saved as {output_video}")

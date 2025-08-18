import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import cv2
import numpy as np
import time

class VisDroneDataset(Dataset):
    def __init__(self, root_dir, annotation_dir, transforms=None):
        self.root_dir = root_dir
        self.annotation_dir = annotation_dir
        self.transforms = transforms
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]
        
        # VisDrone classes
        self.classes = [
            '_', 'pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 
            'tricycle', 'awning-tricycle', 'bus', 'motor'
        ]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        ann_path = os.path.join(self.annotation_dir, self.image_files[idx].replace('.jpg', '.txt'))
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes, labels = [], []
        with open(ann_path, 'r') as f:
            for line in f.readlines():
                vals = line.strip().split(',')
                if len(vals) < 5:
                    continue
                x, y, w, h, cls_id = map(int, vals[:5])
                if w <= 0 or h <= 0:  # skip invalid boxes
                    continue
                if cls_id <= 0 or cls_id >= len(self.classes):  # skip invalid class ids
                    continue
                boxes.append([x, y, x+w, y+h])
                labels.append(cls_id)

        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)

        if self.transforms:
            transformed = self.transforms(image=image, bboxes=boxes, labels=labels)
            image = transformed["image"]
            boxes = transformed["bboxes"]
            labels = transformed["labels"]

        # Convert back to tensors
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64)
        }

        return image.float(), target

# Albumentations transforms
train_transforms = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    ToTensorV2()
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

val_transforms = A.Compose([
    ToTensorV2()
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

# Data loaders
train_dataset = VisDroneDataset("/kaggle/input/visdrone-dataset/VisDrone_Dataset/VisDrone2019-DET-train/images", "/kaggle/input/visdrone-dataset/VisDrone_Dataset/VisDrone2019-DET-train/labels", transforms=train_transforms)
val_dataset = VisDroneDataset("/kaggle/input/visdrone-dataset/VisDrone_Dataset/VisDrone2019-DET-val/images", "/kaggle/input/visdrone-dataset/VisDrone_Dataset/VisDrone2019-DET-val/labels", transforms=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# Model: Faster R-CNN
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, len(train_dataset.classes))

# Optimizer and scheduler
optimizer = optim.SGD(model.parameters(), lr=0.0025, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# Training loop with checkpoints + AMP + gradient clipping
scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

def train_model(model, train_loader, val_loader, device, num_epochs=10, checkpoint_dir="checkpoints"):
    model.to(device)
    best_val_loss = float("inf")
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_losses = []
    val_losses = []

    steps_per_epoch = len(train_loader)
    bs = getattr(train_loader, 'batch_size', None) or 1
    n_train = len(train_loader.dataset)
    print(f"Dataset size: {n_train} images | Batch size: {bs} | Steps/epoch: {steps_per_epoch}")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        batch_times = []
        start_epoch = time.time()

        for step, (images, targets) in enumerate(train_loader):
            t0 = time.time()
            images = [img.to(device, non_blocking=True) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(losses).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += losses.item()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            batch_times.append(time.time() - t0)
            
            if step % 200 ==0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{len(train_loader)}], Loss: {losses.item():.4f}")

        lr_scheduler.step()
        avg_loss = epoch_loss / max(1, steps_per_epoch)
        train_losses.append(avg_loss)

        # Validation (placeholder: using proxy of train loss)
        model.train()  # temporarily switch to train mode to get losses
        val_loss = 0.0
        with torch.enable_grad():  # don't update weights
            for images, targets in val_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)
                val_loss += sum(loss for loss in loss_dict.values()).item()
        avg_val_loss = val_loss / max(1, len(val_loader))
        val_losses.append(avg_val_loss)

        # Timing stats
        epoch_time = time.time() - start_epoch
        mean_bt = float(np.mean(batch_times)) if batch_times else 0.0
        imgs_per_sec = (bs * steps_per_epoch) / epoch_time if epoch_time > 0 else 0.0
        print(
            f"Epoch [{epoch+1}/{num_epochs}] | Loss: {avg_loss:.4f} | Val: {avg_val_loss:.4f} | "
            f"Epoch time: {epoch_time:.1f}s | ~{imgs_per_sec:.1f} imgs/s | mean batch {mean_bt*1000:.1f} ms"
        )

        # Save best checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved Best Model at {checkpoint_path}")

    # Save final checkpoint
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "final_model.pth"))
    print("Training complete. Final model saved.")
    return train_losses, val_losses

# Device setup
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Run training
history=train_model(model, train_loader, val_loader, device, num_epochs=2)

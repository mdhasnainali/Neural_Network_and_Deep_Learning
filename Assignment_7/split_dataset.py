import os
import random
import shutil

random.seed(42)

# Paths
wider_train_img_dir = 'archive/WIDER_train/WIDER_train/images'
yolo_img_train_dir = 'yolo_format_data/images/train'
yolo_img_val_dir = 'yolo_format_data/images/val'
yolo_lbl_train_dir = 'yolo_format_data/labels/train'
yolo_lbl_val_dir = 'yolo_format_data/labels/val'
yolo_lbl_src_dir = 'yolo_format_data/labels/all'  

for d in [yolo_img_train_dir, yolo_img_val_dir, yolo_lbl_train_dir, yolo_lbl_val_dir]:
    os.makedirs(d, exist_ok=True)

img_paths = []
for root, _, files in os.walk(wider_train_img_dir):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_paths.append(os.path.join(root, file))

print(f'Total images found: {len(img_paths)}')

# Shuffle and split
random.shuffle(img_paths)
split_ratio = 0.9
split_idx = int(len(img_paths) * split_ratio)
train_imgs = img_paths[:split_idx]
val_imgs = img_paths[split_idx:]

def copy_with_structure(src, dst_root):
    """Copy file from src to dst_root, preserving subfolder structure."""
    rel_path = os.path.relpath(src, wider_train_img_dir)
    dst = os.path.join(dst_root, rel_path)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    return dst

def copy_label_for_img(img_path, dst_root):
    """Copy corresponding YOLO label file for an image."""
    rel_path = os.path.relpath(img_path, wider_train_img_dir)
    label_path = os.path.splitext(rel_path)[0] + '.txt'
    src_label = os.path.join(yolo_lbl_src_dir, label_path)
    dst_label = os.path.join(dst_root, label_path)
    if os.path.exists(src_label):
        os.makedirs(os.path.dirname(dst_label), exist_ok=True)
        shutil.copy2(src_label, dst_label)

# Copy images and labels
for img in train_imgs:
    dst_img = copy_with_structure(img, yolo_img_train_dir)
    copy_label_for_img(img, yolo_lbl_train_dir)

for img in val_imgs:
    dst_img = copy_with_structure(img, yolo_img_val_dir)
    copy_label_for_img(img, yolo_lbl_val_dir)

print(f'Train images: {len(train_imgs)}')
print(f'Val images: {len(val_imgs)}')

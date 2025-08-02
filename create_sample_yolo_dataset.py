import os
import random
import shutil

# === CONFIGURATION ===
SRC_IMG = 'yolo_dataset/train/images'
SRC_LBL = 'yolo_dataset/train/labels'
DST_IMG = 'yolo_dataset/sample/train/images'
DST_LBL = 'yolo_dataset/sample/train/labels'
NUM_SAMPLES = 20  # Change this to however many images you want

# === CREATE DESTINATION FOLDERS ===
os.makedirs(DST_IMG, exist_ok=True)
os.makedirs(DST_LBL, exist_ok=True)

# === GET IMAGE FILES ===
img_files = [f for f in os.listdir(SRC_IMG) if f.endswith(('.jpg', '.jpeg', '.png'))]
sample_files = random.sample(img_files, min(NUM_SAMPLES, len(img_files)))

# === COPY IMAGES AND LABELS ===
for f in sample_files:
    shutil.copy(os.path.join(SRC_IMG, f), DST_IMG)
    label_file = f.rsplit('.', 1)[0] + '.txt'
    src_label_path = os.path.join(SRC_LBL, label_file)
    if os.path.exists(src_label_path):
        shutil.copy(src_label_path, DST_LBL)
    else:
        print(f"Warning: Label file not found for {f}")

print(f"Sample dataset created with {len(sample_files)} images in {DST_IMG}") 
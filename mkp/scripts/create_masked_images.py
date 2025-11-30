"""
create_masked_images.py

This script generates masked images for the MKP (Masked Keypoint Prediction)
self-supervised learning pipeline, using a dataset that follows the Ultralytics
YOLOv8 keypoint folder structure.

Correct dataset path:
    /Users/taircarmon/Desktop/ssl-prawn/mkp/scripts/mkp/prawn_2025_circ_small_v1/

Correct dataset structure:
    prawn_2025_circ_small_v1/
        images/
            train/   -> training images
            val/     -> validation images
        labels/
            train/   -> YOLO keypoint label files (.txt)
            val/     -> YOLO keypoint label files (.txt)
        data.yaml    -> dataset configuration file

For every image:
    images/train/img_001.jpg
there is a matching label file:
    labels/train/img_001.txt

YOLO keypoint label format:
    class  x_center  y_center  w  h  kp1_x kp1_y kp1_v  ... kpK_x kpK_y kpK_v

What this script does:
------------------------------------------------------------
1. Iterates through:
       images/train/
       images/val/
2. Loads each image.
3. Loads the corresponding YOLO keypoints from labels/train/ or labels/val/.
4. Selects one or more keypoints to mask.
5. Applies a mask centered on each keypoint location:
       - black square mask   (default)
       - blur mask           (optional)
       - noise patch         (optional)
6. Saves masked images into:
       images/train_masked/
       images/val_masked/

Purpose:
------------------------------------------------------------
Masked Keypoint Prediction (MKP) is a self-supervised pretext task
where keypoint regions are hidden, and a model is later trained to
reconstruct the missing keypoints. This forces the encoder to learn
prawn anatomy and body geometry.

This script prepares masked images ONLY (no model training).

Usage:
------------------------------------------------------------
    python create_masked_images.py

After running, new folders will be created:
    images/train_masked/
    images/val_masked/

These folders will contain masked versions of the images,
ready for MKP SSL training.
"""
# python
import random
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

MASK_SIZE = 40
MASK_TYPE = "black"
DATASET_PATH = Path("C:/Users/carmonta/Desktop/ssl-prawn/mkp/scripts/mkp_data/prawn_2025_circ_small_v1/")

def apply_mask_one(image, keypoint, mask_size, mask_type):
    """Apply a single square mask centered on one keypoint (keypoint = (x_rel, y_rel, v))."""
    h, w = image.shape[:2]
    x_rel, y_rel, _ = keypoint
    cx = int(x_rel * w)
    cy = int(y_rel * h)
    half = max(1, mask_size // 2)
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(w, cx + half)
    y2 = min(h, cy + half)

    patched = image.copy()
    if x2 <= x1 or y2 <= y1:
        return patched

    if mask_type == "black":
        patched[y1:y2, x1:x2] = 0
    elif mask_type == "blur":
        roi = patched[y1:y2, x1:x2]
        if roi.size > 0:
            patched[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (15, 15), 0)
    elif mask_type == "noise":
        noise = np.random.randint(0, 256, (y2 - y1, x2 - x1, 3), dtype=np.uint8)
        patched[y1:y2, x1:x2] = noise
    return patched

def process_dataset(split):
    images_dir = DATASET_PATH / "images" / split
    labels_dir = DATASET_PATH / "labels" / split
    output_dir = DATASET_PATH / "images" / f"{split}_masked"
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = list(images_dir.glob("*"))
    for img_path in tqdm(image_files, desc=f"Processing {split} set"):
        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            continue

        # load keypoints (same format as before)
        keypoints = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                kp_data = parts[5:]
                for i in range(0, len(kp_data), 3):
                    x = float(kp_data[i])
                    y = float(kp_data[i+1])
                    v = int(float(kp_data[i+2]))
                    keypoints.append((x, y, v))

        # filter visible keypoints and pick exactly one
        visible = [kp for kp in keypoints if kp[2] > 0]
        if not visible:
            continue
        chosen = random.choice(visible)

        masked_image = apply_mask_one(image, chosen, MASK_SIZE, MASK_TYPE)
        output_path = output_dir / img_path.name
        cv2.imwrite(str(output_path), masked_image)

if __name__ == "__main__":
    process_dataset("train")
    process_dataset("val")


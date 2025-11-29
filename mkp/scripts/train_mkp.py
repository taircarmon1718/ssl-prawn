"""
train_mkp.py

This script trains a Masked Keypoint Prediction (MKP) self-supervised model.
The model receives *masked images* as input and tries to predict the true
(unmasked) keypoint locations from the original YOLO label files.

Input: images/train_masked/, images/val_masked/
Labels: labels/train/, labels/val/   (unchanged)
Config: mkp.yaml
Output: mkp_pretrained.pt  (used later for downstream fine-tuning)
"""

from ultralytics import YOLO

def main():
    # Load a small YOLO pose model (you can change to yolov8s-pose if you want)
    model = YOLO("yolov8n-pose.pt")

    # Train on MKP masked dataset
    model.train(
        data="mkp.yaml",
        epochs=30,
        imgsz=640,
        batch=16,
        lr0=0.0008,
        name="mkp_pretraining",
        pretrained=True
    )

    # Save final pretrained weights
    model.export(format="torchscript")  # optional
    print("MKP self-supervised training finished.")
    print("Model saved in: runs/pose/mkp_pretraining/weights/")

if __name__ == "__main__":
    main()

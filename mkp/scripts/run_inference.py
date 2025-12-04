import torch
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as T

# ייבוא המודל (וודא ש-train_keypoint_detector.py באותה תיקייה)
try:
    from train_keypoint_detector import PrawnPredictor
except ImportError:
    print("Error: Could not import PrawnPredictor. Run this from the folder containing 'train_keypoint_detector.py'")
    exit()

# --- הגדרות ---
IMAGE_SIZE = 224
# שנה לנתיב שבו שמרת את המודל הסופי (אחרי 50 Epochs)
MODEL_PATH = Path("runs/prawn_detector_epoch_50.pth")

# נתיב לתיקיית התמונות והתוויות
DATA_ROOT = Path(r"C:\Users\carmonta\Desktop\ssl-prawn\mkp\scripts\mkp_data\prawn_2025_circ_small_v1")
IMAGES_DIR = DATA_ROOT / "images" / "train"
LABELS_DIR = DATA_ROOT / "labels" / "train"

OUTPUT_DIR = Path("runs/inference_comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# המיפוי של ה-Keypoints (חייב להתאים למה שהיה באימון)
# 'rostrum': 0, 'tail': 4, 'carapace': 3, 'eyes': 2
EYES_INDEX_IN_YOLO = 2


def get_ground_truth_coords(image_name):
    """קורא את קובץ ה-Label ומחזיר את הקואורדינטות האמיתיות של העין"""
    label_path = LABELS_DIR / f"{image_name}.txt"
    if not label_path.exists():
        return None

    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                data = [float(x) for x in line.split()]
                # YOLO Format: class, x, y, w, h, k1x, k1y, k1v, ...
                # חישוב האינדקס של העיניים
                start_idx = 5 + (EYES_INDEX_IN_YOLO * 3)

                if len(data) > start_idx + 1:
                    vx = data[start_idx]
                    vy = data[start_idx + 1]
                    viz = data[start_idx + 2]
                    if viz > 0:  # אם הנקודה מסומנת כגלויה
                        return (vx, vy)
    except:
        return None
    return None


def predict_and_compare(model, image_path, device):
    # 1. טעינת תמונה
    orig_img = cv2.imread(str(image_path))
    if orig_img is None: return

    h, w, _ = orig_img.shape

    # המרה ל-RGB ושינוי גודל ל-224x224 בשביל המודל
    img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    transform = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor()
    ])

    img_tensor = transform(pil_img).unsqueeze(0).to(device)

    # 2. הרצת המודל (Prediction)
    model.eval()
    with torch.no_grad():
        coords = model(img_tensor)
        pred_nx, pred_ny = coords[0][0].item(), coords[0][1].item()

    # 3. הכנת תמונות להשוואה
    # מעתיקים את התמונה המקורית פעמיים
    img_prediction = orig_img.copy()
    img_ground_truth = orig_img.copy()

    # --- ציור הניבוי (Prediction) ---
    pred_px = int(pred_nx * w)
    pred_py = int(pred_ny * h)

    # עיגול אדום קטן
    cv2.circle(img_prediction, (pred_px, pred_py), radius=4, color=(0, 0, 255), thickness=-1)
    cv2.putText(img_prediction, "Model Pred", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # --- ציור האמת (Ground Truth) ---
    gt_coords = get_ground_truth_coords(image_path.stem)

    if gt_coords:
        gt_nx, gt_ny = gt_coords
        gt_px = int(gt_nx * w)
        gt_py = int(gt_ny * h)

        # עיגול ירוק קטן
        cv2.circle(img_ground_truth, (gt_px, gt_py), radius=4, color=(0, 255, 0), thickness=-1)

        # חישוב מרחק (שגיאה בפיקסלים)
        dist = np.sqrt((pred_px - gt_px) ** 2 + (pred_py - gt_py) ** 2)
        error_text = f"GT (Err: {dist:.1f}px)"
        cv2.putText(img_ground_truth, error_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(img_ground_truth, "No GT Label", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    # 4. חיבור התמונות זו לצד זו
    # פס הפרדה לבן
    separator = np.ones((h, 10, 3), dtype=np.uint8) * 255
    combined_img = np.hstack([img_prediction, separator, img_ground_truth])

    # 5. שמירה
    save_path = OUTPUT_DIR / f"compare_{image_path.name}"
    cv2.imwrite(str(save_path), combined_img)
    print(f"Saved: {save_path}")


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # טעינת המודל
    model = PrawnPredictor(pretrained_weights_path=None).to(device)

    if MODEL_PATH.exists():
        print(f"Loading trained detector from: {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    else:
        print(f"Error: Model file not found at {MODEL_PATH}")
        exit()

    # בחירת 20 תמונות אקראיות
    image_files = list(IMAGES_DIR.glob("*.jpg")) + list(IMAGES_DIR.glob("*.png"))

    import random

    selected_images = random.sample(image_files, min(20, len(image_files)))

    print("Running comparison...")
    for img_file in selected_images:
        predict_and_compare(model, img_file, device)

    print(f"\n✅ Done! Open the folder: {OUTPUT_DIR.absolute()}")
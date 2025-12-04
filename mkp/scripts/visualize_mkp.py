import torch
import numpy as np
import cv2
from pathlib import Path
from einops import rearrange
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# --- ייבוא המחלקות מהקבצים הקיימים שלך ---
# ודא שהקבצים mkp_model.py ו-PrawnMKPDataset.py נמצאים באותה תיקייה
try:
    from mkp_model import PrawnMKPModel
    from PrawnMKPDataset import PrawnMKPDataset, PATCH_SIZE, IMAGE_SIZE, NUM_PATCHES
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("וודא שאתה מריץ את הקוד הזה מאותה תיקייה שבה נמצאים mkp_model.py ו-PrawnMKPDataset.py")
    exit()

# --- הגדרות נתיבים (מותאם למה ששלחת) ---
# נתיב לקובץ המשקלים
WEIGHTS_PATH = Path(r"C:\Users\carmonta\Desktop\ssl-prawn\mkp\scripts\runs\prawn_encoder_best.pth")

# נתיב לתיקיית הדאטה (אותו נתיב שהשתמשת בו לאימון)
DATA_ROOT = Path(r"C:\Users\carmonta\Desktop\ssl-prawn\mkp\scripts\mkp_data\train_on_all")

# תיקיית פלט לתמונות
OUTPUT_DIR = Path("visualization_results")
OUTPUT_DIR.mkdir(exist_ok=True)


def denormalize(tensor):
    """ממיר טנסור (0-1) לתמונה (0-255) לתצוגה"""
    img = tensor.permute(1, 2, 0).cpu().numpy()
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def visualize_prediction(model, dataset, index, device):
    # 1. שליפת נתונים
    sample = dataset[index]
    img_tensor = sample['image'].unsqueeze(0).to(device)  # Add batch dim (1, C, H, W)
    unmasked = [sample['unmasked_indices'].to(device)]
    masked = [sample['masked_indices'].to(device)]
    img_path = Path(sample['image_path']).name

    print(f"Processing image: {img_path}...")

    # 2. הרצת המודל (Inference)
    model.eval()
    with torch.no_grad():
        # המודל מחזיר את כל הפאצ'ים (B, N, Pixels)
        preds = model(img_tensor, unmasked, masked)

    # 3. בניית התמונות להשוואה

    # א. תמונה מקורית
    original_img = denormalize(img_tensor[0])

    # ב. קלט (מה המודל ראה - רק עיניים)
    # יוצרים מסכה שחורה
    masked_view = img_tensor.clone()
    patches = rearrange(masked_view, 'b c (h p1) (w p2) -> b (h w) c p1 p2', p1=PATCH_SIZE, p2=PATCH_SIZE)
    # מאפסים את הפאצ'ים המוסתרים (צובעים בשחור)
    patches[0, sample['masked_indices']] = 0
    masked_view = rearrange(patches, 'b (h w) c p1 p2 -> b c (h p1) (w p2)', h=IMAGE_SIZE // PATCH_SIZE)
    input_img = denormalize(masked_view[0])

    # ג. שחזור (התוצאה)
    # לוקחים את התמונה המקורית ומחליפים את החלקים החסרים בניבוי של המודל
    recon_patches = rearrange(img_tensor.clone(), 'b c (h p1) (w p2) -> b (h w) c p1 p2', p1=PATCH_SIZE, p2=PATCH_SIZE)

    # המודל מחזיר (1, N, 768). נהפוך ל (1, N, 3, 16, 16)
    pred_reshaped = rearrange(preds, 'b n (c p1 p2) -> b n c p1 p2', p1=PATCH_SIZE, p2=PATCH_SIZE, c=3)

    # החלפה: שמים את הניבוי רק איפה שהיה מוסתר
    mask_idx = sample['masked_indices']
    if len(mask_idx) > 0:
        recon_patches[0, mask_idx] = pred_reshaped[0, mask_idx]

    recon_tensor = rearrange(recon_patches, 'b (h w) c p1 p2 -> b c (h p1) (w p2)', h=IMAGE_SIZE // PATCH_SIZE)
    output_img = denormalize(recon_tensor[0])

    # 4. חיבור ושמירה
    # פס לבן מפריד
    sep = np.ones((IMAGE_SIZE, 5, 3), dtype=np.uint8) * 255

    # חיבור: מקור | קלט | שחזור
    final_vis = np.hstack([original_img, sep, input_img, sep, output_img])

    save_path = OUTPUT_DIR / f"vis_{index}_{img_path}"
    cv2.imwrite(str(save_path), final_vis)
    print(f"Saved: {save_path}")


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. טעינת המודל
    print("Loading model architecture...")
    model = PrawnMKPModel().to(device)

    # 2. טעינת המשקלים
    if WEIGHTS_PATH.exists():
        print(f"Loading weights from: {WEIGHTS_PATH}")
        state_dict = torch.load(WEIGHTS_PATH, map_location=device)
        model.load_state_dict(state_dict)
        print("✅ Weights loaded successfully.")
    else:
        print(f"❌ Error: Weights file not found at {WEIGHTS_PATH}")
        exit()

    # 3. טעינת הנתונים
    print("Loading dataset...")
    # משתמשים ב-'eyes' כפי שאומן
    dataset = PrawnMKPDataset(DATA_ROOT, target_keypoints=['eyes'], split_type='train')

    # 4. יצירת ויזואליזציה ל-5 תמונות אקראיות
    num_samples = 5
    indices = np.random.choice(len(dataset), num_samples, replace=False)

    print(f"\nGenerating visualizations for {num_samples} random images...")
    for idx in indices:
        visualize_prediction(model, dataset, idx, device)

    print(f"\nDone! Check the folder: {OUTPUT_DIR.absolute()}")
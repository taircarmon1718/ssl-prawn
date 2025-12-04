import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from tqdm import tqdm
import numpy as np
import cv2
import torchvision.transforms as T
from PIL import Image

# --- IMPORTS ---
try:
    from mkp_model import PrawnMKPModel, VIT_DIM
except ImportError:
    from mkp.scripts.mkp_model import PrawnMKPModel, VIT_DIM

try:
    from PrawnMKPDataset import PATCH_SIZE, IMAGE_SIZE
except ImportError:
    from mkp.scripts.PrawnMKPDataset import PATCH_SIZE, IMAGE_SIZE

# --- Configuration ---
BATCH_SIZE = 16
LR = 1e-4
NUM_EPOCHS = 100
KEYPOINT_NAME = 'eyes'


# ==========================================
# 1. DATASET FOR SUPERVISED LEARNING
# ==========================================
class PrawnKeypointDataset(Dataset):
    """
    Dataset for final task: Returns Image + Coords (X, Y)
    """

    def __init__(self, data_root, target_kp, split_type='train'):
        self.image_dir = data_root / "images" / split_type
        self.label_dir = data_root / "labels" / split_type
        self.target_kp = target_kp
        self.image_paths = list(self.image_dir.glob("*.jpg")) + list(self.image_dir.glob("*.png"))

        self.transform = T.Compose([
            T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            T.ToTensor()
        ])

        # 🌟 UPDATE: Corrected Mapping
        # 0=Carapace, 1=Eyes, 2=Rostrum, 3=Tail
        self.kp_map ={
    'carapace': 0,
    'eyes': 1,
    'rostrum': 2,
    'tail': 3
}


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(image)

        label_path = self.label_dir / f"{img_path.stem}.txt"
        target_coords = torch.tensor([0.0, 0.0])
        has_kp = torch.tensor(0.0)

        if label_path.exists():
            try:
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        data = [float(x) for x in line.split()]

                        kp_idx = self.kp_map[self.target_kp]
                        start = 5 + (kp_idx * 3)

                        if len(data) > start + 1:
                            x, y, v = data[start], data[start + 1], data[start + 2]
                            if v > 0:
                                target_coords = torch.tensor([x, y])
                                has_kp = torch.tensor(1.0)
                                break
            except:
                pass

        return img_tensor, target_coords, has_kp, str(img_path)


# ==========================================
# 2. THE FINE-TUNING MODEL
# ==========================================
class PrawnPredictor(nn.Module):
    def __init__(self, pretrained_weights_path=None):
        super().__init__()

        full_model = PrawnMKPModel()

        if pretrained_weights_path:
            print(f"Loading pre-trained weights from {pretrained_weights_path}...")
            state_dict = torch.load(pretrained_weights_path)
            try:
                full_model.load_state_dict(state_dict, strict=False)
                print("Weights loaded successfully.")
            except Exception as e:
                print(f"Warning during loading: {e}")

        # Extract Encoder components
        self.patch_embed = full_model.patch_embed
        self.cls_token = full_model.cls_token
        self.pos_embed = full_model.pos_embed
        self.encoder_blocks = full_model.encoder_blocks
        self.encoder_norm = full_model.encoder_norm

        # Prediction Head
        self.head = nn.Sequential(
            nn.Linear(VIT_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        B = x.shape[0]
        # Encoder Pass
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]  # Add Pos Embed (no CLS yet)

        # Add CLS Token
        cls_tokens = (self.cls_token + self.pos_embed[:, :1, :]).expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.encoder_blocks:
            x = blk(x)
        x = self.encoder_norm(x)

        # Predict from CLS token
        cls_output = x[:, 0]
        coords = self.head(cls_output)
        return coords


# ==========================================
# 3. TRAINING LOOP
# ==========================================
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Fine-Tuning on {device}...")

    # PATHS
    DATA_ROOT = Path(r"C:\Users\carmonta\Desktop\ssl-prawn\mkp\scripts\mkp_data\prawn_2025_circ_small_v1")
    PRETRAINED_PATH = Path("runs/prawn_encoder_best.pth")

    if not PRETRAINED_PATH.exists():
        print(f"⚠️ Warning: Pretrained weights not found at {PRETRAINED_PATH}")
        PRETRAINED_PATH = None

    model = PrawnPredictor(pretrained_weights_path=PRETRAINED_PATH).to(device)

    train_dataset = PrawnKeypointDataset(DATA_ROOT, KEYPOINT_NAME, 'train')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        valid_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")

        for imgs, targets, has_kp, _ in pbar:
            imgs = imgs.to(device)
            targets = targets.to(device)
            has_kp = has_kp.to(device)

            mask = has_kp > 0
            if mask.sum() == 0: continue

            optimizer.zero_grad()
            preds = model(imgs)

            loss = criterion(preds[mask], targets[mask])

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            valid_batches += 1
            pbar.set_postfix({'mse_loss': loss.item()})

        avg_loss = epoch_loss / valid_batches if valid_batches > 0 else 0
        print(f"Epoch {epoch + 1} Avg Loss: {avg_loss:.6f}")

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"runs/prawn_detector_epoch_{epoch + 1}.pth")

    print("Training Complete! The model can now predict coordinates.")
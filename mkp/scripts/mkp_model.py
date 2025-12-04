import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from einops import rearrange
from pathlib import Path
from tqdm import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Import Dataset
try:
    from mkp.scripts.PrawnMKPDataset import PrawnMKPDataset, PATCH_SIZE, NUM_PATCHES
except ImportError:
    from PrawnMKPDataset import PrawnMKPDataset, PATCH_SIZE, NUM_PATCHES

# --- Configuration ---
VIT_DIM = 384
VIT_DEPTH = 6
VIT_HEADS = 6
DECODER_DIM = 256
DECODER_DEPTH = 4
BATCH_SIZE = 16

# 🌟 IMPROVED TRAINING CONFIG
NUM_EPOCHS = 200
BASE_LR = 1.5e-4
WEIGHT_DECAY = 0.05


# --- Components ---

class PatchEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, VIT_DIM, kernel_size=PATCH_SIZE, stride=PATCH_SIZE)

    def forward(self, x):
        x = self.proj(x)  # (B, Dim, H, W)
        return x.flatten(2).transpose(1, 2)  # (B, N, Dim)


class Block(nn.Module):
    """ Standard Transformer Block """

    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        # x shape: (B, N, Dim)
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class PrawnMKPModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.patch_embed = PatchEmbed()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, VIT_DIM))
        self.pos_embed = nn.Parameter(torch.randn(1, NUM_PATCHES + 1, VIT_DIM) * .02)
        self.encoder_blocks = nn.ModuleList([Block(VIT_DIM, VIT_HEADS) for _ in range(VIT_DEPTH)])
        self.encoder_norm = nn.LayerNorm(VIT_DIM)

        # Decoder
        self.decoder_embed = nn.Linear(VIT_DIM, DECODER_DIM)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, DECODER_DIM))
        self.decoder_pos_embed = nn.Parameter(torch.randn(1, NUM_PATCHES + 1, DECODER_DIM) * .02)
        self.decoder_blocks = nn.ModuleList([Block(DECODER_DIM, 8) for _ in range(DECODER_DEPTH)])
        self.decoder_norm = nn.LayerNorm(DECODER_DIM)
        self.decoder_pred = nn.Linear(DECODER_DIM, PATCH_SIZE ** 2 * 3)

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x, unmasked_indices_list, masked_indices_list):
        B = x.shape[0]

        # 1. Patch Embed
        x = self.patch_embed(x)  # (B, N, Dim)

        # 2. Add Pos Embed (excluding CLS)
        x = x + self.pos_embed[:, 1:, :]

        # 3. Encoder Preparation (Visible only + CLS)
        encoded_batch = []
        for i in range(B):
            indices = unmasked_indices_list[i]
            if len(indices) > 0:
                visible_tokens = x[i, indices, :]  # (K, Dim)
            else:
                visible_tokens = x[i]  # Fallback

            # Prepare CLS: (1, 1, Dim) -> squeeze -> (1, Dim)
            cls = (self.cls_token + self.pos_embed[:, :1, :]).squeeze(0)

            # 🌟 FIX: Concatenate (1, Dim) + (K, Dim) -> (K+1, Dim)
            # הסרתי את unsqueeze(0) שהיה כאן וגרם לשגיאה
            combined = torch.cat([cls, visible_tokens], dim=0)

            encoded_batch.append(combined)

        # Pad to max length in batch
        max_len = max([t.shape[0] for t in encoded_batch])
        padded_encoded = torch.zeros(B, max_len, VIT_DIM, device=x.device)

        for i, t in enumerate(encoded_batch):
            padded_encoded[i, :t.shape[0], :] = t

        # Run Encoder Blocks
        x_enc = padded_encoded
        for blk in self.encoder_blocks:
            x_enc = blk(x_enc)
        x_enc = self.encoder_norm(x_enc)

        # 4. Decoder Preparation
        x_dec = self.decoder_embed(x_enc)

        full_batch = []
        for i in range(B):
            visible_dec = x_dec[i, 1:, :]  # Skip CLS
            valid_len = len(unmasked_indices_list[i])
            visible_dec = visible_dec[:valid_len]

            full_seq = torch.zeros(NUM_PATCHES, DECODER_DIM, device=x.device)

            # Place visible
            indices_u = unmasked_indices_list[i].long()
            full_seq[indices_u] = visible_dec

            # Place masks
            indices_m = masked_indices_list[i].long()
            if len(indices_m) > 0:
                mask_tokens = self.mask_token.squeeze(0).repeat(len(indices_m), 1)
                full_seq[indices_m] = mask_tokens

            # Add decoder pos embed
            full_seq = full_seq + self.decoder_pos_embed[:, 1:, :].squeeze(0)
            full_batch.append(full_seq)

        x_full = torch.stack(full_batch)

        for blk in self.decoder_blocks:
            x_full = blk(x_full)
        x_full = self.decoder_norm(x_full)

        pred = self.decoder_pred(x_full)
        return pred


def calculate_loss(model, batch, device):
    imgs = batch['image'].to(device)
    unmasked = [u.to(device) for u in batch['unmasked_indices']]
    masked = [m.to(device) for m in batch['masked_indices']]

    pred = model(imgs, unmasked, masked)

    loss = 0
    valid_samples = 0

    target = rearrange(imgs, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=PATCH_SIZE, p2=PATCH_SIZE)

    for i in range(len(imgs)):
        mask_idx = masked[i]
        if len(mask_idx) == 0: continue

        curr_pred = pred[i, mask_idx, :]
        curr_target = target[i, mask_idx, :]

        loss += F.mse_loss(curr_pred, curr_target)
        valid_samples += 1

    if valid_samples == 0: return torch.tensor(0.0, device=device, requires_grad=True)
    return loss / valid_samples


def collate_fn(batch):
    return {
        'image': torch.stack([b['image'] for b in batch]),
        'unmasked_indices': [b['unmasked_indices'] for b in batch],
        'masked_indices': [b['masked_indices'] for b in batch],
        'image_path': [b['image_path'] for b in batch]
    }


# ---------------------------------------------------
# Visualization Helper
# ---------------------------------------------------
def visualize_masked_sample(dataset, num_samples=2):
    print(f"\n--- Visualizing {num_samples} Masked Samples (Sanity Check) ---")

    import random
    indices = random.sample(range(len(dataset)), num_samples)

    plt.figure(figsize=(10, 5))

    for i, idx in enumerate(indices):
        sample = dataset[idx]
        img_tensor = sample['image']
        masked_indices = sample['masked_indices']

        img_np = img_tensor.permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1)

        patches_per_side = img_np.shape[0] // PATCH_SIZE
        for patch_idx in masked_indices:
            r = int(patch_idx // patches_per_side)
            c = int(patch_idx % patches_per_side)
            y1, y2 = r * PATCH_SIZE, (r + 1) * PATCH_SIZE
            x1, x2 = c * PATCH_SIZE, (c + 1) * PATCH_SIZE

            img_np[y1:y2, x1:x2, :] = 0.0

        plt.subplot(1, num_samples, i + 1)
        plt.imshow(img_np)
        plt.title(f"Sample {i + 1}: Masked Eyes")
        plt.axis('off')

    plt.tight_layout()
    plt.show()
    print("Visualization window closed. Starting training...")


# --- Main Execution ---

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Initialize
    model = PrawnMKPModel().to(device)

    # Optimizer + Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    # 2. Load Data
    print("\n--- Loading Dataset ---")
    data_root = Path(__file__).resolve().parent / "mkp_data" / "train_on_all"
    target_kps = ['eyes']

    try:
        dataset = PrawnMKPDataset(data_root, target_kps, split_type='train')
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
        print(f"Loaded {len(dataset)} images.")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        exit()

    # 🌟 CALL VISUALIZATION
    visualize_masked_sample(dataset, num_samples=2)

    # 3. Training Loop
    print("\n--- Starting Training ---")
    save_dir = Path("runs")
    save_dir.mkdir(exist_ok=True)

    best_loss = float('inf')

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}")

        for batch in pbar:
            optimizer.zero_grad()
            loss = calculate_loss(model, batch, device)

            if torch.isnan(loss):
                print("WARNING: NaN Loss ignored.")
                continue

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({'loss': loss.item(), 'lr': f"{current_lr:.2e}"})

        scheduler.step()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch} Done. Avg Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_dir / "prawn_encoder_best.pth")
            print(f"🎉 New Best Model Saved! (Loss: {best_loss:.4f})")

        if epoch % 10 == 0:
            torch.save(model.state_dict(), save_dir / "prawn_encoder_latest.pth")

    print("Training Complete.")
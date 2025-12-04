import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# --- Configuration ---
PATCH_SIZE = 16
IMAGE_SIZE = 224
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2

# We want to mask exactly one patch (1x1), the one containing the eye
KEYPOINT_MASK_RADIUS_PATCHES = 1

# Mapping
KEYPOINT_MAPPING = {
    'carapace': 0,
    'eyes': 1,
    'rostrum': 2,
    'tail': 3
}


class PrawnMKPDataset(Dataset):
    def __init__(self, data_root: Path, target_keypoints: list, split_type: str = 'train', transform=None):
        self.data_root = data_root
        self.image_dir = data_root / "images" / split_type
        self.label_dir = data_root / "labels" / split_type
        self.target_keypoints = target_keypoints

        self.image_paths = list(self.image_dir.glob("*.jpg")) + list(self.image_dir.glob("*.png"))
        self.label_paths = {p.stem: self.label_dir / f"{p.stem}.txt" for p in self.image_paths}

        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {self.image_dir}")

        # Simple Transform: [0, 1] Scaling only
        self.transform = transform or T.Compose([
            T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def _read_keypoints(self, label_path):
        """Reads YOLO labels."""
        objects = []
        if not label_path or not label_path.exists():
            return objects

        try:
            with open(label_path, 'r') as f:
                for line in f:
                    data = [float(x) for x in line.split()]
                    if len(data) < 5: continue
                    bbox = tuple(data[1:5])
                    kps = {}
                    # Check Keypoints
                    if len(data) > 5:
                        kp_data = data[5:]
                        for i, name in enumerate(KEYPOINT_MAPPING.keys()):
                            # If index exists and visibility > 0
                            if (i * 3 + 2) < len(kp_data) and kp_data[i * 3 + 2] > 0:
                                kps[name] = (kp_data[i * 3], kp_data[i * 3 + 1])
                    objects.append({'bbox': bbox, 'keypoints': kps})
        except:
            pass
        return objects

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image_tensor = self.transform(image)

        objects = self._read_keypoints(self.label_paths.get(img_path.stem))

        # List of patches to mask (eyes only)
        eye_patches_indices = set()

        patches_per_side = IMAGE_SIZE // PATCH_SIZE

        # Iterate over all objects in the image
        for obj in objects:
            for kp_name in self.target_keypoints:
                if kp_name in obj['keypoints']:
                    # Extract normalized coordinates
                    nx, ny = obj['keypoints'][kp_name]

                    # Convert to grid coordinates
                    px = int(nx * patches_per_side)
                    py = int(ny * patches_per_side)

                    # Ensure boundaries (in case it's near the edge)
                    px = max(0, min(px, patches_per_side - 1))
                    py = max(0, min(py, patches_per_side - 1))

                    # Calculate 1D index and add to mask list
                    patch_idx = py * patches_per_side + px
                    eye_patches_indices.add(patch_idx)

        # ---------------------------------------------------------
        # 🌟 New Masking Logic (Exactly as in Debug) 🌟
        # ---------------------------------------------------------

        # 1. Masked = Only the eye patches
        masked_indices = list(eye_patches_indices)

        # 2. Unmasked = All other patches in the image (including body and background)
        all_indices = set(range(NUM_PATCHES))
        unmasked_indices = list(all_indices - eye_patches_indices)

        # Safety: If no eyes found, mask a random patch so code doesn't crash
        if not masked_indices:
            # Choose a random patch to mask (e.g., the center)
            fallback_mask = NUM_PATCHES // 2
            masked_indices = [fallback_mask]
            # Update unmasked
            unmasked_indices = list(all_indices - {fallback_mask})

        return {
            'image': image_tensor,
            'unmasked_indices': torch.tensor(unmasked_indices, dtype=torch.long),
            'masked_indices': torch.tensor(masked_indices, dtype=torch.long),
            'image_path': str(img_path),
            'keypoints_dict': {}, 'bboxes': []
        }
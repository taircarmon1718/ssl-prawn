import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
import sys
import glob

# Add current path to sys.path
sys.path.append(str(Path(__file__).parent))

print("--- 🔄 Importing Modules ---")
try:
    # 1. Import Stage 1 Config & Model (Renamed from 'train' to 'mkp_model')
    try:
        import mkp_model as stage1_config
        from mkp_model import PrawnMKPModel, collate_fn

        print("✅ Loaded Stage 1: mkp_model.py")
    except ImportError as e:
        print(f"⚠️ Could not import 'mkp_model.py'. Trying 'train.py' as fallback...")
        import train as stage1_config
        from train import PrawnMKPModel, collate_fn

    # 2. Import Dataset
    import PrawnMKPDataset as dataset_module
    from PrawnMKPDataset import PrawnMKPDataset

    # 3. Import Stage 2 (Renamed from 'fine_tuning' to 'train_keypoint_detector')
    ft_path = Path(__file__).parent / "train_keypoint_detector.py"

    if not ft_path.exists():
        print(f"⚠️ Warning: 'train_keypoint_detector.py' not found at {ft_path}")
        print("   Stage 2 metrics will be N/A.")
        HAS_STAGE_2 = False
    else:
        try:
            from train_keypoint_detector import PrawnPredictor, PrawnKeypointDataset

            HAS_STAGE_2 = True
            print("✅ Loaded Stage 2: train_keypoint_detector.py")
        except ImportError as e:
            print(f"❌ Error importing 'train_keypoint_detector.py': {e}")
            HAS_STAGE_2 = False
        except Exception as e:
            print(f"❌ Crash inside 'train_keypoint_detector.py': {e}")
            HAS_STAGE_2 = False

    print("✅ All Imports successful.")

except ImportError as e:
    print(f"❌ Critical Error: Missing files. Details: {e}")
    print("Ensure you have: mkp_model.py, train_keypoint_detector.py, PrawnMKPDataset.py")
    exit()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --- NEW: Evaluation Functions ---

def evaluate_stage1_metrics(model, dataset, device, batch_size=16):
    """
    Calculates MSE and PSNR over the entire dataset for Stage 1.
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

    total_mse = 0
    total_psnr = 0
    valid_batches = 0

    print("   • Running Stage 1 Evaluation...")
    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval Stage 1"):
            imgs = batch['image'].to(device)
            unmasked = [u.to(device) for u in batch['unmasked_indices']]
            masked = [m.to(device) for m in batch['masked_indices']]

            # Skip batches with no masked items if any
            if all(len(m) == 0 for m in masked): continue

            # Get predictions
            pred = model(imgs, unmasked, masked)  # (B, N_masked, Patch_dim)

            # Prepare targets
            from einops import rearrange
            target = rearrange(imgs, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                               p1=dataset_module.PATCH_SIZE, p2=dataset_module.PATCH_SIZE)

            batch_mse = 0
            count = 0

            for i in range(len(imgs)):
                mask_idx = masked[i]
                if len(mask_idx) == 0: continue

                curr_pred = pred[i, mask_idx, :]
                curr_target = target[i, mask_idx, :]

                # Calc MSE for this image
                mse = F.mse_loss(curr_pred, curr_target).item()
                batch_mse += mse

                # Calc PSNR: 10 * log10(1 / mse) since images are [0,1]
                if mse > 0:
                    psnr = 10 * math.log10(1.0 / mse)
                else:
                    psnr = 100  # Perfect score
                total_psnr += psnr
                count += 1

            if count > 0:
                total_mse += batch_mse
                valid_batches += count

    avg_mse = total_mse / valid_batches if valid_batches > 0 else 0
    avg_psnr = total_psnr / valid_batches if valid_batches > 0 else 0

    return avg_mse, avg_psnr


def evaluate_stage2_metrics(model, dataset, device):
    """
    Calculates Mean Pixel Error (Euclidean Distance) for Stage 2.
    """
    model.eval()
    # Use batch size 1 for simplicity in logic, or larger if needed
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    total_pixel_error = 0
    total_mse = 0
    count = 0

    print("   • Running Stage 2 Evaluation...")
    with torch.no_grad():
        for img, target_coords, has_kp, _ in tqdm(loader, desc="Eval Stage 2"):
            if has_kp.item() == 0: continue

            img = img.to(device)
            target_coords = target_coords.to(device)  # (1, 2)

            # Predict
            pred_coords = model(img)  # (1, 2)

            # MSE (Normalized 0-1)
            mse = F.mse_loss(pred_coords, target_coords).item()
            total_mse += mse

            # Pixel Error
            # Denormalize to Image Size (224x224)
            W, H = dataset_module.IMAGE_SIZE, dataset_module.IMAGE_SIZE

            gt_x = target_coords[0, 0].item() * W
            gt_y = target_coords[0, 1].item() * H
            pred_x = pred_coords[0, 0].item() * W
            pred_y = pred_coords[0, 1].item() * H

            dist = math.sqrt((gt_x - pred_x) ** 2 + (gt_y - pred_y) ** 2)
            total_pixel_error += dist
            count += 1

    avg_pixel_error = total_pixel_error / count if count > 0 else 0
    avg_mse = total_mse / count if count > 0 else 0

    return avg_mse, avg_pixel_error


# --- Visualization Functions ---

def generate_mkp_visualization(model, dataset, device, save_path="results_comparison.png"):
    model.eval()
    found = False
    idx = 1
    while not found and idx < len(dataset):
        sample = dataset[idx]
        if len(sample['masked_indices']) > 0:
            found = True
        else:
            idx += 1

    if not found: return

    img = sample['image'].unsqueeze(0).to(device)
    unmasked = [sample['unmasked_indices'].to(device)]
    masked = [sample['masked_indices'].to(device)]

    with torch.no_grad():
        pred_pixels = model(img, unmasked, masked)

    img_np = img.squeeze().permute(1, 2, 0).cpu().numpy()
    img_np = np.clip(img_np, 0, 1)

    PATCH_SIZE = dataset_module.PATCH_SIZE
    patches_per_side = img_np.shape[0] // PATCH_SIZE

    masked_img_viz = img_np.copy()
    for mask_idx in masked[0]:
        r = int(mask_idx // patches_per_side)
        c = int(mask_idx % patches_per_side)
        y1, y2 = r * PATCH_SIZE, (r + 1) * PATCH_SIZE
        x1, x2 = c * PATCH_SIZE, (c + 1) * PATCH_SIZE
        masked_img_viz[y1:y2, x1:x2, :] = 0.5

    recon_img = masked_img_viz.copy()
    pred_pixels = pred_pixels.cpu().numpy()

    for i, mask_idx in enumerate(masked[0]):
        patch_pixels = pred_pixels[0, i, :]
        patch_viz = patch_pixels.reshape(PATCH_SIZE, PATCH_SIZE, 3)
        r = int(mask_idx // patches_per_side)
        c = int(mask_idx % patches_per_side)
        y1, y2 = r * PATCH_SIZE, (r + 1) * PATCH_SIZE
        x1, x2 = c * PATCH_SIZE, (c + 1) * PATCH_SIZE
        recon_img[y1:y2, x1:x2, :] = patch_viz

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = ["Original", "Masked Input", "Reconstruction"]
    images = [img_np, masked_img_viz, recon_img]

    for ax, img_show, title in zip(axes, images, titles):
        ax.imshow(img_show)
        ax.set_title(title)
        ax.axis('off')
        for mask_idx in masked[0]:
            r = int(mask_idx // patches_per_side)
            c = int(mask_idx % patches_per_side)
            rect = plt.Rectangle((c * PATCH_SIZE, r * PATCH_SIZE), PATCH_SIZE, PATCH_SIZE,
                                 linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def generate_stage2_visualization(model, dataset, device, save_path="stage2_results.png"):
    model.eval()
    import random
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    sample = None
    for i in indices:
        data = dataset[i]
        if data[2] > 0:
            sample = data
            break

    if sample is None: return

    img_tensor, target_coords, _, _ = sample
    img_input = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        pred_coords = model(img_input)

    target_np = target_coords.numpy()
    pred_np = pred_coords.cpu().numpy()[0]
    img_np = img_tensor.permute(1, 2, 0).numpy()
    img_np = np.clip(img_np, 0, 1)

    H, W = img_np.shape[:2]
    gt_x, gt_y = target_np[0] * W, target_np[1] * H
    pred_x, pred_y = pred_np[0] * W, pred_np[1] * H
    error_px = np.sqrt((gt_x - pred_x) ** 2 + (gt_y - pred_y) ** 2)

    plt.figure(figsize=(8, 8))
    plt.imshow(img_np)
    plt.scatter(gt_x, gt_y, c='lime', s=150, marker='+', linewidths=2, label='Ground Truth')
    plt.scatter(pred_x, pred_y, c='red', s=120, marker='x', linewidths=2, label=f'Pred (Err: {error_px:.1f}px)')
    plt.title("Stage 2 Result")
    plt.legend()
    plt.axis('off')
    plt.savefig(save_path, dpi=300)
    plt.close()


# --- MAIN ---

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 40)
    print("   PRAWN MKP REPORT DATA & METRICS   ")
    print("=" * 40)

    # Initialize stats dictionary to store everything
    report_data = {
        'total_images': 'N/A',
        'image_size': 'N/A',
        'patch_size': 'N/A',
        's1_params': 'N/A',
        's1_mse': 'N/A',
        's1_psnr': 'N/A',
        's1_lr': 'N/A',
        's1_epochs': 'N/A',
        's2_params': 'N/A',
        's2_mse': 'N/A',
        's2_pixel_error': 'N/A',
        's2_lr': 'N/A',
        's2_epochs': 'N/A',
        's1_image_path': 'N/A',
        's2_image_path': 'N/A'
    }

    data_root = Path("mkp_data/train_on_all")
    target_kps = ['eyes']

    # 1. Dataset Stats
    try:
        dataset_s1 = PrawnMKPDataset(data_root, target_kps, split_type='train')
        report_data['total_images'] = len(dataset_s1)
        report_data['image_size'] = f"{dataset_module.IMAGE_SIZE}x{dataset_module.IMAGE_SIZE}"
        report_data['patch_size'] = f"{dataset_module.PATCH_SIZE}x{dataset_module.PATCH_SIZE}"
    except Exception as e:
        print(f"Dataset Load Error: {e}")
        dataset_s1 = None

    # 2. Stage 1 Metrics & Config
    try:
        model_s1 = PrawnMKPModel().to(device)
        report_data['s1_params'] = f"{count_parameters(model_s1):,}"
        report_data['s1_epochs'] = stage1_config.NUM_EPOCHS
        report_data['s1_lr'] = stage1_config.BASE_LR

        weights_s1 = Path("runs/prawn_encoder_best.pth")
        if weights_s1.exists():
            model_s1.load_state_dict(torch.load(weights_s1, map_location=device))

            # Metrics
            if dataset_s1:
                mse, psnr = evaluate_stage1_metrics(model_s1, dataset_s1, device)
                report_data['s1_mse'] = f"{mse:.6f}"
                report_data['s1_psnr'] = f"{psnr:.2f} dB"

                # Viz
                generate_mkp_visualization(model_s1, dataset_s1, device)
                report_data['s1_image_path'] = str(Path("results_comparison.png").absolute())
        else:
            print("   ⚠️ No Stage 1 weights found.")
    except Exception as e:
        print(f"   Error in Stage 1: {e}")

    # 3. Stage 2 Metrics & Config
    if HAS_STAGE_2:
        try:
            # Import config from module
            from train_keypoint_detector import LR as LR2, NUM_EPOCHS as EP2
            model_s2 = PrawnPredictor().to(device)
            report_data['s2_params'] = f"{count_parameters(model_s2):,}"
            report_data['s2_lr'] = LR2
            report_data['s2_epochs'] = EP2

            # Find latest weights
            runs_dir = Path("runs")
            stage2_weights = list(runs_dir.glob("prawn_detector_epoch_*.pth"))

            if stage2_weights:
                stage2_weights.sort(key=lambda x: int(x.stem.split('_')[-1]))
                latest_weight = stage2_weights[-1]
                model_s2.load_state_dict(torch.load(latest_weight, map_location=device))

                # Create dataset
                dataset_s2 = PrawnKeypointDataset(data_root, 'eyes', 'train')

                # Metrics
                mse_s2, pixel_error = evaluate_stage2_metrics(model_s2, dataset_s2, device)
                report_data['s2_mse'] = f"{mse_s2:.6f}"
                report_data['s2_pixel_error'] = f"{pixel_error:.2f} px"

                # Viz
                generate_stage2_visualization(model_s2, dataset_s2, device)
                report_data['s2_image_path'] = str(Path("stage2_results.png").absolute())
            else:
                print("   ⚠️ No Stage 2 weights found (check 'runs/' folder).")
        except Exception as e:
            print(f"   Error in Stage 2 eval: {e}")

    # --- 📋 FINAL REPORT SUMMARY BLOCK ---
    print("\n" + "=" * 60)
    print("       📋 FINAL REPORT SUMMARY (COPY TO OVERLEAF)       ")
    print("=" * 60)

    print("\n--- A. METHODOLOGY DATA ---")
    print(f"• Dataset Size:     {report_data['total_images']} images")
    print(f"• Image Resolution: {report_data['image_size']}")
    print(f"• Patch Size:       {report_data['patch_size']}")
    print(f"• Target Keypoint:  {target_kps[0]}")

    print("\n--- B. MODEL CONFIGURATION ---")
    print(f"[Stage 1 - Pre-Training]")
    print(f"• Architecture:     ViT Masked Autoencoder")
    print(f"• Parameters:       {report_data['s1_params']}")
    print(f"• Epochs:           {report_data['s1_epochs']}")
    print(f"• Learning Rate:    {report_data['s1_lr']}")
    print(f"[Stage 2 - Fine-Tuning]")
    print(f"• Architecture:     ViT Encoder + Regression Head")
    print(f"• Parameters:       {report_data['s2_params']}")
    print(f"• Epochs:           {report_data['s2_epochs']}")
    print(f"• Learning Rate:    {report_data['s2_lr']}")

    print("\n--- C. QUANTITATIVE RESULTS (Use in Table) ---")
    print(f"{'Metric':<25} | {'Value':<15} | {'Meaning'}")
    print("-" * 60)
    print(f"{'S1 Reconstruction MSE':<25} | {report_data['s1_mse']:<15} | Lower is better")
    print(f"{'S1 PSNR':<25} | {report_data['s1_psnr']:<15} | Higher is better (>20 is good)")
    print(f"{'S2 Regression MSE':<25} | {report_data['s2_mse']:<15} | Coordinate precision")
    print(f"{'S2 Mean Pixel Error':<25} | {report_data['s2_pixel_error']:<15} | Avg mistake distance")

    print("\n--- D. FIGURES ---")
    print(f"• Figure 1 (Reconstruction): {report_data['s1_image_path']}")
    print(f"• Figure 2 (Localization):   {report_data['s2_image_path']}")

    print("=" * 60)


if __name__ == "__main__":
    main()
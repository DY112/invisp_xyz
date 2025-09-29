#!/usr/bin/env python3
"""
Test script for InvISP on sRGB↔XYZ with configurable input type and model flow.

Usage example (run from project_root/scripts):
  python test_invisp_xyz.py \
    --checkpoint ../exps/run_name/checkpoint/0002.pth \
    --input_type xyz \
    --model_flow forward

This will save visualizations under (relative to execution directory):
  ../results/run_name/test/0002/{stem}_psnr_{:.2f}_ssim_{:.4f}.jpg
"""

import os
import sys
import math
from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# Add project root to Python path (project_root = parent of this file's directory)
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.model import InvISPNet
from dataset.XYZ_dataset import SRGB2XYZDataset


def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor, max_val: float = 1.0) -> float:
    """Calculate PSNR between two images in [0,1]. Accepts CHW or BCHW tensors."""
    if img1.dim() == 4:
        img1 = img1[0]
        img2 = img2[0]
    mse = F.mse_loss(img1, img2).item()
    if mse < 1e-10:
        return 100.0
    return 20.0 * math.log10(max_val / math.sqrt(mse))


def calculate_ssim(img1: torch.Tensor, img2: torch.Tensor, data_range: float = 1.0) -> float:
    """Calculate SSIM using skimage for tensors in [0,1]. Accepts CHW or BCHW tensors."""
    if img1.dim() == 4:
        img1 = img1[0]
        img2 = img2[0]
    img1_np = img1.detach().cpu().permute(1, 2, 0).numpy()
    img2_np = img2.detach().cpu().permute(1, 2, 0).numpy()
    img1_np = np.clip(img1_np, 0, data_range)
    img2_np = np.clip(img2_np, 0, data_range)
    from skimage.metrics import structural_similarity as ssim
    # channel_axis supported in recent skimage; keep multichannel for backward compat
    try:
        return ssim(img1_np, img2_np, data_range=data_range, channel_axis=2)
    except TypeError:
        return ssim(img1_np, img2_np, data_range=data_range, multichannel=True)


def compute_metrics_unit01(target_01: torch.Tensor, pred_01: torch.Tensor) -> Tuple[float, float]:
    """Return (PSNR, SSIM) for tensors in [0,1]."""
    psnr = calculate_psnr(pred_01, target_01, max_val=1.0)
    ssim_val = calculate_ssim(pred_01, target_01, data_range=1.0)
    return psnr, ssim_val


def visualize_triplet(input_01: torch.Tensor, target_01: torch.Tensor, pred_01: torch.Tensor,
                      filename: str, title_prefix: str, save_dir: Path) -> Tuple[float, float, Path]:
    """Plot input/target/pred in [0,1], compute metrics, and save figure."""
    input_hwc = input_01.detach().float().permute(1, 2, 0).cpu().numpy()
    target_hwc = target_01.detach().float().permute(1, 2, 0).cpu().numpy()
    pred_hwc = pred_01.detach().float().permute(1, 2, 0).cpu().numpy()

    psnr, ssim_val = compute_metrics_unit01(torch.from_numpy(target_hwc).permute(2, 0, 1),
                                            torch.from_numpy(pred_hwc).permute(2, 0, 1))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(np.clip(input_hwc, 0, 1)); axes[0].set_title('Input', fontsize=12); axes[0].axis('off')
    axes[1].imshow(np.clip(target_hwc, 0, 1)); axes[1].set_title('Target', fontsize=12); axes[1].axis('off')
    axes[2].imshow(np.clip(pred_hwc, 0, 1)); axes[2].set_title(f'Predicted\nPSNR: {psnr:.2f}, SSIM: {ssim_val:.4f}', fontsize=12); axes[2].axis('off')
    plt.suptitle(f'{title_prefix} - {filename}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    stem = Path(filename).stem
    save_dir.mkdir(parents=True, exist_ok=True)
    save_filename = f"{stem}_psnr_{psnr:.2f}_ssim_{ssim_val:.4f}.jpg"
    save_path = save_dir / save_filename
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved: {save_path}")
    return psnr, ssim_val, save_path


def parse_run_and_ckpt_names(ckpt_path: Path) -> Tuple[str, str]:
    """Extract run_name and ckpt_name from a path like exps/run_name/checkpoint/ckpt.pth"""
    parts = ckpt_path.as_posix().split('/')
    run_name = None
    ckpt_name = ckpt_path.stem
    for i, p in enumerate(parts):
        if p == 'exps' and i + 2 < len(parts) and parts[i + 2] == 'checkpoint':
            run_name = parts[i + 1]
            break
    if run_name is None:
        # Fallback: parent of 'checkpoint' dir if present
        try:
            idx = parts.index('checkpoint')
            if idx > 0:
                run_name = parts[idx - 1]
        except ValueError:
            run_name = 'unknown_run'
    return run_name, ckpt_name


def build_dataloader(manifest_path: Path, dataset_subsets, image_size, xyz_norm_mode,
                     input_type: str, num_workers: int = 4, batch_size: int = 1) -> DataLoader:
    """Create test dataloader using test split and no random cropping."""
    training_flow = 'forward' if input_type == 'xyz' else 'backward'
    dataset = SRGB2XYZDataset(
        manifest_path=manifest_path,
        dataset_subsets=dataset_subsets,
        image_size=image_size,
        xyz_norm_mode=xyz_norm_mode,
        enable_random_crop=False,
        training_flow=training_flow,
        split='test',
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="InvISP XYZ Test")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint (e.g., ../exps/run/checkpoint/xxxx.pth)')
    parser.add_argument('--input_type', type=str, required=True, choices=['xyz', 'srgb'], help='Input domain to feed into model')
    parser.add_argument('--model_flow', type=str, required=True, choices=['forward', 'backward'], help='Use model forward or inverse mapping')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_root', type=str, default='../results', help='Where to save results (relative to execution directory by default)')
    parser.add_argument('--manifest_path', type=str, default='../../rawgen/trainval.json')
    parser.add_argument('--dataset_subsets', nargs='+', default=['all'])
    parser.add_argument('--image_size', type=int, default=None, help='Optional resize (square). If None, keep original size')
    parser.add_argument('--xyz_norm_mode', type=str, default='unit', choices=['unit', 'd65'])
    parser.add_argument('--max_images', type=int, default=None)
    args = parser.parse_args()

    # Resolve paths relative to current working directory (user expects ../exps and ../results from scripts/)
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    run_name, ckpt_name = parse_run_and_ckpt_names(ckpt_path)
    save_root = Path(args.save_root)
    out_dir = save_root / run_name / 'test' / ckpt_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device(args.device)

    # Build model
    net = InvISPNet(channel_in=3, channel_out=3, block_num=8)
    net = net.to(device)

    # Load checkpoint (pure state_dict)
    state = torch.load(str(ckpt_path), map_location=device)
    try:
        net.load_state_dict(state)
    except RuntimeError:
        # Maybe saved from DDP with module. prefix removed; try strict=False
        net.load_state_dict(state, strict=False)
    net.eval()

    # Data
    image_size = None if args.image_size in (None, 0) else (args.image_size, args.image_size)
    dataloader = build_dataloader(
        manifest_path=Path(args.manifest_path),
        dataset_subsets=args.dataset_subsets,
        image_size=image_size,
        xyz_norm_mode=args.xyz_norm_mode,
        input_type=args.input_type,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
    )

    # Inference loop
    total_psnr = 0.0
    total_ssim = 0.0
    count = 0

    title_prefix = f"{args.input_type.upper()}→({'sRGB' if args.input_type=='xyz' else 'XYZ'}) | flow={args.model_flow} | run={run_name} | ckpt={ckpt_name}"

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Testing', leave=True)
        for batch in pbar:
            input_data, target_data, filename = batch
            # batch is size B, but we'll process per-image for visualization and metrics
            bsz = input_data.shape[0]
            for b in range(bsz):
                x = input_data[b:b+1].to(device)
                y = target_data[b:b+1].to(device)
                name = filename[b]

                if args.model_flow == 'forward':
                    pred = net(x)
                else:
                    pred = net.forward_rev(x)

                pred = torch.clamp(pred, 0.0, 1.0)

                # Compute metrics on [0,1]
                psnr, ssim_val, save_path = visualize_triplet(
                    input_01=x[0].detach().cpu(),
                    target_01=y[0].detach().cpu(),
                    pred_01=pred[0].detach().cpu(),
                    filename=name,
                    title_prefix=title_prefix,
                    save_dir=out_dir,
                )

                total_psnr += psnr
                total_ssim += ssim_val
                count += 1

                if args.max_images is not None and count >= args.max_images:
                    break

            if args.max_images is not None and count >= args.max_images:
                break

            # Update progress bar
            avg_psnr = total_psnr / max(count, 1)
            avg_ssim = total_ssim / max(count, 1)
            pbar.set_postfix({
                'avg_psnr': f'{avg_psnr:.2f}',
                'avg_ssim': f'{avg_ssim:.4f}',
                'count': count,
            })

    if count > 0:
        print(f"Done. {count} images | Mean PSNR: {total_psnr / count:.2f} | Mean SSIM: {total_ssim / count:.4f}")
    else:
        print("No images processed.")


if __name__ == '__main__':
    torch.set_num_threads(4)
    main()



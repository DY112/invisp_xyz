#!/usr/bin/env python3
"""
Training script for InvISP with sRGB-XYZ image pairs.

This script trains the InvISP network to perform bidirectional conversion
between sRGB and XYZ color spaces using the invertible architecture.
"""

import numpy as np
import os
import sys
import time
import random
import argparse
import json
import math
from pathlib import Path
from typing import Union, Tuple

# Add project root to Python path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import lr_scheduler
from torch.amp import autocast, GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb
from PIL import Image
import cv2
from tqdm import tqdm

from model.model import InvISPNet
from dataset.XYZ_dataset import SRGB2XYZDataset
from utils.JPEG import DiffJPEG

# DDP initialization
def init_ddp(local_rank, world_size):
    """Initialize distributed training."""
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=local_rank
    )
    torch.cuda.set_device(local_rank)
    # Only print from main process
    if local_rank == 0:
        print(f"[INFO] DDP initialized: rank {local_rank}/{world_size}")

def cleanup_ddp():
    """Cleanup distributed training."""
    dist.destroy_process_group()

# Metrics calculation utilities
def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor, max_val: float = 1.0) -> float:
    """Calculate PSNR between two images."""
    mse = F.mse_loss(img1, img2).item()
    if mse < 1e-10:
        return 100.0
    return 20 * math.log10(max_val / math.sqrt(mse))

def calculate_ssim(img1: torch.Tensor, img2: torch.Tensor, data_range: float = 1.0) -> float:
    """Calculate SSIM between two images using skimage."""
    # Handle batch dimension - take first image if batch size > 1
    if img1.dim() == 4:  # [batch_size, channels, height, width]
        img1 = img1[0]  # Take first image from batch
        img2 = img2[0]  # Take first image from batch
    
    # Convert tensors to numpy arrays [channels, height, width] -> [height, width, channels]
    img1_np = img1.detach().cpu().permute(1, 2, 0).numpy()
    img2_np = img2.detach().cpu().permute(1, 2, 0).numpy()
    
    # Ensure images are in [0, data_range] range
    img1_np = np.clip(img1_np, 0, data_range)
    img2_np = np.clip(img2_np, 0, data_range)
    
    # Calculate SSIM using skimage
    from skimage.metrics import structural_similarity as ssim
    return ssim(img1_np, img2_np, data_range=data_range, multichannel=True, channel_axis=2)

def calculate_metrics(output: torch.Tensor, target: torch.Tensor) -> Tuple[float, float]:
    """Calculate both PSNR and SSIM metrics."""
    psnr = calculate_psnr(output, target)
    ssim = calculate_ssim(output, target)
    return psnr, ssim

# Visualization utilities
def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert tensor to PIL Image."""
    # Ensure tensor is in [0, 1] range
    tensor = torch.clamp(tensor, 0, 1)
    # Convert to numpy and transpose from CHW to HWC
    np_img = tensor.detach().cpu().permute(1, 2, 0).numpy()
    # Convert to uint8
    np_img = (np_img * 255).astype(np.uint8)
    return Image.fromarray(np_img)

def save_validation_samples(net, dataloader, args, epoch, results_dir, num_samples=5):
    """Save validation samples with forward/reverse reconstruction."""
    net.eval()
    saved_count = 0
    
    with torch.no_grad():
        for i_batch, (input_data, target_data, filename) in enumerate(dataloader):
            if saved_count >= num_samples:
                break
                
            # Move to GPU
            input_data = input_data.cuda()
            target_data = target_data.cuda()
            
            # Forward pass
            with autocast('cuda'):
                output = net(input_data)
                output = torch.clamp(output, 0, 1)
                
                # Reverse pass
                if hasattr(net, 'module'):
                    reconstructed_input = net.module.forward_rev(output)
                else:
                    reconstructed_input = net.forward_rev(output)
                reconstructed_input = torch.clamp(reconstructed_input, 0, 1)
            
            # Calculate metrics (using first image in batch)
            forward_psnr, forward_ssim = calculate_metrics(output[0:1], target_data[0:1])
            reverse_psnr, reverse_ssim = calculate_metrics(reconstructed_input[0:1], input_data[0:1])
            
            # Create visualization
            if args.training_flow == "forward":
                # XYZ -> sRGB: input=XYZ, target=sRGB, output=sRGB
                input_img = tensor_to_pil(input_data[0])
                target_img = tensor_to_pil(target_data[0])
                output_img = tensor_to_pil(output[0])
                recon_img = tensor_to_pil(reconstructed_input[0])
                
                # Create side-by-side visualization
                width, height = input_img.size
                combined_img = Image.new('RGB', (width * 4, height))
                combined_img.paste(input_img, (0, 0))
                combined_img.paste(target_img, (width, 0))
                combined_img.paste(output_img, (width * 2, 0))
                combined_img.paste(recon_img, (width * 3, 0))
                
                # Save with metrics in filename
                base_name = Path(filename[0]).stem
                save_path = results_dir / f"valid_epoch{epoch:02d}_{base_name}_psnr_{forward_psnr:.2f}_ssim_{forward_ssim:.3f}.jpg"
                combined_img.save(save_path)
                
            else:
                # sRGB -> XYZ: input=sRGB, target=XYZ, output=XYZ
                input_img = tensor_to_pil(input_data[0])
                target_img = tensor_to_pil(target_data[0])
                output_img = tensor_to_pil(output[0])
                recon_img = tensor_to_pil(reconstructed_input[0])
                
                # Create side-by-side visualization
                width, height = input_img.size
                combined_img = Image.new('RGB', (width * 4, height))
                combined_img.paste(input_img, (0, 0))
                combined_img.paste(target_img, (width, 0))
                combined_img.paste(output_img, (width * 2, 0))
                combined_img.paste(recon_img, (width * 3, 0))
                
                # Save with metrics in filename
                base_name = Path(filename[0]).stem
                save_path = results_dir / f"valid_epoch{epoch:02d}_{base_name}_psnr_{forward_psnr:.2f}_ssim_{forward_ssim:.3f}.jpg"
                combined_img.save(save_path)
            
            saved_count += 1
    
    net.train()

# GPU selection - will be overridden by gpu_ids argument if provided
def setup_gpu(args):
    """Setup GPU configuration."""
    if args.distributed:
        # For DDP, use local_rank
        torch.cuda.set_device(args.local_rank)
        # Only print from main process
        if args.local_rank == 0:
            print(f"[INFO] DDP: Using GPU {args.local_rank}")
    elif args.gpu_ids != "0":
        # Use specified GPU IDs
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
        print(f"[INFO] Using GPUs: {args.gpu_ids}")
    else:
        # Auto-select GPU with most free memory
        try:
            os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
            with open('tmp', 'r') as f:
                lines = f.readlines()
            if lines:
                gpu_id = str(np.argmax([int(x.split()[2]) for x in lines]))
                os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
                print(f"[INFO] Auto-selected GPU: {gpu_id}")
            else:
                os.environ['CUDA_VISIBLE_DEVICES'] = '0'
                print("[INFO] Using GPU: 0 (fallback)")
            os.system('rm tmp')
        except:
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            print("[INFO] Using GPU: 0 (fallback)")

# DiffJPEG will be initialized after GPU setup

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="InvISP XYZ Training")
    
    # Required arguments
    parser.add_argument("--task", type=str, required=True, help="Name of this training")
    
    # Dataset arguments
    parser.add_argument("--manifest_path", type=str, 
                       default="/media/ssd2/users/dykim/dataset/sRGB_XYZ_pairs/trainval.json",
                       help="Path to manifest JSON file")
    parser.add_argument("--dataset_subsets", nargs='+', default=["all"],
                       help="Dataset subsets to use (e.g., a5k nus raise)")
    parser.add_argument("--training_flow", type=str, default="forward", 
                       choices=["forward", "backward"],
                       help="Training flow: forward (XYZ->sRGB) or backward (sRGB->XYZ)")
    parser.add_argument("--image_size", type=int, default=512,
                       help="Image size for training (square)")
    parser.add_argument("--xyz_norm_mode", type=str, default="unit",
                       choices=["unit", "d65"], help="XYZ normalization mode")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=300, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--rgb_weight", type=float, default=1.0, help="Weight for RGB loss")
    parser.add_argument("--xyz_weight", type=float, default=1.0, help="Weight for XYZ loss")
    parser.add_argument("--bidirectional", dest='bidirectional', action='store_true',
                       help="Use bidirectional loss (both forward and backward)")
    
    # Multi-GPU parameters
    parser.add_argument("--gpu_ids", type=str, default="0", help="GPU IDs to use (e.g., '0,1,2,3')")
    parser.add_argument("--distributed", dest='distributed', action='store_true',
                       help="Use distributed training")
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training")
    parser.add_argument("--world_size", type=int, default=1, help="World size for distributed training")
    
    # Training options
    parser.add_argument("--out_path", type=str, default="./exps/", help="Path to save checkpoint.")
    parser.add_argument("--resume", dest='resume', action='store_true', help="Resume training.")
    parser.add_argument("--loss", type=str, default="L1", choices=["L1", "L2"], help="Loss function.")
    parser.add_argument("--aug", dest='aug', action='store_true', help="Use data augmentation.")
    
    # JPEG compression options
    parser.add_argument("--use_jpeg", dest='use_jpeg', action='store_true', 
                       help="Use JPEG compression simulation (disabled by default)")
    parser.add_argument("--jpeg_quality", type=int, default=90, 
                       help="JPEG quality factor (1-100, default: 90)")
    
    # Validation options
    parser.add_argument("--val_freq", type=int, default=5, 
                       help="Validation frequency (every N epochs, default: 5)")
    parser.add_argument("--val_batch_size", type=int, default=1, 
                       help="Validation batch size (default: 1)")
    parser.add_argument("--val_samples", type=int, default=5, 
                       help="Number of validation samples to visualize (default: 5)")
    
    # Wandb options
    parser.add_argument("--use_wandb", dest='use_wandb', action='store_true',
                       help="Use wandb for logging (disabled by default)")
    parser.add_argument("--wandb_project", type=str, default="invisp-xyz",
                       help="Wandb project name (default: invisp-xyz)")
    parser.add_argument("--wandb_entity", type=str, default=None,
                       help="Wandb entity name (optional)")
    
    args = parser.parse_args()
    
    # Convert image_size to tuple if needed
    if isinstance(args.image_size, int):
        args.image_size = (args.image_size, args.image_size)
    
    return args

def setup_directories(args):
    """Create output directories."""
    os.makedirs(args.out_path, exist_ok=True)
    os.makedirs(args.out_path + f"{args.task}", exist_ok=True)
    os.makedirs(args.out_path + f"{args.task}/checkpoint", exist_ok=True)
    
    # Create results directory for validation samples
    results_dir = Path("results") / args.task
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save command line arguments
    with open(args.out_path + f"{args.task}/commandline_args.json", 'w') as f:
        json.dump(args.__dict__, f, indent=2)

def create_dataset(args, is_train=True):
    """Create dataset for training or validation."""
    dataset = SRGB2XYZDataset(
        manifest_path=Path(args.manifest_path),
        dataset_subsets=args.dataset_subsets,
        image_size=args.image_size,
        xyz_norm_mode=args.xyz_norm_mode,
        crop_size_min=256,
        crop_size_max=512,
        crop_prob=0.8,
        enable_random_crop=is_train,
        training_flow=args.training_flow,
        is_train=is_train
    )
    return dataset

def create_dataloader(dataset, args, is_train=True):
    """Create dataloader with appropriate sampler."""
    if args.distributed and is_train:
        sampler = DistributedSampler(dataset, shuffle=True)
        shuffle = False  # sampler handles shuffling
    else:
        sampler = None
        shuffle = True
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=4,
        drop_last=True
    )
    return dataloader

def compute_loss(output, target, loss_type="L1"):
    """Compute loss between output and target."""
    if loss_type == "L1":
        return F.l1_loss(output, target)
    elif loss_type == "L2":
        return F.mse_loss(output, target)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

def train_epoch(net, dataloader, optimizer, args, epoch, diffjpeg_instance, scaler):
    """Train for one epoch."""
    net.train()
    total_loss = 0.0
    total_rgb_loss = 0.0
    total_xyz_loss = 0.0
    num_batches = 0
    
    # Create progress bar (only on main process)
    if not args.distributed or args.local_rank == 0:
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}", 
                   leave=False, disable=False)
    else:
        pbar = dataloader
    
    for i_batch, (input_data, target_data, filename) in enumerate(pbar):
        step_time = time.time()
        
        # Move to GPU
        input_data = input_data.cuda()
        target_data = target_data.cuda()
        
        # Forward pass with AMP
        with autocast('cuda'):
            output = net(input_data)
            output = torch.clamp(output, 0, 1)
            
            # Compute primary loss
            primary_loss = compute_loss(output, target_data, args.loss)
            
            # Apply JPEG compression simulation if enabled
            if diffjpeg_instance is not None:
                compressed_output = diffjpeg_instance(output)
                # Ensure contiguous memory layout to avoid gradient stride issues
                compressed_output = compressed_output.contiguous()
            else:
                # Use output directly without JPEG compression
                compressed_output = output
            
            # Compute reconstruction loss (inverse direction)
            if hasattr(net, 'module'):
                # DDP case
                reconstructed_input = net.module.forward_rev(compressed_output)
            else:
                # Single GPU case
                reconstructed_input = net.forward_rev(compressed_output)
            reconstruction_loss = compute_loss(reconstructed_input, input_data, args.loss)
        
        # Combine losses
        if args.bidirectional:
            # Use both directions for training
            if args.training_flow == "forward":
                # XYZ -> sRGB -> XYZ
                loss = args.rgb_weight * primary_loss + args.xyz_weight * reconstruction_loss
                rgb_loss = primary_loss
                xyz_loss = reconstruction_loss
            else:
                # sRGB -> XYZ -> sRGB
                loss = args.xyz_weight * primary_loss + args.rgb_weight * reconstruction_loss
                xyz_loss = primary_loss
                rgb_loss = reconstruction_loss
        else:
            # Use only primary direction
            loss = primary_loss
            if args.training_flow == "forward":
                rgb_loss = primary_loss
                xyz_loss = torch.tensor(0.0, device=input_data.device)
            else:
                xyz_loss = primary_loss
                rgb_loss = torch.tensor(0.0, device=input_data.device)
        
        # Backward pass with AMP
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        
        # Ensure gradients are properly synchronized in DDP
        if args.distributed:
            # Manually synchronize gradients to avoid stride issues
            for param in net.parameters():
                if param.grad is not None:
                    param.grad = param.grad.contiguous()
        
        scaler.step(optimizer)
        scaler.update()
        
        # Update statistics
        total_loss += loss.item()
        total_rgb_loss += rgb_loss.item()
        total_xyz_loss += xyz_loss.item()
        num_batches += 1
        
        # Update progress bar (only on main process)
        if not args.distributed or args.local_rank == 0:
            pbar.set_postfix({
                'Loss': f'{loss.item():.5f}',
                'RGB': f'{rgb_loss.item():.5f}',
                'XYZ': f'{xyz_loss.item():.5f}',
                'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })
    
    return total_loss / num_batches, total_rgb_loss / num_batches, total_xyz_loss / num_batches

def validate_epoch(net, dataloader, args, epoch, diffjpeg_instance):
    """Validate for one epoch."""
    net.eval()
    total_loss = 0.0
    total_rgb_loss = 0.0
    total_xyz_loss = 0.0
    total_forward_psnr = 0.0
    total_forward_ssim = 0.0
    total_reverse_psnr = 0.0
    total_reverse_ssim = 0.0
    num_batches = 0
    
    # Create progress bar (only on main process)
    if not args.distributed or args.local_rank == 0:
        pbar = tqdm(dataloader, desc=f"Validation", 
                   leave=False, disable=False)
    else:
        pbar = dataloader
    
    with torch.no_grad():
        for i_batch, (input_data, target_data, filename) in enumerate(pbar):
            # Move to GPU
            input_data = input_data.cuda()
            target_data = target_data.cuda()
            
            # Forward pass with AMP
            with autocast('cuda'):
                output = net(input_data)
                output = torch.clamp(output, 0, 1)
                
                # Compute primary loss
                primary_loss = compute_loss(output, target_data, args.loss)
                
                # Apply JPEG compression simulation if enabled
                if diffjpeg_instance is not None:
                    compressed_output = diffjpeg_instance(output)
                    compressed_output = compressed_output.contiguous()
                else:
                    compressed_output = output
                
                # Compute reconstruction loss (inverse direction)
                if hasattr(net, 'module'):
                    # DDP case
                    reconstructed_input = net.module.forward_rev(compressed_output)
                else:
                    # Single GPU case
                    reconstructed_input = net.forward_rev(compressed_output)
                reconstruction_loss = compute_loss(reconstructed_input, input_data, args.loss)
            
            # Combine losses
            if args.bidirectional:
                # Use both directions for training
                if args.training_flow == "forward":
                    # XYZ -> sRGB -> XYZ
                    loss = args.rgb_weight * primary_loss + args.xyz_weight * reconstruction_loss
                    rgb_loss = primary_loss
                    xyz_loss = reconstruction_loss
                else:
                    # sRGB -> XYZ -> sRGB
                    loss = args.xyz_weight * primary_loss + args.rgb_weight * reconstruction_loss
                    xyz_loss = primary_loss
                    rgb_loss = reconstruction_loss
            else:
                # Use only primary direction
                loss = primary_loss
                if args.training_flow == "forward":
                    rgb_loss = primary_loss
                    xyz_loss = torch.tensor(0.0, device=input_data.device)
                else:
                    xyz_loss = primary_loss
                    rgb_loss = torch.tensor(0.0, device=input_data.device)
            
            # Calculate metrics (using first image in batch)
            forward_psnr, forward_ssim = calculate_metrics(output[0:1], target_data[0:1])
            reverse_psnr, reverse_ssim = calculate_metrics(reconstructed_input[0:1], input_data[0:1])
            
            # Update statistics
            total_loss += loss.item()
            total_rgb_loss += rgb_loss.item()
            total_xyz_loss += xyz_loss.item()
            total_forward_psnr += forward_psnr
            total_forward_ssim += forward_ssim
            total_reverse_psnr += reverse_psnr
            total_reverse_ssim += reverse_ssim
            num_batches += 1
            
            # Update progress bar (only on main process)
            if not args.distributed or args.local_rank == 0:
                pbar.set_postfix({
                    'Loss': f'{loss.item():.5f}',
                    'F_PSNR': f'{forward_psnr:.2f}',
                    'R_PSNR': f'{reverse_psnr:.2f}',
                    'F_SSIM': f'{forward_ssim:.3f}'
                })
    
    net.train()
    
    return {
        'loss': total_loss / num_batches,
        'rgb_loss': total_rgb_loss / num_batches,
        'xyz_loss': total_xyz_loss / num_batches,
        'forward_psnr': total_forward_psnr / num_batches,
        'forward_ssim': total_forward_ssim / num_batches,
        'reverse_psnr': total_reverse_psnr / num_batches,
        'reverse_ssim': total_reverse_ssim / num_batches
    }

def main():
    """Main training function."""
    args = parse_args()

    if 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])
    if 'WORLD_SIZE' in os.environ:
        args.world_size = int(os.environ['WORLD_SIZE'])
    
    # Initialize DDP if needed
    if args.distributed:
        init_ddp(args.local_rank, args.world_size)
        # Only print on main process
        if args.local_rank == 0:
            print(f"Parsed arguments: {args}")
    else:
        print(f"Parsed arguments: {args}")
    
    # Setup GPU configuration
    setup_gpu(args)
    
    # Setup directories (only on main process)
    if not args.distributed or args.local_rank == 0:
        setup_directories(args)
    
    # Initialize wandb (only on main process)
    if args.use_wandb and (not args.distributed or args.local_rank == 0):
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.task,
            config=vars(args)
        )
        print(f"[INFO] Wandb initialized for project: {args.wandb_project}")
    
    # Initialize DiffJPEG conditionally
    diffjpeg_instance = None
    if args.use_jpeg:
        if args.distributed:
            diffjpeg_instance = DiffJPEG(differentiable=True, quality=args.jpeg_quality).cuda(args.local_rank)
        else:
            diffjpeg_instance = DiffJPEG(differentiable=True, quality=args.jpeg_quality).cuda()
        if not args.distributed or args.local_rank == 0:
            print(f"[INFO] JPEG compression simulation enabled (quality: {args.jpeg_quality})")
    else:
        if not args.distributed or args.local_rank == 0:
            print("[INFO] JPEG compression simulation disabled")
    
    # Initialize model
    net = InvISPNet(channel_in=3, channel_out=3, block_num=8)
    
    # Move model to GPU
    if args.distributed:
        net = net.cuda(args.local_rank)
    else:
        net = net.cuda()
    
    # Setup DDP if specified with optimized settings
    if args.distributed:
        net = DDP(net, device_ids=[args.local_rank], output_device=args.local_rank, 
                 find_unused_parameters=False, broadcast_buffers=False)
        if args.local_rank == 0:
            print(f"[INFO] Using DistributedDataParallel with {args.world_size} GPUs")
    else:
        print("[INFO] Using single GPU training")
    
    # Load pretrained weights if resuming
    if args.resume:
        checkpoint_path = args.out_path + f"{args.task}/checkpoint/latest.pth"
        if os.path.isfile(checkpoint_path):
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=f'cuda:{args.local_rank}' if args.distributed else 'cuda')
            if args.distributed:
                net.module.load_state_dict(checkpoint)
            else:
                net.load_state_dict(checkpoint)
            if not args.distributed or args.local_rank == 0:
                print(f"[INFO] Loaded checkpoint: {checkpoint_path}")
        else:
            if not args.distributed or args.local_rank == 0:
                print(f"[WARNING] Checkpoint not found: {checkpoint_path}")
    
    # Initialize optimizer and scheduler
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50, 80], gamma=0.5)
    
    # Initialize AMP scaler
    scaler = GradScaler('cuda')
    if not args.distributed or args.local_rank == 0:
        print("[INFO] AMP (Automatic Mixed Precision) enabled for memory optimization")
    
    # Create datasets
    if not args.distributed or args.local_rank == 0:
        print("[INFO] Creating training dataset...")
    train_dataset = create_dataset(args, is_train=True)
    train_dataloader = create_dataloader(train_dataset, args, is_train=True)
    
    # Create validation dataset
    if not args.distributed or args.local_rank == 0:
        print("[INFO] Creating validation dataset...")
    val_dataset = create_dataset(args, is_train=False)
    val_dataloader = create_dataloader(val_dataset, args, is_train=False)
    
    # Calculate effective batch size
    if args.distributed:
        effective_batch_size = args.batch_size * args.world_size
        if args.local_rank == 0:
            print(f"[INFO] DDP batch size: {args.batch_size} per GPU, {effective_batch_size} total")
    else:
        effective_batch_size = args.batch_size
    
    if not args.distributed or args.local_rank == 0:
        print(f"[INFO] Training dataset size: {len(train_dataset)}")
        print(f"[INFO] Validation dataset size: {len(val_dataset)}")
        print(f"[INFO] Training flow: {args.training_flow}")
        print(f"[INFO] Image size: {args.image_size}")
        print(f"[INFO] Dataset subsets: {args.dataset_subsets}")
        print(f"[INFO] Validation frequency: every {args.val_freq} epochs")
    
    # Training loop
    if not args.distributed or args.local_rank == 0:
        print("[INFO] Starting training...")
    
    for epoch in range(args.epochs):
        epoch_time = time.time()
        
        # Set epoch for DistributedSampler
        if args.distributed:
            train_dataloader.sampler.set_epoch(epoch)
        
        # Train for one epoch
        avg_loss, avg_rgb_loss, avg_xyz_loss = train_epoch(
            net, train_dataloader, optimizer, args, epoch, diffjpeg_instance, scaler
        )
        
        # Update learning rate
        scheduler.step()
        
        # Validation (only on main process)
        val_metrics = None
        if (not args.distributed or args.local_rank == 0) and (epoch + 1) % args.val_freq == 0:
            val_time = time.time()
            
            # Run validation
            val_metrics = validate_epoch(net, val_dataloader, args, epoch, diffjpeg_instance)
            
            # Save validation samples
            results_dir = Path("results") / args.task
            save_validation_samples(net, val_dataloader, args, epoch+1, results_dir, args.val_samples)
            
            val_duration = time.time() - val_time
            print(f"[INFO] Validation completed in {val_duration:.1f}s")
        
        # Save checkpoint (only on main process)
        if not args.distributed or args.local_rank == 0:
            # Get model state dict
            if args.distributed:
                model_state_dict = net.module.state_dict()
            else:
                model_state_dict = net.state_dict()
            
            torch.save(model_state_dict, args.out_path + f"{args.task}/checkpoint/latest.pth")
            
            # Save periodic checkpoints
            if (epoch + 1) % 10 == 0:
                checkpoint_path = args.out_path + f"{args.task}/checkpoint/{epoch+1:04d}.pth"
                torch.save(model_state_dict, checkpoint_path)
                print(f"[INFO] Saved checkpoint: {checkpoint_path}")
        
        # Log to wandb (only on main process)
        if args.use_wandb and (not args.distributed or args.local_rank == 0):
            log_dict = {
                'epoch': epoch + 1,
                'train/loss': avg_loss,
                'train/rgb_loss': avg_rgb_loss,
                'train/xyz_loss': avg_xyz_loss,
                'train/learning_rate': optimizer.param_groups[0]['lr']
            }
            
            if val_metrics is not None:
                # Add descriptive metric names based on training flow
                if args.training_flow == "forward":
                    # XYZ -> sRGB -> XYZ
                    log_dict.update({
                        'val/loss': val_metrics['loss'],
                        'val/rgb_loss': val_metrics['rgb_loss'],
                        'val/xyz_loss': val_metrics['xyz_loss'],
                        'val/xyz_to_srgb_psnr': val_metrics['forward_psnr'],
                        'val/xyz_to_srgb_ssim': val_metrics['forward_ssim'],
                        'val/srgb_to_xyz_psnr': val_metrics['reverse_psnr'],
                        'val/srgb_to_xyz_ssim': val_metrics['reverse_ssim']
                    })
                else:
                    # sRGB -> XYZ -> sRGB
                    log_dict.update({
                        'val/loss': val_metrics['loss'],
                        'val/rgb_loss': val_metrics['rgb_loss'],
                        'val/xyz_loss': val_metrics['xyz_loss'],
                        'val/srgb_to_xyz_psnr': val_metrics['forward_psnr'],
                        'val/srgb_to_xyz_ssim': val_metrics['forward_ssim'],
                        'val/xyz_to_srgb_psnr': val_metrics['reverse_psnr'],
                        'val/xyz_to_srgb_ssim': val_metrics['reverse_ssim']
                    })
            
            wandb.log(log_dict)
        
        # Print epoch summary (only on main process)
        if not args.distributed or args.local_rank == 0:
            epoch_duration = time.time() - epoch_time
            summary = f"[Epoch {epoch+1}/{args.epochs}] Train: {avg_loss:.5f} | Time: {epoch_duration:.1f}s"
            
            if val_metrics is not None:
                if args.training_flow == "forward":
                    summary += f" | Val: {val_metrics['loss']:.5f} | XYZ→sRGB: {val_metrics['forward_psnr']:.2f}dB | sRGB→XYZ: {val_metrics['reverse_psnr']:.2f}dB"
                else:
                    summary += f" | Val: {val_metrics['loss']:.5f} | sRGB→XYZ: {val_metrics['forward_psnr']:.2f}dB | XYZ→sRGB: {val_metrics['reverse_psnr']:.2f}dB"
            
            print(summary)
        
        # Synchronize all processes
        if args.distributed:
            dist.barrier()
    
    if not args.distributed or args.local_rank == 0:
        print("[INFO] Training completed!")
    
    # Cleanup wandb (only on main process)
    if args.use_wandb and (not args.distributed or args.local_rank == 0):
        wandb.finish()
    
    # Cleanup DDP
    if args.distributed:
        cleanup_ddp()

if __name__ == '__main__':
    torch.set_num_threads(4)
    main()

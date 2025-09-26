# InvISP XYZ Training

This directory contains the training scripts and dataset classes for training InvISP (Invertible Image Signal Processing) with sRGB-XYZ image pairs.

## Files

- `dataset/XYZ_dataset.py`: Modified dataset class for sRGB-XYZ pairs with InvISP compatibility
- `scripts/train_invisp_xyz.py`: Main training script for InvISP with XYZ data (standalone, no config.py dependency)
- `train_invisp_xyz.sh`: Example bash script for running training

## Dataset Structure

The dataset should be organized as follows:
```
/media/ssd2/users/dykim/dataset/sRGB_XYZ_pairs/
├── trainval.json          # Manifest file with train/val splits
├── a5k/
│   ├── sRGB/             # sRGB images (_srgb.png)
│   └── XYZ/              # XYZ images (_xyz.png)
├── nus/
│   ├── sRGB/
│   └── XYZ/
└── raise/
    ├── sRGB/
    └── XYZ/
```

## Usage

### Basic Training

```bash
# Forward training (XYZ -> sRGB)
python scripts/train_invisp_xyz.py \
    --task "invisp_xyz_forward" \
    --manifest_path "/media/ssd2/users/dykim/dataset/sRGB_XYZ_pairs/trainval.json" \
    --training_flow "forward" \
    --image_size 512 \
    --batch_size 1 \
    --epochs 300

# Backward training (sRGB -> XYZ)
python scripts/train_invisp_xyz.py \
    --task "invisp_xyz_backward" \
    --manifest_path "/media/ssd2/users/dykim/dataset/sRGB_XYZ_pairs/trainval.json" \
    --training_flow "backward" \
    --image_size 512 \
    --batch_size 1 \
    --epochs 300
```

### Multi-GPU Training

```bash
# Single GPU training (default)
python scripts/train_invisp_xyz.py \
    --task "invisp_xyz_single" \
    --gpu_ids "0" \
    --batch_size 2

# Multi-GPU training
python scripts/train_invisp_xyz.py \
    --task "invisp_xyz_multi" \
    --gpu_ids "0,1,2,3" \
    --distributed \
    --batch_size 1  # 1 per GPU = 4 total
```

### Advanced Options

```bash
python scripts/train_invisp_xyz.py \
    --task "invisp_xyz_custom" \
    --manifest_path "/path/to/trainval.json" \
    --dataset_subsets "a5k" "nus" \  # Use specific datasets
    --training_flow "forward" \       # or "backward"
    --image_size 1024 \              # Square image size
    --xyz_norm_mode "d65" \           # XYZ normalization mode
    --batch_size 2 \
    --lr 0.0002 \
    --epochs 500 \
    --rgb_weight 1.0 \
    --xyz_weight 1.0 \
    --bidirectional \                # Use bidirectional loss
    --aug                            # Enable data augmentation
```

## Key Parameters

### Training Flow
- `--training_flow forward`: Train XYZ -> sRGB conversion
- `--training_flow backward`: Train sRGB -> XYZ conversion

### Image Size
- `--image_size 512`: Resize to 512x512 square
- `--image_size 1024`: Resize to 1024x1024 square

### Dataset Subsets
- `--dataset_subsets "all"`: Use all available datasets
- `--dataset_subsets "a5k" "nus"`: Use specific datasets only

### XYZ Normalization
- `--xyz_norm_mode "unit"`: Normalize XYZ to [0,1] range
- `--xyz_norm_mode "d65"`: Normalize XYZ using D65 white point

### Loss Configuration
- `--bidirectional`: Use both forward and backward losses
- `--rgb_weight 1.0`: Weight for RGB loss component
- `--xyz_weight 1.0`: Weight for XYZ loss component

### Multi-GPU Training
- `--gpu_ids "0,1,2,3"`: Specify GPU IDs to use
- `--distributed`: Enable multi-GPU training with DataParallel
- `--batch_size`: Batch size per GPU (total batch size = batch_size × num_gpus)

## Output

The training script will create:
- `./exps/{task_name}/checkpoint/latest.pth`: Latest checkpoint
- `./exps/{task_name}/checkpoint/{epoch:04d}.pth`: Periodic checkpoints
- `./exps/{task_name}/commandline_args.json`: Training configuration

## Notes

1. The dataset class automatically handles train/val splits from the manifest file
2. XYZ images are expected to be uint16 format (camera_white_level normalized * 16383)
3. sRGB images are expected to be 8-bit PNG format
4. The model uses [0,1] range for both input and output tensors
5. JPEG compression simulation is applied during training for robustness

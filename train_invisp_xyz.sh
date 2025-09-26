#!/bin/bash

# Training script for InvISP with sRGB-XYZ pairs
# This script runs from the project root directory
# Supports both single GPU and multi-GPU distributed training

# =================================================================
#                       CONFIGURATION
# =================================================================
# Set paths
MANIFEST_PATH="/ssd2/dataset/sRGB_XYZ_pairs/trainval.json"

# Training parameters
TASK_NAME="invisp_xyz_forward"
OUTPUT_PATH="./exps/"
BATCH_SIZE=3
LEARNING_RATE=0.0001
EPOCHS=300
IMAGE_SIZE=512
LOSS_TYPE="L1"

# Dataset parameters
DATASET_SUBSETS=(a5k raise)  # or "all" for all datasets
TRAINING_FLOW="forward"      # "forward" for XYZ->sRGB, "backward" for sRGB->XYZ
XYZ_NORM_MODE="unit"         # "unit" or "d65"

# JPEG compression parameters
USE_JPEG=false          # Set to true to enable JPEG compression simulation
JPEG_QUALITY=90         # JPEG quality factor (1-100)

# Validation parameters
VAL_FREQ=1              # Validation frequency (every N epochs)
VAL_BATCH_SIZE=1        # Validation batch size
VAL_SAMPLES=5           # Number of validation samples to visualize

# Wandb parameters
USE_WANDB=true         # Set to true to enable wandb logging
WANDB_PROJECT="invisp-xyz"  # Wandb project name
WANDB_ENTITY=""         # Wandb entity name (optional, leave empty for personal account)

# Execution parameters
GPU_IDS="0,1,2,3"      # Specify GPU IDs to use (e.g., "0" or "0,1,2,3")
USE_DISTRIBUTED=true   # Set to true to enable distributed training
# =================================================================

# Common arguments for the python script
# These are used in both single-GPU and distributed mode
COMMON_ARGS=(
    "--task" "$TASK_NAME"
    "--out_path" "$OUTPUT_PATH"
    "--manifest_path" "$MANIFEST_PATH"
    "--dataset_subsets" "${DATASET_SUBSETS[@]}"
    "--training_flow" "$TRAINING_FLOW"
    "--image_size" "$IMAGE_SIZE"
    "--xyz_norm_mode" "$XYZ_NORM_MODE"
    "--batch_size" "$BATCH_SIZE"
    "--lr" "$LEARNING_RATE"
    "--epochs" "$EPOCHS"
    "--loss" "$LOSS_TYPE"
    "--rgb_weight" "1.0"
    "--xyz_weight" "1.0"
    "--bidirectional"
    "--aug"
    "--jpeg_quality" "$JPEG_QUALITY"
    "--val_freq" "$VAL_FREQ"
    "--val_batch_size" "$VAL_BATCH_SIZE"
    "--val_samples" "$VAL_SAMPLES"
    "--wandb_project" "$WANDB_PROJECT"
)

# Add JPEG flag conditionally
if [ "$USE_JPEG" = true ]; then
    COMMON_ARGS+=("--use_jpeg")
fi

# Add wandb flag conditionally
if [ "$USE_WANDB" = true ]; then
    COMMON_ARGS+=("--use_wandb")
    if [ -n "$WANDB_ENTITY" ]; then
        COMMON_ARGS+=("--wandb_entity" "$WANDB_ENTITY")
    fi
fi

# Print configuration info
echo "[INFO] Training configuration:"
echo "  - Task: $TASK_NAME"
echo "  - Training flow: $TRAINING_FLOW"
echo "  - Dataset subsets: ${DATASET_SUBSETS[*]}"
echo "  - JPEG simulation: $USE_JPEG (quality: $JPEG_QUALITY)"
echo "  - Validation frequency: every $VAL_FREQ epochs"
echo "  - Wandb logging: $USE_WANDB (project: $WANDB_PROJECT)"
echo "  - Batch size: $BATCH_SIZE"
echo "  - Learning rate: $LEARNING_RATE"
echo "  - Epochs: $EPOCHS"

# Run training based on the configuration
if [ "$USE_DISTRIBUTED" = true ]; then
    # --- DISTRIBUTED TRAINING (DDP) ---
    NUM_GPUS=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)
    echo "[INFO] Starting distributed training with $NUM_GPUS GPUs: [$GPU_IDS]"
    
    # Use torchrun for DDP. It automatically manages environment variables.
    # CUDA_VISIBLE_DEVICES tells torchrun which GPUs it's allowed to use.
    CUDA_VISIBLE_DEVICES=$GPU_IDS torchrun --nproc_per_node=$NUM_GPUS scripts/train_invisp_xyz.py \
        "${COMMON_ARGS[@]}" \
        --distributed
else
    # --- SINGLE GPU TRAINING ---
    echo "[INFO] Starting single GPU training on GPU: [$GPU_IDS]"
    
    # Use standard python and manually specify the GPU.
    CUDA_VISIBLE_DEVICES=$GPU_IDS python scripts/train_invisp_xyz.py \
        "${COMMON_ARGS[@]}" \
        --gpu_ids "$GPU_IDS"
fi

echo "Training completed!"
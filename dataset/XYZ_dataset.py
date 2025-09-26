"""
Dataset classes and utilities for sRGB-XYZ image pairs.

This module provides dataset classes with support for dataset subset selection,
random cropping, resizing, and various normalization modes.
"""

import json
import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2


def ensure_chw_float_tensor(img: Image.Image) -> torch.Tensor:
    """Convert PIL Image to CHW float tensor."""
    arr = np.array(img, dtype=np.float32)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    t = torch.from_numpy(arr.transpose(2, 0, 1))
    return t


def to_chw_tensor(img: Image.Image, dtype=torch.float32) -> torch.Tensor:
    """Convert PIL Image to CHW tensor with specified dtype."""
    arr = np.array(img, dtype=np.float32)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    t = torch.from_numpy(arr).permute(2, 0, 1)
    return t.to(dtype)


def srgb01_to_m11(t: torch.Tensor) -> torch.Tensor:
    """Convert sRGB from [0,1] to [-1,1] range."""
    return t * 2.0 - 1.0


def m11_to_01(t: torch.Tensor) -> torch.Tensor:
    """Convert from [-1,1] to [0,1] range."""
    return (t.clamp(-1, 1) + 1.0) / 2.0


def to_float01(arr: np.ndarray) -> np.ndarray:
    """Convert array to float32 in [0,1] range."""
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    if arr.dtype == np.uint8:
        return (arr.astype(np.float32) / 255.0).astype(np.float32)
    if arr.dtype == np.uint16:
        return (arr.astype(np.float32) / 65535.0).astype(np.float32)
    return arr.astype(np.float32)


def ensure_multiple_of(x: int, base=8) -> int:
    """Ensure x is a multiple of base."""
    return x - (x % base)


class XYZNorm:
    """XYZ normalization utilities."""
    D65 = torch.tensor([0.95047, 1.0, 1.08883], dtype=torch.float32)
    
    def __init__(self, mode="unit"):
        self.mode = mode
    
    def to_unit(self, xyz: torch.Tensor) -> torch.Tensor:
        if self.mode == "unit": 
            return xyz
        elif self.mode == "d65": 
            return xyz / self.D65.view(3, 1, 1).to(xyz.device, xyz.dtype)
        else: 
            raise ValueError("Unknown xyz_norm mode")
    
    def from_unit(self, xyz_unit: torch.Tensor) -> torch.Tensor:
        if self.mode == "unit": 
            return xyz_unit
        elif self.mode == "d65": 
            return xyz_unit * self.D65.view(3, 1, 1).to(xyz_unit.device, xyz_unit.dtype)
        else: 
            raise ValueError("Unknown xyz_norm mode")


def load_manifest_pairs(
    manifest_path: Path, 
    dataset_subsets: Optional[List[str]] = None,
    is_train: bool = True
) -> Tuple[List[Tuple[Path, Path]], Dict[str, str]]:
    """
    Load image pairs from manifest with optional dataset subset filtering.
    
    Args:
        manifest_path: Path to the manifest JSON file
        dataset_subsets: List of dataset types to include (e.g., ["a5k", "raise"]).
                        If None or ["all"], includes all dataset types.
        is_train: Whether to load training or validation data
    
    Returns:
        Tuple of (pairs, meta) where:
        - pairs: List of (sRGB_path, XYZ_path) tuples
        - meta: Dictionary with metadata (root, srgb_suffix, xyz_suffix)
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    
    # Get train or val items based on is_train flag
    if is_train:
        items = manifest.get("train", [])
    else:
        items = manifest.get("val", [])
    
    meta_root = Path(manifest.get("meta", {}).get("root", "."))
    srgb_suffix = manifest.get("meta", {}).get("srgb_suffix", "_srgb.png")
    xyz_suffix = manifest.get("meta", {}).get("xyz_suffix", "_xyz.png")
    
    # Get available dataset types from manifest
    available_types = set()
    for item in items:
        if "dataset_type" in item:
            available_types.add(item["dataset_type"])
    
    # Determine which dataset types to include
    if dataset_subsets is None or "all" in dataset_subsets:
        selected_types = available_types
    else:
        selected_types = set(dataset_subsets)
        # Validate that all requested types exist
        invalid_types = selected_types - available_types
        if invalid_types:
            raise ValueError(f"Invalid dataset types: {invalid_types}. Available types: {sorted(available_types)}")
    
    pairs: List[Tuple[Path, Path]] = []
    for item in items:
        # Skip items not in selected dataset types
        if "dataset_type" in item and item["dataset_type"] not in selected_types:
            continue
            
        if "srgb" in item and "xyz" in item:
            # Backward compatibility format
            s_path = Path(item["srgb"])
            x_path = Path(item["xyz"])
        else:
            # Standard format
            ds_type = item["dataset_type"]
            base = item["basename"]
            s_path = meta_root / ds_type / "sRGB" / f"{base}{srgb_suffix}"
            x_path = meta_root / ds_type / "XYZ" / f"{base}{xyz_suffix}"
        
        pairs.append((s_path, x_path))
    
    meta = {
        "root": str(meta_root),
        "srgb_suffix": srgb_suffix,
        "xyz_suffix": xyz_suffix,
    }
    
    return pairs, meta


class SRGB2XYZDataset(Dataset):
    """
    Dataset for sRGB to XYZ image pairs with support for dataset subset selection.
    
    This dataset loads sRGB-XYZ image pairs from a manifest file and supports
    various preprocessing options including random cropping, resizing, and normalization.
    """
    
    def __init__(
        self,
        manifest_path: Path,
        dataset_subsets: Optional[List[str]] = None,
        image_size: Optional[Union[int, Tuple[int, int]]] = None,
        xyz_norm_mode: str = "unit",
        crop_size_min: int = 512,
        crop_size_max: int = 1024,
        crop_prob: float = 0.8,
        enable_random_crop: bool = True,
        pairs: Optional[List[Tuple[Path, Path]]] = None,
        training_flow: str = "forward",
        is_train: bool = True
    ):
        """
        Initialize the dataset.
        
        Args:
            manifest_path: Path to the manifest JSON file
            dataset_subsets: List of dataset types to include (e.g., ["a5k", "raise"]).
                           If None or ["all"], includes all dataset types.
            image_size: Target image size (int for square, tuple for (height, width))
            xyz_norm_mode: XYZ normalization mode ("unit" or "d65")
            crop_size_min: Minimum random crop size
            crop_size_max: Maximum random crop size
            crop_prob: Probability of applying random crop
            enable_random_crop: Whether to enable random cropping
            pairs: Pre-computed pairs list (if provided, manifest_path and dataset_subsets are ignored)
            training_flow: "forward" (XYZ->sRGB) or "backward" (sRGB->XYZ)
            is_train: Whether this is training dataset (affects data splitting)
        """
        self.image_size = image_size
        self.crop_size_min, self.crop_size_max, self.crop_prob = crop_size_min, crop_size_max, crop_prob
        self.enable_random_crop = enable_random_crop
        self.xyz_norm = XYZNorm(mode=xyz_norm_mode)
        self.training_flow = training_flow
        self.is_train = is_train
        
        if pairs is not None:
            # Use provided pairs
            self.samples = [(Path(s), Path(x)) for (s, x) in pairs]
        else:
            # Load pairs from manifest with train/val split
            pairs, _ = load_manifest_pairs(manifest_path, dataset_subsets, is_train)
            self.samples = pairs
        
        if not self.samples:
            raise ValueError("No samples found in dataset")
        
        self.resize = None
        if image_size:
            if isinstance(image_size, int):
                # Square resize
                img_size = ensure_multiple_of(image_size, 8)
                self.resize = transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.LANCZOS)
            elif isinstance(image_size, tuple) and len(image_size) == 2:
                # Custom size resize
                h, w = image_size
                h = ensure_multiple_of(h, 8)
                w = ensure_multiple_of(w, 8)
                self.resize = transforms.Resize((h, w), interpolation=transforms.InterpolationMode.LANCZOS)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def _random_crop_pair(self, s_img: Image.Image, x_img: np.ndarray):
        """Apply random crop to image pair."""
        if not self.enable_random_crop or random.random() > self.crop_prob:
            return s_img, x_img
        
        crop_size = ensure_multiple_of(random.randint(self.crop_size_min, self.crop_size_max), 8)
        w, h = s_img.size
        if w < crop_size or h < crop_size:
            return s_img, x_img
        
        left, top = random.randint(0, w - crop_size), random.randint(0, h - crop_size)
        right, bottom = left + crop_size, top + crop_size
        return s_img.crop((left, top, right, bottom)), x_img[top:bottom, left:right]
    
    def __getitem__(self, idx: int):
        """Get a sample from the dataset."""
        spath, xpath = self.samples[idx]
        
        # Load sRGB image
        s_img = Image.open(spath).convert("RGB")
        
        # Load XYZ image
        x_img = cv2.imread(str(xpath), cv2.IMREAD_UNCHANGED)
        if x_img is None:
            raise ValueError(f"Failed to load XYZ image: {xpath}")
        if x_img.ndim == 3:
            x_img = cv2.cvtColor(x_img, cv2.COLOR_BGR2RGB)
        
        # Apply random crop if enabled
        s_img, x_img = self._random_crop_pair(s_img, x_img)
        
        # Apply resize if specified
        if self.resize:
            s_img = self.resize(s_img)
            new_size = (self.resize.size[1], self.resize.size[0])
            x_img = cv2.resize(x_img, new_size, interpolation=cv2.INTER_LANCZOS4)
        
        # Convert to tensors
        s_t = to_chw_tensor(s_img, dtype=torch.float32) / 255.0
        
        if x_img.dtype == np.uint8:
            x_t = torch.from_numpy(x_img.astype(np.float32).transpose(2, 0, 1)) / 255.0
        elif x_img.dtype == np.uint16:
            x_t = torch.from_numpy(x_img.astype(np.float32).transpose(2, 0, 1)) / 65535.0
        else:
            x_t = torch.from_numpy(x_img.astype(np.float32).transpose(2, 0, 1))
        
        # Apply XYZ normalization
        x_t = self.xyz_norm.to_unit(x_t)
        
        # Convert to [0,1] range for InvISP (instead of [-1,1])
        s_t = s_t  # Already in [0,1]
        x_t = x_t  # Already in [0,1] after normalization
        
        # Return based on training_flow
        if self.training_flow == "forward":
            # XYZ -> sRGB: input=XYZ, target=sRGB
            return x_t, s_t, spath.name
        else:
            # sRGB -> XYZ: input=sRGB, target=XYZ  
            return s_t, x_t, spath.name


class ValPairsDataset(Dataset):
    """
    Simple dataset for validation pairs without random cropping.
    
    This is a lightweight dataset class specifically designed for validation
    that loads sRGB-XYZ pairs and returns them in [0,1] range.
    """
    
    def __init__(self, pairs: List[Tuple[Path, Path]]):
        """
        Initialize validation dataset.
        
        Args:
            pairs: List of (sRGB_path, XYZ_path) tuples
        """
        self.pairs = pairs
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int):
        """Get a validation sample."""
        spath, xpath = self.pairs[idx]
        
        # Load sRGB image
        s_img = Image.open(spath).convert("RGB")
        s_t01 = ensure_chw_float_tensor(s_img) / 255.0  # CHW [0,1]
        
        # Load XYZ image
        import imageio.v3 as iio
        x_arr = iio.imread(xpath)
        x01 = to_float01(x_arr)  # HWC [0,1]
        x_t01 = torch.from_numpy(x01.transpose(2, 0, 1))  # CHW
        
        return s_t01, x_t01

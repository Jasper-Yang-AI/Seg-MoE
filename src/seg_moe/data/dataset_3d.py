"""
3D NIfTI Segmentation Dataset for Seg-MoE.

Loads multi-modal MRI volumes as [C, D, H, W] tensors.
Supports:
  - Multi-modal stacking (C=3 for prostate)
  - Percentile clip + z-score intensity normalisation (per-channel)
  - Random crop to spatial_size (training)
  - Pad-if-needed for volumetric consistency
  - MONAI-compatible output dict {'image': ..., 'label': ...}
  - Fallback to plain Dataset (image, mask, meta) matching 2D API

Usage:
    from seg_moe.data.dataset_3d import SegmentationDataset3D, build_3d_transforms
    train_ds = SegmentationDataset3D(train_rows, dataset_cfg, augs_cfg, is_train=True)
    val_ds   = SegmentationDataset3D(val_rows,   dataset_cfg, augs_cfg, is_train=False)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Intensity helpers
# ---------------------------------------------------------------------------

def _percentile_znorm(vol: np.ndarray, lo: float = 0.5, hi: float = 99.5) -> np.ndarray:
    """Per-channel percentile clip + z-score normalisation.

    Args:
        vol: [C, D, H, W] float32
    Returns:
        normalised [C, D, H, W] float32
    """
    out = np.empty_like(vol)
    for c in range(vol.shape[0]):
        v = vol[c]
        p_lo = np.percentile(v, lo)
        p_hi = np.percentile(v, hi)
        v = np.clip(v, p_lo, p_hi)
        mean = v.mean()
        std = v.std() + 1e-8
        out[c] = (v - mean) / std
    return out


# ---------------------------------------------------------------------------
# Label map
# ---------------------------------------------------------------------------

def _apply_label_map(mask: np.ndarray, label_map: Dict[int, int]) -> np.ndarray:
    if not label_map:
        return mask
    mapped = np.empty_like(mask)
    mapped[:] = mask
    for k, v in label_map.items():
        mapped[mask == k] = v
    return mapped


# ---------------------------------------------------------------------------
# MONAI transform builder
# ---------------------------------------------------------------------------

def build_3d_transforms(augs_cfg: Optional[Dict[str, Any]], is_train: bool, spatial_size: Tuple[int, ...]):
    """Build MONAI Compose transform from augs_cfg YAML dict.

    Returns a callable that accepts a dict {'image': [C,D,H,W], 'label': [D,H,W]}
    and returns the augmented dict.
    """
    try:
        import monai.transforms as T
    except ImportError:
        raise ImportError("MONAI required for 3D transforms. pip install monai>=1.3")

    # Always-on transforms
    base = [
        T.EnsureTyped(keys=["image", "label"], dtype=[torch.float32, torch.long]),
        T.EnsureChannelFirstd(keys=["label"], channel_dim="no_channel"),
    ]

    if is_train:
        # Random spatial crop
        crop = [
            T.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=spatial_size,
                pos=2,        # twice more foreground patches
                neg=1,
                num_samples=1,
            )
        ]
        # User augmentations from YAML
        aug_list = (augs_cfg or {}).get("train", [])
        user_aug = _build_user_transforms(aug_list, is_train=True)
        transforms = base + crop + user_aug
    else:
        transforms = base   # Val: just type cast; sliding window does the rest

    return T.Compose(transforms)


def _build_user_transforms(aug_list: List[Dict], is_train: bool):
    """Build MONAI transforms from YAML list."""
    try:
        import monai.transforms as T
    except ImportError:
        return []

    _SKIP = {"RandCropByPosNegLabeld"}   # handled separately
    transforms = []

    _MAP = {
        "RandFlipd":             T.RandFlipd,
        "RandRotate90d":         T.RandRotate90d,
        "RandAffined":           T.RandAffined,
        "RandScaleIntensityd":   T.RandScaleIntensityd,
        "RandShiftIntensityd":   T.RandShiftIntensityd,
        "RandGaussianNoised":    T.RandGaussianNoised,
        "RandGaussianSmoothd":   T.RandGaussianSmoothd,
    }

    for t in aug_list:
        name = t.get("name", "")
        if name in _SKIP or not name:
            continue
        cls = _MAP.get(name)
        if cls is None:
            print(f"[3D augs] Unknown transform '{name}', skipping")
            continue
        kwargs = {k: v for k, v in t.items() if k != "name"}
        # Most dict transforms need keys=['image'] or keys=['image','label']
        if "keys" not in kwargs:
            # Spatial transforms apply to both; intensity apply to image only
            _INTENSITY = {"RandScaleIntensityd", "RandShiftIntensityd",
                          "RandGaussianNoised", "RandGaussianSmoothd"}
            kwargs["keys"] = ["image"] if name in _INTENSITY else ["image", "label"]
            if name not in _INTENSITY and "mode" not in kwargs:
                kwargs["mode"] = ["bilinear", "nearest"]
        try:
            transforms.append(cls(**kwargs))
        except Exception as e:
            print(f"[3D augs] Could not build '{name}': {e}, skipping")

    return transforms


# ---------------------------------------------------------------------------
# Main Dataset
# ---------------------------------------------------------------------------

class SegmentationDataset3D(Dataset):
    """3D segmentation dataset loading NIfTI volumes.

    Each sample dict (from splits JSONL) must have:
        id             : str
        image_paths    : List[str]   (one per modality)
        mask_path      : str
        patient_id     : str
        split          : str

    Returns (image_tensor, mask_tensor, meta):
        image_tensor : float32 [C, D, H, W]
        mask_tensor  : int64   [D, H, W]
        meta         : dict
    """

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        dataset_cfg: Dict[str, Any],
        augs_cfg: Optional[Dict[str, Any]] = None,
        *,
        is_train: bool = False,
        limit: Optional[int] = None,
    ) -> None:
        self.samples = samples[: (limit or len(samples))]
        self.dataset_cfg = dataset_cfg
        self.num_classes = int(dataset_cfg["task"]["num_classes"])
        self.image_channels = int(dataset_cfg["input"].get("image_channels", 1))
        self.label_map = {int(k): int(v) for k, v in dataset_cfg["task"].get("label_map", {}).items()}
        self.is_train = is_train

        sz = dataset_cfg["input"]["spatial_size"]
        self.spatial_size = tuple(int(s) for s in sz)    # (H, W, D)

        intens = dataset_cfg.get("intensity", {})
        self.lo = float((intens.get("percentile_clip") or [0.5, 99.5])[0])
        self.hi = float((intens.get("percentile_clip") or [0.5, 99.5])[1])

        self.transform = build_3d_transforms(augs_cfg, is_train, self.spatial_size)

    def __len__(self) -> int:
        return len(self.samples)

    # ---- I/O ----

    def _load_volume(self, paths: List[str]) -> np.ndarray:
        """Load and stack modality volumes → [C, D, H, W] float32."""
        try:
            import SimpleITK as sitk
        except ImportError:
            raise ImportError("SimpleITK required: pip install SimpleITK")

        channels = []
        for p in paths:
            img = sitk.ReadImage(str(p))
            arr = sitk.GetArrayFromImage(img).astype(np.float32)  # [D, H, W]
            channels.append(arr)

        # If only 1 modality but expecting C channels, replicate
        while len(channels) < self.image_channels:
            channels.append(channels[-1].copy())

        vol = np.stack(channels[:self.image_channels], axis=0)   # [C, D, H, W]
        return vol

    def _load_mask(self, path: str) -> np.ndarray:
        try:
            import SimpleITK as sitk
        except ImportError:
            raise ImportError("SimpleITK required: pip install SimpleITK")
        img = sitk.ReadImage(str(path))
        arr = sitk.GetArrayFromImage(img).astype(np.int64)       # [D, H, W]
        arr = _apply_label_map(arr, self.label_map)
        return arr

    # ---- __getitem__ ----

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img_paths = s.get("image_paths") or [s["image_path"]]
        mask_path = s["mask_path"]

        vol = self._load_volume(img_paths)                      # [C, D, H, W]
        vol = _percentile_znorm(vol, self.lo, self.hi)
        mask = self._load_mask(mask_path)                       # [D, H, W]

        data_dict = {"image": vol, "label": mask}
        data_dict = self.transform(data_dict)

        # RandCropByPosNegLabeld returns a list of 1 sample
        if isinstance(data_dict, list):
            data_dict = data_dict[0]

        img_t = data_dict["image"]                  # [C, D, H, W] tensor
        # label: after EnsureChannelFirst → [1, D, H, W]; squeeze to [D, H, W]
        lbl_t = data_dict["label"]
        if isinstance(lbl_t, torch.Tensor):
            if lbl_t.ndim == 4 and lbl_t.shape[0] == 1:
                lbl_t = lbl_t.squeeze(0)
            lbl_t = lbl_t.long()
        else:
            lbl_t = torch.from_numpy(np.asarray(lbl_t)).long()
            if lbl_t.ndim == 4 and lbl_t.shape[0] == 1:
                lbl_t = lbl_t.squeeze(0)

        if not isinstance(img_t, torch.Tensor):
            img_t = torch.from_numpy(np.asarray(img_t)).float()

        meta = {
            "id":          s.get("id", ""),
            "patient_id":  s.get("patient_id", ""),
            "split":       s.get("split", ""),
            "image_paths": img_paths,
            "mask_path":   mask_path,
        }
        return img_t, lbl_t, meta

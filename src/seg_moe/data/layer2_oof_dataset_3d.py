"""
3D Layer2 OOF Dataset — feeds Layer2 3D experts.

Input to Layer2:
    x = concat([volume[C,D,H,W], oof_probs[K*M,D,H,W],
                entropy[1,D,H,W], disagreement[M,D,H,W]])

OOF logits are loaded from manifest produced by
    scripts/inference/generate_layer1_oof_3d.py

Shape conventions:
    vol      : [C, D, H, W]
    probs    : [K, M, D, H, W]   (derived from npz key 'logits')
    entropy  : [1, D, H, W]      Shannon entropy over expert-mean probs
    disagree : [M, D, H, W]      std across expert probs (per class)
    x_cat    : [C + K*M + 1 + M, D, H, W]
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from seg_moe.data.dataset_3d import (
    SegmentationDataset3D,
    _apply_label_map,
    _percentile_znorm,
    build_3d_transforms,
)
from seg_moe.data.oof import OOFRecord, load_oof_manifest
from seg_moe.utils.spatial import parse_3d_size


# ---------------------------------------------------------------------------
# Uncertainty helpers
# ---------------------------------------------------------------------------

def _entropy_3d(probs_mean: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Shannon entropy of mean expert probability map.

    Args:
        probs_mean: [M, D, H, W]  (already softmaxed, sum-1 over M)
    Returns:
        entropy: [1, D, H, W]
    """
    p = np.clip(probs_mean, eps, 1.0)
    h = -(p * np.log(p)).sum(axis=0, keepdims=True)   # [1, D, H, W]
    # Normalise to [0, 1] by dividing by log(M)
    h_max = np.log(probs_mean.shape[0])
    return (h / (h_max + eps)).astype(np.float32)


def _disagreement_3d(probs: np.ndarray) -> np.ndarray:
    """Per-class expert disagreement (std across experts).

    Args:
        probs: [K, M, D, H, W]
    Returns:
        disagree: [M, D, H, W]
    """
    return probs.std(axis=0).astype(np.float32)  # [M, D, H, W]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class Layer2OOFDataset3D(Dataset):
    """3D Layer2 dataset that concatenates volume with Layer1 OOF probs.

    Each __getitem__ returns (image [C+K*M+M+1, D, H, W], mask [D, H, W], meta).
    """

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        dataset_cfg: Dict[str, Any],
        oof_manifest_path: str | Path,
        *,
        expected_num_experts: Optional[int] = None,
        augs_cfg: Optional[Dict[str, Any]] = None,
        is_train: bool = False,
        limit: Optional[int] = None,
        add_uncertainty: bool = True,
    ) -> None:
        self.samples = samples[: (limit or len(samples))]
        self.dataset_cfg = dataset_cfg
        self.num_classes = int(dataset_cfg["task"]["num_classes"])
        self.image_channels = int(dataset_cfg["input"].get("image_channels", 1))
        self.label_map = {int(k): int(v) for k, v in dataset_cfg["task"].get("label_map", {}).items()}
        self.expected_num_experts = int(expected_num_experts) if expected_num_experts else None
        self.add_uncertainty = add_uncertainty
        self.is_train = is_train

        sz = dataset_cfg["input"]["spatial_size"]
        self.spatial_size = parse_3d_size(sz)

        intens = dataset_cfg.get("intensity", {})
        self.lo = float((intens.get("percentile_clip") or [0.5, 99.5])[0])
        self.hi = float((intens.get("percentile_clip") or [0.5, 99.5])[1])

        self.oof_map = load_oof_manifest(oof_manifest_path)

        # Build transform without crop (we do it manually after concat)
        self.transform = build_3d_transforms(None, is_train=False, spatial_size=self.spatial_size)
        self._train_crop = is_train
        self._augs_cfg = augs_cfg

        # Lazy MONAI import for random crop
        self._crop_fn = None

    def _get_crop_fn(self):
        if self._crop_fn is not None:
            return self._crop_fn
        try:
            import monai.transforms as T
            self._crop_fn = T.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=self.spatial_size,
                pos=2, neg=1, num_samples=1,
            )
        except ImportError:
            self._crop_fn = None
        return self._crop_fn

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        sid = str(s["id"])

        # ---- Load OOF logits/probs ----
        if sid not in self.oof_map:
            raise KeyError(f"Sample '{sid}' not found in OOF manifest. "
                           "Run generate_layer1_oof_3d.py first.")
        oof_rec: OOFRecord = self.oof_map[sid]
        oof_data = np.load(str(oof_rec.prob_path))
        if "logits" in oof_data:
            logits = oof_data["logits"].astype(np.float32)    # [K, M, D, H, W]
            logits = logits - logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            probs = exp_logits / (exp_logits.sum(axis=1, keepdims=True) + 1e-8)
        elif "probs" in oof_data:
            probs = oof_data["probs"].astype(np.float32)      # legacy [K, M, D, H, W]
        else:
            raise KeyError(f"OOF cache missing 'logits'/'probs' for sample '{sid}': {oof_rec.prob_path}")
        K, M, D, H, W = probs.shape
        if self.expected_num_experts and K != self.expected_num_experts:
            raise ValueError(f"Expected {self.expected_num_experts} experts, got {K} for {sid}")

        # ---- Load volume ----
        img_paths = s.get("image_paths") or [s["image_path"]]
        try:
            import SimpleITK as sitk
        except ImportError:
            raise ImportError("SimpleITK required: pip install SimpleITK")

        channels = []
        for p in img_paths[:self.image_channels]:
            arr = sitk.GetArrayFromImage(sitk.ReadImage(str(p))).astype(np.float32)
            channels.append(arr)
        while len(channels) < self.image_channels:
            channels.append(channels[-1].copy())
        vol = np.stack(channels, axis=0)                # [C, D, H, W]
        vol = _percentile_znorm(vol, self.lo, self.hi)

        # ---- Load mask ----
        mask_arr = sitk.GetArrayFromImage(sitk.ReadImage(str(s["mask_path"]))).astype(np.int64)
        mask_arr = _apply_label_map(mask_arr, self.label_map)

        # ---- Build uncertainty channels ----
        probs_mean = probs.mean(axis=0)                 # [M, D, H, W]
        if self.add_uncertainty:
            ent = _entropy_3d(probs_mean)               # [1, D, H, W]
            dis = _disagreement_3d(probs)               # [M, D, H, W]
            extra = np.concatenate([ent, dis], axis=0)  # [1+M, D, H, W]
        else:
            extra = np.zeros((0, D, H, W), dtype=np.float32)

        probs_flat = probs.reshape(K * M, D, H, W)      # [K*M, D, H, W]
        if extra.shape[0] > 0:
            cat = np.concatenate([vol, probs_flat, extra], axis=0)
        else:
            cat = np.concatenate([vol, probs_flat], axis=0)

        # ---- Random crop (training) ----
        cat_t = torch.from_numpy(cat).float()           # [C_total, D, H, W]
        mask_t = torch.from_numpy(mask_arr).long()

        if self._train_crop:
            crop_fn = self._get_crop_fn()
            if crop_fn is not None:
                try:
                    import monai.transforms as T
                    label_4d = mask_t.unsqueeze(0)
                    result = crop_fn({"image": cat_t, "label": label_4d})
                    if isinstance(result, list):
                        result = result[0]
                    cat_t = result["image"]
                    mask_t = result["label"].squeeze(0).long()
                except Exception:
                    pass  # fall through to full volume if crop fails

        meta = {
            "id":         sid,
            "patient_id": s.get("patient_id", ""),
            "split":      s.get("split", ""),
        }
        return cat_t, mask_t, meta

    @property
    def in_channels(self) -> int:
        K = self.expected_num_experts or 3
        M = self.num_classes
        C = self.image_channels
        unc = (1 + M) if self.add_uncertainty else 0
        return C + K * M + unc

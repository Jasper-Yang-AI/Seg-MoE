"""
3D Gating Patch Dataset — loads Layer2 OOF logits [K, M, D, H, W] and GT masks,
splits into 3D patches, returns (logit_flat, logits_struct, mask_patch).

Mirror of gating_patch_dataset.py but fully 3D.

Return shapes per __getitem__:
    logit_flat : [K*M, pD, pH, pW]   — gating network input
    logits     : [K,  M, pD, pH, pW] — weighted fusion  
    mask       : [pD, pH, pW]         — GT patch  (int64)
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from seg_moe.data.oof import load_oof_manifest


# ---------------------------------------------------------------------------
# 3D patch position helpers
# ---------------------------------------------------------------------------

def compute_patch_positions_3d(
    spatial_shape: Tuple[int, int, int],
    patch_size: Tuple[int, int, int],
    stride: Tuple[int, int, int],
) -> List[Tuple[int, int, int]]:
    """Return (d0, h0, w0) top-left positions (shifted inward at borders)."""
    positions: list[tuple[int, int, int]] = []
    D, H, W = spatial_shape
    pd, ph, pw = patch_size
    sd, sh, sw = stride

    def _axis_pos(size: int, ps: int, s: int) -> List[int]:
        pts = list(range(0, size - ps + 1, s))
        if not pts or pts[-1] + ps < size:
            pts.append(max(0, size - ps))
        return pts

    for d in _axis_pos(D, pd, sd):
        for h in _axis_pos(H, ph, sh):
            for w in _axis_pos(W, pw, sw):
                positions.append((d, h, w))

    return list(dict.fromkeys(positions))


def _extract_3d_patch(
    x: np.ndarray,
    pos: Tuple[int, int, int],
    patch_size: Tuple[int, int, int],
) -> np.ndarray:
    """x: [..., D, H, W] → patch at pos."""
    d0, h0, w0 = pos
    pd, ph, pw = patch_size
    return x[..., d0:d0+pd, h0:h0+ph, w0:w0+pw]


# ---------------------------------------------------------------------------
# Main dataset
# ---------------------------------------------------------------------------

class GatingPatchDataset3D(Dataset):
    """3D patch gating dataset.

    Loads Layer2 OOF logits [K, M, D, H, W] per volume (saved by
    generate_layer2_oof_3d.py --save-logits) and ground-truth NIfTI masks,
    partitions them into 3D patches for gating network training.
    """

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        dataset_cfg: Dict[str, Any],
        oof_manifest_path: str | Path,
        *,
        expected_num_experts: int = 3,
        patch_size: Tuple[int, int, int] = (32, 32, 16),
        stride: Tuple[int, int, int] = (16, 16, 8),
        is_train: bool = True,
        foreground_oversample_ratio: float = 0.5,
        limit: Optional[int] = None,
        cache_in_memory: bool = False,
    ) -> None:
        self.dataset_cfg = dataset_cfg
        self.patch_size = tuple(int(p) for p in patch_size)
        self.stride = tuple(int(s) for s in stride)
        self.is_train = is_train
        self.fg_ratio = float(foreground_oversample_ratio) if is_train else 0.0
        self.K = expected_num_experts
        self.M = int(dataset_cfg["task"]["num_classes"])
        self.label_map = {int(k): int(v) for k, v in dataset_cfg["task"].get("label_map", {}).items()}
        self.cache_in_memory = cache_in_memory
        self._mem_cache: Dict[str, Any] = {}

        # Filter samples to those in manifest
        self.oof_map = load_oof_manifest(oof_manifest_path)
        self.samples = [s for s in samples if str(s["id"]) in self.oof_map]
        if limit and limit < len(self.samples):
            self.samples = self.samples[:limit]

        # Build flat patch index
        self._patch_index: List[Tuple[int, Tuple[int, int, int]]] = []   # (sample_idx, (d,h,w))
        self._fg_index: List[int] = []     # indices into _patch_index with foreground

        # We need to peek at a volume to get spatial shape for patch positions
        # Defer until first use to avoid loading all volumes at init
        self._spatial_shape: Optional[Tuple[int, int, int]] = None
        self._positions: Optional[List[Tuple[int, int, int]]] = None

    def _init_index(self) -> None:
        if self._patch_index:
            return

        for si, s in enumerate(self.samples):
            sid = str(s["id"])
            oof_rec = self.oof_map[sid]
            data = np.load(str(oof_rec.prob_path))
            key = "logits" if "logits" in data else "probs"
            vol = data[key]                     # [K, M, D, H, W] or [K, M, H, W]
            if vol.ndim == 4:                   # old 2D-style format
                _, _, H, W = vol.shape
                spatial = (1, H, W)
            else:
                _, _, D, H, W = vol.shape
                spatial = (D, H, W)

            positions = compute_patch_positions_3d(spatial, self.patch_size, self.stride)
            for pos in positions:
                pidx = len(self._patch_index)
                self._patch_index.append((si, pos))

                # Check foreground
                if self.fg_ratio > 0 and "mask_path" in s:
                    pass   # lazy FG detection done at runtime
                else:
                    self._fg_index.append(pidx)

        if not self._fg_index:
            self._fg_index = list(range(len(self._patch_index)))

    def __len__(self) -> int:
        self._init_index()
        return len(self._patch_index)

    def _load_sample(self, si: int):
        """Load (logits [K, M, D, H, W], mask [D, H, W]) for sample si."""
        s = self.samples[si]
        sid = str(s["id"])

        if self.cache_in_memory and sid in self._mem_cache:
            return self._mem_cache[sid]

        oof_rec = self.oof_map[sid]
        data = np.load(str(oof_rec.prob_path))

        if "logits" in data:
            logits = data["logits"].astype(np.float32)      # [K, M, D, H, W]
        else:
            probs = data["probs"].astype(np.float32)
            eps = 1e-6
            probs = np.clip(probs, eps, 1 - eps)
            logits = np.log(probs / (1 - probs))            # log-odds approximation

        # Load mask
        mask_path = s.get("mask_path")
        if mask_path and Path(mask_path).exists():
            try:
                import SimpleITK as sitk
                mask = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path))).astype(np.int64)
                if self.label_map:
                    from seg_moe.data.dataset_3d import _apply_label_map
                    mask = _apply_label_map(mask, self.label_map)
            except Exception:
                mask = np.zeros(logits.shape[2:], dtype=np.int64)
        else:
            mask = np.zeros(logits.shape[2:], dtype=np.int64)

        # Handle missing D dim (2D → expand)
        if logits.ndim == 4:
            logits = logits[:, :, None]
            mask = mask[None] if mask.ndim == 2 else mask

        result = (logits, mask)
        if self.cache_in_memory:
            self._mem_cache[sid] = result
        return result

    def __getitem__(self, idx: int):
        self._init_index()

        # Foreground oversampling
        if self.fg_ratio > 0 and random.random() < self.fg_ratio and self._fg_index:
            idx = random.choice(self._fg_index)

        si, pos = self._patch_index[idx]
        logits, mask = self._load_sample(si)             # [K,M,D,H,W], [D,H,W]

        logit_patch = _extract_3d_patch(logits, pos, self.patch_size)   # [K, M, pd, ph, pw]
        mask_patch = _extract_3d_patch(mask, pos, self.patch_size)       # [pd, ph, pw]

        logit_flat = logit_patch.reshape(self.K * self.M, *self.patch_size)   # [K*M, pd, ph, pw]
        logits_struct = logit_patch                                             # [K,  M, pd, ph, pw]

        return (
            torch.from_numpy(logit_flat).float(),
            torch.from_numpy(np.ascontiguousarray(logits_struct)).float(),
            torch.from_numpy(mask_patch).long(),
        )

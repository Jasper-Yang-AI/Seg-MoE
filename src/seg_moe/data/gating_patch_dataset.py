"""
Gating Patch Dataset: 加载缓存的 Layer2 OOF logits [K,M,H,W] 和 GT mask,
按 patch 切分, 返回 (logit_flat, logits_structured, mask_patch).

正确流程:
  Layer1 train → L1 OOF → Layer2 train → L2 OOF (--save-logits) → Gating train → Eval
  门控网络输入 = Layer2 专家的 OOF logits (而非 probs, 非 Layer1!)

若 npz 中无 "logits" 字段 (旧版 OOF), 自动 log-odds 近似: log(p/(1-p))

Return Shape (每条样本):
  logit_flat : [K*M, pH, pW]   — 门控网络输入
  logits     : [K, M, pH, pW]  — 加权融合用 (logits 域)
  mask       : [pH, pW]        — GT mask patch (int64)
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from seg_moe.data.oof import load_oof_manifest, get_oof_prob_path
from seg_moe.utils.patches import compute_patch_positions


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class GatingPatchDataset(Dataset):
    """Dataset for patch-level gating network training.

    Loads pre-cached **Layer2** expert logit maps ``[K, M, H, W]`` and GT masks,
    splits into patches, returns per-patch samples.

    OOF must be generated with:
        generate_layer2_oof.py --save-logits

    If only probs are present in the npz (legacy), falls back to log-odds approximation.

    Parameters
    ----------
    foreground_oversample_ratio : float
        Probability of forcefully drawing a foreground-containing patch
        during training (0 = disabled, uniform sampling).

    Returns (per __getitem__)
    -------
    logit_flat : [K*M, pH, pW]   — gating network input
    logits     : [K, M, pH, pW]  — for weighted fusion (logits domain)
    mask       : [pH, pW]        — ground-truth mask patch (int64)
    """

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        dataset_cfg: Dict[str, Any],
        oof_manifest_path: str | Path,
        *,
        expected_num_experts: int = 3,
        patch_size: int = 64,
        stride: int = 32,
        is_train: bool = True,
        foreground_oversample_ratio: float = 0.0,
        limit: int | None = None,
        cache_in_memory: bool = True,
    ) -> None:
        self.dataset_cfg = dataset_cfg
        self.patch_size = patch_size
        self.stride = stride
        self.is_train = is_train
        self.fg_ratio = float(foreground_oversample_ratio) if is_train else 0.0
        self.K = expected_num_experts
        self.M = int(dataset_cfg["task"]["num_classes"])
        self.label_map = {
            int(k): int(v)
            for k, v in dataset_cfg["task"].get("label_map", {}).items()
        }
        self.cache_in_memory = cache_in_memory

        # Load OOF manifest
        self.oof_map = load_oof_manifest(oof_manifest_path)

        # Filter to samples that have OOF
        self.samples: list[dict] = []
        for s in samples:
            sid = str(s["id"])
            if sid in self.oof_map:
                self.samples.append(s)
        if limit and limit < len(self.samples):
            self.samples = self.samples[:limit]

        # Compute patch positions
        img_size = dataset_cfg["input"]["image_size"]
        H, W = int(img_size[0]), int(img_size[1])
        self.H, self.W = H, W
        self.positions = compute_patch_positions(H, W, patch_size, stride)
        self.patches_per_image = len(self.positions)

        # In-memory cache
        self._logits_cache: dict[str, np.ndarray] = {}  # [K,M,H,W]
        self._mask_cache: dict[str, np.ndarray] = {}
        if cache_in_memory:
            self._preload()

        # Foreground-patch index for oversampling
        self._fg_patch_index: Optional[Dict[int, List[int]]] = None
        if self.fg_ratio > 0:
            self._build_fg_index()

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------

    def _preload(self) -> None:
        for s in self.samples:
            sid = str(s["id"])
            self._logits_cache[sid] = self._load_logits(sid)
            self._mask_cache[sid] = self._load_mask(s)

    def _load_logits(self, sample_id: str) -> np.ndarray:
        """Return logits [K,M,H,W]. Falls back to log-odds if only probs available."""
        prob_path = get_oof_prob_path(self.oof_map, sample_id)
        data = np.load(prob_path)
        if "logits" in data:
            return data["logits"].astype(np.float32)
        # Legacy fallback: compute log-odds from probs
        probs = data["probs"].astype(np.float32)
        eps = 1e-6
        p_clamp = np.clip(probs, eps, 1 - eps)
        return np.log(p_clamp / (1 - p_clamp)).astype(np.float32)

    def _load_mask(self, sample: dict) -> np.ndarray:
        mask_path = sample.get("mask_path") or sample.get("mask")
        m = Image.open(mask_path)
        if m.mode != "L":
            m = m.convert("L")
        arr = np.array(m, dtype=np.uint8)
        if self.label_map:
            mapped = arr.copy()
            for k, v in self.label_map.items():
                mapped[arr == k] = v
            arr = mapped
        return arr.astype(np.int64)

    # ------------------------------------------------------------------
    # Foreground-oversample index
    # ------------------------------------------------------------------

    def _build_fg_index(self) -> None:
        self._fg_patch_index = {}
        ps = self.patch_size
        for img_idx, s in enumerate(self.samples):
            sid = str(s["id"])
            mask = self._mask_cache[sid] if self.cache_in_memory else self._load_mask(s)
            self._fg_patch_index[img_idx] = [
                patch_idx
                for patch_idx, (y, x) in enumerate(self.positions)
                if (mask[y : y + ps, x : x + ps] > 0).any()
            ]

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples) * self.patches_per_image

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img_idx = idx // self.patches_per_image
        patch_idx = idx % self.patches_per_image

        # Foreground oversampling
        if self.fg_ratio > 0 and self._fg_patch_index is not None:
            fg_list = self._fg_patch_index.get(img_idx, [])
            if fg_list and random.random() < self.fg_ratio:
                patch_idx = random.choice(fg_list)

        sample = self.samples[img_idx]
        sid = str(sample["id"])

        if self.cache_in_memory:
            logits = self._logits_cache[sid]
            mask = self._mask_cache[sid]
        else:
            logits = self._load_logits(sid)
            mask = self._load_mask(sample)

        y, x = self.positions[patch_idx]
        ps = self.patch_size

        logit_patch = logits[:, :, y : y + ps, x : x + ps].copy()  # [K,M,pH,pW]
        mask_patch = mask[y : y + ps, x : x + ps].copy()            # [pH,pW]

        # Training augmentation: random H/V flip
        if self.is_train:
            if np.random.random() > 0.5:
                logit_patch = np.flip(logit_patch, axis=-1).copy()
                mask_patch = np.flip(mask_patch, axis=-1).copy()
            if np.random.random() > 0.5:
                logit_patch = np.flip(logit_patch, axis=-2).copy()
                mask_patch = np.flip(mask_patch, axis=-2).copy()

        K, M, pH, pW = logit_patch.shape
        logit_flat = logit_patch.reshape(K * M, pH, pW)              # [K*M, pH,pW]

        return (
            torch.from_numpy(logit_flat),    # [K*M, pH, pW]  — gate input
            torch.from_numpy(logit_patch),   # [K, M, pH, pW] — for fuse_logits
            torch.from_numpy(mask_patch),    # [pH, pW]
        )

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from seg_moe.data.oof import load_oof_manifest, get_oof_prob_path
from seg_moe.utils.patches import compute_patch_positions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_npz_field(path: Path, field: str, fallback_shape: tuple | None = None) -> np.ndarray:
    """Load field from .npz; return zero array of fallback_shape if field absent."""
    data = np.load(path)
    if field in data:
        return data[field].astype(np.float32)
    if fallback_shape is not None:
        return np.zeros(fallback_shape, dtype=np.float32)
    raise KeyError(
        f"Field '{field}' not found in {path}. "
        "Re-run generate_layer2_oof.py with --save-logits to include logits."
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class GatingPatchDataset(Dataset):
    """Dataset for patch-level gating network training.

    Loads pre-cached **Layer2** expert probability maps (and optionally logits)
    ``[K, M, H, W]`` and GT masks, splits into patches, returns per-patch samples.

    Note: oof_manifest_path should point to the **Layer2** OOF manifest
    (oof_manifest_layer2.jsonl), NOT the Layer1 one.

    Parameters
    ----------
    input_domain : str
        One of "probs" | "logits" | "probs+logits".
        Controls what is concatenated to form the gate network input.
    foreground_oversample_ratio : float
        Probability of forcefully drawing a foreground-containing patch
        during training (0 = disabled, uniform sampling).

    Returns (per __getitem__)
    -------
    input_flat : [C_in, pH, pW]    — gating network input
    probs      : [K, M, pH, pW]    — for weighted fusion (probs domain)
    logits     : [K, M, pH, pW]    — for weighted fusion (logits domain)
    mask       : [pH, pW]          — ground-truth mask patch (int64)
    """

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        dataset_cfg: Dict[str, Any],
        oof_manifest_path: str | Path,
        *,
        expected_num_experts: int = 3,
        patch_size: int = 64,
        stride: int = 32,
        is_train: bool = True,
        input_domain: str = "probs",          # "probs" | "logits" | "probs+logits"
        foreground_oversample_ratio: float = 0.0,
        limit: int | None = None,
        cache_in_memory: bool = True,
    ) -> None:
        assert input_domain in {"probs", "logits", "probs+logits"}, (
            f"input_domain must be 'probs', 'logits', or 'probs+logits', got '{input_domain}'"
        )
        self.dataset_cfg = dataset_cfg
        self.patch_size = patch_size
        self.stride = stride
        self.is_train = is_train
        self.input_domain = input_domain
        self.fg_ratio = float(foreground_oversample_ratio) if is_train else 0.0
        self.K = expected_num_experts
        self.M = int(dataset_cfg["task"]["num_classes"])
        self.label_map = {
            int(k): int(v)
            for k, v in dataset_cfg["task"].get("label_map", {}).items()
        }
        self.cache_in_memory = cache_in_memory

        # Load OOF manifest
        self.oof_map = load_oof_manifest(oof_manifest_path)

        # Filter to samples that have OOF probs
        self.samples: list[dict] = []
        for s in samples:
            sid = str(s["id"])
            if sid in self.oof_map:
                self.samples.append(s)
        if limit and limit < len(self.samples):
            self.samples = self.samples[:limit]

        # Compute patch positions
        img_size = dataset_cfg["input"]["image_size"]
        H, W = int(img_size[0]), int(img_size[1])
        self.H, self.W = H, W
        self.positions = compute_patch_positions(H, W, patch_size, stride)
        self.patches_per_image = len(self.positions)

        # In-memory cache
        self._probs_cache: dict[str, np.ndarray] = {}   # [K,M,H,W]
        self._logits_cache: dict[str, np.ndarray] = {}  # [K,M,H,W]
        self._mask_cache: dict[str, np.ndarray] = {}
        if cache_in_memory:
            self._preload()

        # Build foreground patch index for oversampling
        # _fg_patch_index[img_idx] = list of patch_idx containing fg pixels
        self._fg_patch_index: Optional[Dict[int, List[int]]] = None
        if self.fg_ratio > 0:
            self._build_fg_index()

    # ------------------------------------------------------------------
    # Cache / loading
    # ------------------------------------------------------------------

    def _preload(self) -> None:
        for s in self.samples:
            sid = str(s["id"])
            probs, logits = self._load_probs_logits(sid)
            self._probs_cache[sid] = probs
            self._logits_cache[sid] = logits
            self._mask_cache[sid] = self._load_mask(s)

    def _load_probs_logits(self, sample_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """Return (probs [K,M,H,W], logits [K,M,H,W])."""
        prob_path = get_oof_prob_path(self.oof_map, sample_id)
        probs = _load_npz_field(prob_path, "probs").astype(np.float32)  # [K,M,H,W]

        needs_logits = self.input_domain in {"logits", "probs+logits"}
        if needs_logits:
            # Try to load saved logits; fall back to log-odds approximation
            data = np.load(prob_path)
            if "logits" in data:
                logits = data["logits"].astype(np.float32)
            else:
                eps = 1e-6
                p_clamp = np.clip(probs, eps, 1 - eps)
                logits = np.log(p_clamp / (1 - p_clamp)).astype(np.float32)
        else:
            logits = np.zeros_like(probs)
        return probs, logits

    def _load_mask(self, sample: dict) -> np.ndarray:
        mask_path = sample.get("mask_path") or sample.get("mask")
        m = Image.open(mask_path)
        if m.mode != "L":
            m = m.convert("L")
        arr = np.array(m, dtype=np.uint8)
        if self.label_map:
            mapped = arr.copy()
            for k, v in self.label_map.items():
                mapped[arr == k] = v
            arr = mapped
        return arr.astype(np.int64)

    # ------------------------------------------------------------------
    # Foreground-oversample index
    # ------------------------------------------------------------------

    def _build_fg_index(self) -> None:
        self._fg_patch_index = {}
        ps = self.patch_size
        for img_idx, s in enumerate(self.samples):
            sid = str(s["id"])
            if self.cache_in_memory:
                mask = self._mask_cache[sid]
            else:
                mask = self._load_mask(s)
            fg_patches = [
                patch_idx
                for patch_idx, (y, x) in enumerate(self.positions)
                if (mask[y : y + ps, x : x + ps] > 0).any()
            ]
            self._fg_patch_index[img_idx] = fg_patches

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples) * self.patches_per_image

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        img_idx = idx // self.patches_per_image
        patch_idx = idx % self.patches_per_image

        # Foreground oversampling: with probability fg_ratio pick a fg patch
        if self.fg_ratio > 0 and self._fg_patch_index is not None:
            fg_list = self._fg_patch_index.get(img_idx, [])
            if fg_list and random.random() < self.fg_ratio:
                patch_idx = random.choice(fg_list)

        sample = self.samples[img_idx]
        sid = str(sample["id"])

        # Load / retrieve from cache
        if self.cache_in_memory:
            probs = self._probs_cache[sid]
            logits = self._logits_cache[sid]
            mask = self._mask_cache[sid]
        else:
            probs, logits = self._load_probs_logits(sid)
            mask = self._load_mask(sample)

        # Extract patch
        y, x = self.positions[patch_idx]
        ps = self.patch_size

        prob_patch = probs[:, :, y : y + ps, x : x + ps].copy()    # [K,M,pH,pW]
        logit_patch = logits[:, :, y : y + ps, x : x + ps].copy()  # [K,M,pH,pW]
        mask_patch = mask[y : y + ps, x : x + ps].copy()            # [pH,pW]

        # Training augmentation: random H/V flip
        if self.is_train:
            if np.random.random() > 0.5:
                prob_patch = np.flip(prob_patch, axis=-1).copy()
                logit_patch = np.flip(logit_patch, axis=-1).copy()
                mask_patch = np.flip(mask_patch, axis=-1).copy()
            if np.random.random() > 0.5:
                prob_patch = np.flip(prob_patch, axis=-2).copy()
                logit_patch = np.flip(logit_patch, axis=-2).copy()
                mask_patch = np.flip(mask_patch, axis=-2).copy()

        # Build gate network input according to input_domain
        K, M, pH, pW = prob_patch.shape
        if self.input_domain == "probs":
            input_flat = prob_patch.reshape(K * M, pH, pW)          # [K*M, pH,pW]
        elif self.input_domain == "logits":
            input_flat = logit_patch.reshape(K * M, pH, pW)         # [K*M, pH,pW]
        else:  # "probs+logits"
            input_flat = np.concatenate(
                [prob_patch.reshape(K * M, pH, pW),
                 logit_patch.reshape(K * M, pH, pW)],
                axis=0,                                              # [K*2M, pH,pW]
            )

        return (
            torch.from_numpy(input_flat),                # [C_in, pH, pW]
            torch.from_numpy(prob_patch),                # [K, M, pH, pW]
            torch.from_numpy(logit_patch),               # [K, M, pH, pW]
            torch.from_numpy(mask_patch),                # [pH, pW]
        )


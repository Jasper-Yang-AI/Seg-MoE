"""
Gating Patch Dataset: 加载缓存的 Layer2 OOF 概率图 [K,M,H,W] 和 GT mask,
按 patch 切分, 返回 (prob_patch, probs_structured, mask_patch) 用于门控网络训练.

正确流程:
  Layer1 train → L1 OOF → Layer2 train → **L2 OOF** → Gating train → Eval
  门控网络输入 = Layer2 专家的 OOF 概率图 (而非 Layer1 的).

每个 Dataset 样本 = 一张图的一个 patch.
一个 epoch = 所有训练图像的所有 patches.

示例:
  - image_size = 256×256, patch_size = 64, stride = 32
  - → 7×7 = 49 patches / image
  - 300 images → 14,700 patches / epoch
  - batch_size = 512 → ~29 steps / epoch
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from seg_moe.data.oof import load_oof_manifest, get_oof_prob_path
from seg_moe.utils.patches import compute_patch_positions


class GatingPatchDataset(Dataset):
    """Dataset for patch-level gating network training.

    Loads pre-cached **Layer2** expert probability maps ``[K, M, H, W]`` and GT masks,
    splits into patches, returns per-patch samples.

    Note: oof_manifest_path should point to the **Layer2** OOF manifest
    (oof_manifest_layer2.jsonl), NOT the Layer1 one.

    Returns
    -------
    prob_flat : [K*M, pH, pW]   — gating network input
    probs     : [K, M, pH, pW]  — for fusion (weighted sum)
    mask      : [pH, pW]        — ground-truth mask patch (int64)
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
        limit: int | None = None,
        cache_in_memory: bool = True,
    ) -> None:
        self.dataset_cfg = dataset_cfg
        self.patch_size = patch_size
        self.stride = stride
        self.is_train = is_train
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

        # In-memory cache (optional, ~360 MB for 300 images)
        self._probs_cache: dict[str, np.ndarray] = {}
        self._mask_cache: dict[str, np.ndarray] = {}
        if cache_in_memory:
            self._preload()

    def _preload(self) -> None:
        """Pre-load all prob maps and masks into RAM."""
        for s in self.samples:
            sid = str(s["id"])
            self._probs_cache[sid] = self._load_probs(sid)
            self._mask_cache[sid] = self._load_mask(s)

    def _load_probs(self, sample_id: str) -> np.ndarray:
        prob_path = get_oof_prob_path(self.oof_map, sample_id)
        probs = np.load(prob_path)["probs"].astype(np.float32)  # [K,M,H,W]
        return probs

    def _load_mask(self, sample: dict) -> np.ndarray:
        mask_path = sample["mask_path"]
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

    def __len__(self) -> int:
        return len(self.samples) * self.patches_per_image

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img_idx = idx // self.patches_per_image
        patch_idx = idx % self.patches_per_image

        sample = self.samples[img_idx]
        sid = str(sample["id"])

        # Load / retrieve from cache
        if self.cache_in_memory:
            probs = self._probs_cache[sid]
            mask = self._mask_cache[sid]
        else:
            probs = self._load_probs(sid)
            mask = self._load_mask(sample)

        # Extract patch
        y, x = self.positions[patch_idx]
        ps = self.patch_size

        prob_patch = probs[:, :, y : y + ps, x : x + ps].copy()  # [K, M, pH, pW]
        mask_patch = mask[y : y + ps, x : x + ps].copy()          # [pH, pW]

        # Training augmentation: random H/V flip (prob maps are symmetric)
        if self.is_train:
            if np.random.random() > 0.5:
                prob_patch = np.flip(prob_patch, axis=-1).copy()
                mask_patch = np.flip(mask_patch, axis=-1).copy()
            if np.random.random() > 0.5:
                prob_patch = np.flip(prob_patch, axis=-2).copy()
                mask_patch = np.flip(mask_patch, axis=-2).copy()

        # Flatten for gating input
        K, M, pH, pW = prob_patch.shape
        prob_flat = prob_patch.reshape(K * M, pH, pW)

        return (
            torch.from_numpy(prob_flat),                # [K*M, pH, pW]
            torch.from_numpy(prob_patch),               # [K, M, pH, pW]
            torch.from_numpy(mask_patch),               # [pH, pW]
        )

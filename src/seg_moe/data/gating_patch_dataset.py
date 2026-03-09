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
from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from seg_moe.data.oof import get_oof_prob_path, load_oof_manifest
from seg_moe.utils.patches import compute_patch_positions


class GatingPatchDataset(Dataset):
    """Patch dataset for 2D gating training.

    Each sample returns:
      - logits_patch: [K, M, pH, pW]
      - mask_patch:   [pH, pW]
      - sample_idx:   scalar int64 tensor
      - pos:          [2] int64 tensor with patch top-left (y, x)
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
        self.patch_size = int(patch_size)
        self.stride = int(stride)
        self.is_train = is_train
        self.fg_ratio = float(foreground_oversample_ratio) if is_train else 0.0
        self.K = int(expected_num_experts)
        self.M = int(dataset_cfg["task"]["num_classes"])
        self.label_map = {
            int(k): int(v)
            for k, v in dataset_cfg["task"].get("label_map", {}).items()
        }
        self.cache_in_memory = cache_in_memory
        self.oof_map = load_oof_manifest(oof_manifest_path)

        self.samples = [s for s in samples if str(s["id"]) in self.oof_map]
        if limit and limit < len(self.samples):
            self.samples = self.samples[:limit]

        self._logits_cache: dict[str, np.ndarray] = {}
        self._mask_cache: dict[str, np.ndarray] = {}
        self._patch_index: list[tuple[int, tuple[int, int]]] = []
        self._fg_index: list[int] = []

        self._build_patch_index()

    def _load_logits(self, sample_id: str) -> np.ndarray:
        prob_path = get_oof_prob_path(self.oof_map, sample_id)
        data = np.load(prob_path)
        if "logits" in data:
            logits = data["logits"].astype(np.float32)
        else:
            probs = np.clip(data["probs"].astype(np.float32), 1e-6, 1 - 1e-6)
            logits = np.log(probs / (1 - probs)).astype(np.float32)
        if logits.ndim != 4:
            raise ValueError(f"Expected [K,M,H,W] logits for gating, got shape={logits.shape}")
        return logits

    def _load_mask(self, sample: dict) -> np.ndarray:
        mask_path = sample.get("mask_path") or sample.get("mask")
        if not mask_path:
            raise ValueError(f"Sample missing mask path: {sample}")
        mask = Image.open(mask_path)
        if mask.mode != "L":
            mask = mask.convert("L")
        arr = np.array(mask, dtype=np.uint8)
        if self.label_map:
            mapped = arr.copy()
            for k, v in self.label_map.items():
                mapped[arr == k] = v
            arr = mapped
        return arr.astype(np.int64)

    def _get_logits(self, sample: dict) -> np.ndarray:
        sid = str(sample["id"])
        if sid not in self._logits_cache:
            self._logits_cache[sid] = self._load_logits(sid)
        return self._logits_cache[sid]

    def _get_mask(self, sample: dict) -> np.ndarray:
        sid = str(sample["id"])
        if sid not in self._mask_cache:
            self._mask_cache[sid] = self._load_mask(sample)
        return self._mask_cache[sid]

    def _build_patch_index(self) -> None:
        for sample_idx, sample in enumerate(self.samples):
            logits = self._get_logits(sample)
            mask = self._get_mask(sample)
            _, _, height, width = logits.shape
            if mask.shape != (height, width):
                raise ValueError(
                    f"Gating mask/logit shape mismatch for {sample['id']}: "
                    f"mask={mask.shape}, logits={(height, width)}"
                )

            positions = compute_patch_positions(height, width, self.patch_size, self.stride)
            for pos in positions:
                y, x = pos
                patch_idx = len(self._patch_index)
                self._patch_index.append((sample_idx, pos))
                if (mask[y : y + self.patch_size, x : x + self.patch_size] > 0).any():
                    self._fg_index.append(patch_idx)

        if not self._fg_index:
            self._fg_index = list(range(len(self._patch_index)))

        if not self.cache_in_memory:
            self._logits_cache.clear()
            self._mask_cache.clear()

    def __len__(self) -> int:
        return len(self._patch_index)

    def __getitem__(self, idx: int):
        if self.is_train and self.fg_ratio > 0 and self._fg_index and random.random() < self.fg_ratio:
            idx = random.choice(self._fg_index)

        sample_idx, (y, x) = self._patch_index[idx]
        sample = self.samples[sample_idx]
        logits = self._get_logits(sample) if self.cache_in_memory else self._load_logits(str(sample["id"]))
        mask = self._get_mask(sample) if self.cache_in_memory else self._load_mask(sample)

        ps = self.patch_size
        logits_patch = logits[:, :, y : y + ps, x : x + ps].copy()
        mask_patch = mask[y : y + ps, x : x + ps].copy()

        if self.is_train:
            if np.random.rand() > 0.5:
                logits_patch = np.flip(logits_patch, axis=-1).copy()
                mask_patch = np.flip(mask_patch, axis=-1).copy()
            if np.random.rand() > 0.5:
                logits_patch = np.flip(logits_patch, axis=-2).copy()
                mask_patch = np.flip(mask_patch, axis=-2).copy()

        return (
            torch.from_numpy(logits_patch).float(),
            torch.from_numpy(mask_patch).long(),
            torch.tensor(sample_idx, dtype=torch.long),
            torch.tensor([y, x], dtype=torch.long),
        )


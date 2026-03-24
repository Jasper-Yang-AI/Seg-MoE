"""Patch dataset for 2D gating with optional semantic priors and anatomy context."""
from __future__ import annotations

import math
import random
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

from seg_moe.data.oof import get_oof_prob_path, load_oof_manifest
from seg_moe.data.transforms import imagenet_normalize
from seg_moe.utils.patches import compute_patch_positions


def _softmax_logits(logits: np.ndarray) -> np.ndarray:
    logits = logits.astype(np.float32)
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / (exp_logits.sum(axis=1, keepdims=True) + 1e-8)


def build_layer1_semantic_maps(probs: np.ndarray) -> np.ndarray:
    """Build distilled semantic priors from Layer1 OOF probabilities.

    Input:
      probs: [K, M, H, W]

    Output:
      semantic maps: [2*M + 1, H, W]
      = concat([mean_probs[M], entropy[1], disagreement[M]])
    """
    if probs.ndim != 4:
        raise ValueError(f"Expected Layer1 probs [K,M,H,W], got shape={probs.shape}")
    _, num_classes, _, _ = probs.shape
    mean_probs = probs.mean(axis=0).astype(np.float32)
    disagreement = probs.std(axis=0).astype(np.float32)
    eps = 1e-8
    entropy = -(mean_probs * np.log(mean_probs + eps)).sum(axis=0, keepdims=True)
    if num_classes > 1:
        entropy = entropy / (np.log(float(num_classes)) + eps)
    return np.concatenate([mean_probs, entropy.astype(np.float32), disagreement], axis=0)


def build_position_channels(height: int, width: int) -> np.ndarray:
    """Return normalized XY coordinate channels in [-1, 1]."""
    yy = np.linspace(-1.0, 1.0, num=height, dtype=np.float32)
    xx = np.linspace(-1.0, 1.0, num=width, dtype=np.float32)
    grid_y, grid_x = np.meshgrid(yy, xx, indexing="ij")
    return np.stack([grid_y, grid_x], axis=0).astype(np.float32)


_SLICE_RE = re.compile(r"_z(\d+)$")


def extract_slice_index(sample_id: str) -> int | None:
    match = _SLICE_RE.search(str(sample_id))
    return int(match.group(1)) if match else None


class SampleGroupedBatchSampler(Sampler[list[int]]):
    """Batch sampler that keeps patches from the same slice close together.

    This improves cache locality for gating because each slice contributes many
    overlapping patches. Optionally replaces part of a sample's background
    patches with foreground patches from the same sample to keep foreground
    density similar to the legacy random-oversample path.
    """

    def __init__(
        self,
        sample_patch_ranges: list[tuple[int, int]],
        *,
        batch_size: int,
        sample_fg_indices: list[list[int]] | None = None,
        drop_last: bool = True,
        shuffle_samples: bool = True,
        shuffle_patches_within_sample: bool = True,
        foreground_oversample_ratio: float = 0.0,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.sample_patch_ranges = list(sample_patch_ranges)
        self.sample_fg_indices = sample_fg_indices or [[] for _ in self.sample_patch_ranges]
        if len(self.sample_fg_indices) != len(self.sample_patch_ranges):
            raise ValueError("sample_fg_indices must match sample_patch_ranges length")
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.shuffle_samples = bool(shuffle_samples)
        self.shuffle_patches_within_sample = bool(shuffle_patches_within_sample)
        self.foreground_oversample_ratio = max(0.0, float(foreground_oversample_ratio))
        self._num_indices = sum(max(0, end - start) for start, end in self.sample_patch_ranges)

    def __len__(self) -> int:
        if self.drop_last:
            return self._num_indices // self.batch_size
        return int(math.ceil(self._num_indices / max(1, self.batch_size)))

    def _indices_for_sample(self, sample_idx: int) -> list[int]:
        start, end = self.sample_patch_ranges[sample_idx]
        indices = list(range(start, end))
        if self.shuffle_patches_within_sample:
            random.shuffle(indices)

        fg_indices = self.sample_fg_indices[sample_idx]
        if self.foreground_oversample_ratio <= 0 or not fg_indices:
            return indices

        fg_set = set(fg_indices)
        bg_positions = [pos for pos, patch_idx in enumerate(indices) if patch_idx not in fg_set]
        if not bg_positions:
            return indices

        n_replace = min(len(bg_positions), int(round(len(indices) * self.foreground_oversample_ratio)))
        if n_replace <= 0:
            return indices

        replace_positions = random.sample(bg_positions, n_replace)
        for pos in replace_positions:
            indices[pos] = random.choice(fg_indices)
        return indices

    def __iter__(self):
        sample_order = list(range(len(self.sample_patch_ranges)))
        if self.shuffle_samples:
            random.shuffle(sample_order)

        batch: list[int] = []
        for sample_idx in sample_order:
            for patch_idx in self._indices_for_sample(sample_idx):
                batch.append(patch_idx)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []

        if batch and not self.drop_last:
            yield batch


class GatingPatchDataset(Dataset):
    """Patch dataset for 2D gating training.

    Each sample returns:
      - logits_patch: [K, M, pH, pW]
      - mask_patch:   [pH, pW]
      - sample_idx:   scalar int64 tensor
      - pos:          [2] int64 tensor with patch top-left (y, x)
      - extra:        dict of optional context tensors
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
        cache_max_items: int | None = None,
        layer1_oof_manifest_path: str | Path | None = None,
        use_layer1_semantics: bool = False,
        use_image_context: bool = False,
        use_position_channels: bool = False,
        use_slice_position: bool = False,
    ) -> None:
        self.dataset_cfg = dataset_cfg
        self.patch_size = int(patch_size)
        self.stride = int(stride)
        self.is_train = is_train
        self.fg_ratio = float(foreground_oversample_ratio) if is_train else 0.0
        self.K = int(expected_num_experts)
        self.M = int(dataset_cfg["task"]["num_classes"])
        self.image_channels = int(dataset_cfg["input"].get("image_channels", 3))
        self.label_map = {
            int(k): int(v)
            for k, v in dataset_cfg["task"].get("label_map", {}).items()
        }
        self.cache_in_memory = cache_in_memory
        self.cache_max_items = int(cache_max_items) if cache_max_items is not None else None
        self.use_layer1_semantics = bool(use_layer1_semantics)
        self.use_image_context = bool(use_image_context)
        self.use_position_channels = bool(use_position_channels)
        self.use_slice_position = bool(use_slice_position)

        self.oof_map = load_oof_manifest(oof_manifest_path)
        self.l1_oof_map = None
        if self.use_layer1_semantics:
            if layer1_oof_manifest_path is None:
                raise ValueError("layer1_oof_manifest_path is required when use_layer1_semantics=True")
            self.l1_oof_map = load_oof_manifest(layer1_oof_manifest_path)

        self.samples = [s for s in samples if str(s["id"]) in self.oof_map]
        if self.use_layer1_semantics:
            self.samples = [s for s in self.samples if str(s["id"]) in self.l1_oof_map]
        if limit and limit < len(self.samples):
            self.samples = self.samples[:limit]

        self._logits_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._mask_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._image_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._layer1_semantic_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._position_cache: dict[tuple[int, int], np.ndarray] = {}
        self._slice_pos: dict[str, float] = {}
        self._patch_index: list[tuple[int, tuple[int, int]]] = []
        self._fg_index: list[int] = []
        self._sample_patch_ranges: list[tuple[int, int]] = []
        self._sample_fg_indices: list[list[int]] = []

        self._build_slice_positions()
        self._build_patch_index()

    @property
    def sample_patch_ranges(self) -> list[tuple[int, int]]:
        return list(self._sample_patch_ranges)

    @property
    def sample_fg_indices(self) -> list[list[int]]:
        return [list(v) for v in self._sample_fg_indices]

    def _get_cached(self, cache: OrderedDict[str, np.ndarray], key: str) -> np.ndarray | None:
        if not self.cache_in_memory:
            return None
        value = cache.get(key)
        if value is None:
            return None
        cache.move_to_end(key)
        return value

    def _put_cached(self, cache: OrderedDict[str, np.ndarray], key: str, value: np.ndarray) -> np.ndarray:
        if not self.cache_in_memory:
            return value
        cache[key] = value
        cache.move_to_end(key)
        if self.cache_max_items is not None:
            while len(cache) > self.cache_max_items:
                cache.popitem(last=False)
        return value

    def _read_image(self, path: str) -> np.ndarray:
        img = Image.open(path)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        if self.image_channels == 1:
            img = img.convert("L")
            arr = np.array(img, dtype=np.uint8)[:, :, None]
        else:
            img = img.convert("RGB")
            arr = np.array(img, dtype=np.uint8)
        return arr

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

    def _load_layer1_semantics(self, sample_id: str) -> np.ndarray:
        if self.l1_oof_map is None:
            raise RuntimeError("Layer1 OOF map is not available")
        prob_path = get_oof_prob_path(self.l1_oof_map, sample_id)
        data = np.load(prob_path)
        if "logits" in data:
            probs = _softmax_logits(data["logits"])
        elif "probs" in data:
            probs = data["probs"].astype(np.float32)
        else:
            raise KeyError(f"Layer1 OOF cache missing 'logits'/'probs' for sample_id={sample_id}: {prob_path}")
        return build_layer1_semantic_maps(probs)

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
        cached = self._get_cached(self._logits_cache, sid)
        if cached is not None:
            return cached
        return self._put_cached(self._logits_cache, sid, self._load_logits(sid))

    def _get_mask(self, sample: dict) -> np.ndarray:
        sid = str(sample["id"])
        cached = self._get_cached(self._mask_cache, sid)
        if cached is not None:
            return cached
        return self._put_cached(self._mask_cache, sid, self._load_mask(sample))

    def _get_image(self, sample: dict) -> np.ndarray:
        sid = str(sample["id"])
        cached = self._get_cached(self._image_cache, sid)
        if cached is not None:
            return cached
        return self._put_cached(self._image_cache, sid, self._read_image(sample["image_path"]))

    def _get_layer1_semantics(self, sample: dict) -> np.ndarray:
        sid = str(sample["id"])
        cached = self._get_cached(self._layer1_semantic_cache, sid)
        if cached is not None:
            return cached
        return self._put_cached(self._layer1_semantic_cache, sid, self._load_layer1_semantics(sid))

    def _get_position_channels(self, height: int, width: int) -> np.ndarray:
        key = (height, width)
        if key not in self._position_cache:
            self._position_cache[key] = build_position_channels(height, width)
        return self._position_cache[key]

    def _build_slice_positions(self) -> None:
        if not self.use_slice_position:
            return
        patient_to_z: dict[str, list[int]] = {}
        for sample in self.samples:
            sid = str(sample["id"])
            pid = str(sample.get("patient_id") or sid)
            z = extract_slice_index(sid)
            if z is None:
                continue
            patient_to_z.setdefault(pid, []).append(z)

        for sample in self.samples:
            sid = str(sample["id"])
            pid = str(sample.get("patient_id") or sid)
            z = extract_slice_index(sid)
            if z is None or pid not in patient_to_z:
                self._slice_pos[sid] = 0.5
                continue
            z_vals = patient_to_z[pid]
            z_min = min(z_vals)
            z_max = max(z_vals)
            if z_max <= z_min:
                self._slice_pos[sid] = 0.5
            else:
                self._slice_pos[sid] = float((z - z_min) / (z_max - z_min))

    def _build_patch_index(self) -> None:
        for sample_idx, sample in enumerate(self.samples):
            mask = self._get_mask(sample)
            height, width = mask.shape
            positions = compute_patch_positions(height, width, self.patch_size, self.stride)
            start_idx = len(self._patch_index)
            sample_fg_indices: list[int] = []
            for pos in positions:
                y, x = pos
                patch_idx = len(self._patch_index)
                self._patch_index.append((sample_idx, pos))
                if (mask[y : y + self.patch_size, x : x + self.patch_size] > 0).any():
                    self._fg_index.append(patch_idx)
                    sample_fg_indices.append(patch_idx)
            self._sample_patch_ranges.append((start_idx, len(self._patch_index)))
            self._sample_fg_indices.append(sample_fg_indices)

        if not self._fg_index:
            self._fg_index = list(range(len(self._patch_index)))

        if not self.cache_in_memory:
            self._logits_cache.clear()
            self._mask_cache.clear()
            self._image_cache.clear()
            self._layer1_semantic_cache.clear()

    def __len__(self) -> int:
        return len(self._patch_index)

    def __getitem__(self, idx: int):
        if self.is_train and self.fg_ratio > 0 and self._fg_index and random.random() < self.fg_ratio:
            idx = random.choice(self._fg_index)

        sample_idx, (y, x) = self._patch_index[idx]
        sample = self.samples[sample_idx]
        sid = str(sample["id"])
        logits = self._get_logits(sample) if self.cache_in_memory else self._load_logits(sid)
        mask = self._get_mask(sample) if self.cache_in_memory else self._load_mask(sample)
        if mask.shape != tuple(logits.shape[-2:]):
            raise ValueError(
                f"Gating mask/logit shape mismatch for {sample['id']}: "
                f"mask={mask.shape}, logits={tuple(logits.shape[-2:])}"
            )

        ps = self.patch_size
        logits_patch = logits[:, :, y : y + ps, x : x + ps].copy()
        mask_patch = mask[y : y + ps, x : x + ps].copy()

        image_patch = None
        if self.use_image_context:
            image = self._get_image(sample) if self.cache_in_memory else self._read_image(sample["image_path"])
            image_patch = image[y : y + ps, x : x + ps].copy()

        layer1_patch = None
        if self.use_layer1_semantics:
            semantic = (
                self._get_layer1_semantics(sample)
                if self.cache_in_memory else self._load_layer1_semantics(sid)
            )
            layer1_patch = semantic[:, y : y + ps, x : x + ps].copy()

        coords_patch = None
        if self.use_position_channels:
            coords = self._get_position_channels(mask.shape[0], mask.shape[1])
            coords_patch = coords[:, y : y + ps, x : x + ps].copy()

        if self.is_train:
            if np.random.rand() > 0.5:
                logits_patch = np.flip(logits_patch, axis=-1).copy()
                mask_patch = np.flip(mask_patch, axis=-1).copy()
                if image_patch is not None:
                    image_patch = np.flip(image_patch, axis=1).copy()
                if layer1_patch is not None:
                    layer1_patch = np.flip(layer1_patch, axis=-1).copy()
                if coords_patch is not None:
                    coords_patch = np.flip(coords_patch, axis=-1).copy()
            if np.random.rand() > 0.5:
                logits_patch = np.flip(logits_patch, axis=-2).copy()
                mask_patch = np.flip(mask_patch, axis=-2).copy()
                if image_patch is not None:
                    image_patch = np.flip(image_patch, axis=0).copy()
                if layer1_patch is not None:
                    layer1_patch = np.flip(layer1_patch, axis=-2).copy()
                if coords_patch is not None:
                    coords_patch = np.flip(coords_patch, axis=-2).copy()

        extra: dict[str, torch.Tensor] = {}
        if image_patch is not None:
            image_patch = imagenet_normalize(image_patch)
            image_patch = np.transpose(image_patch.astype(np.float32), (2, 0, 1))
            extra["image"] = torch.from_numpy(image_patch).float()

        if layer1_patch is not None:
            extra["layer1_mean"] = torch.from_numpy(layer1_patch[: self.M]).float()
            extra["layer1_entropy"] = torch.from_numpy(layer1_patch[self.M : self.M + 1]).float()
            extra["layer1_disagreement"] = torch.from_numpy(layer1_patch[self.M + 1 :]).float()

        if coords_patch is not None:
            extra["coords"] = torch.from_numpy(coords_patch).float()

        if self.use_slice_position:
            extra["slice_pos"] = torch.tensor([self._slice_pos.get(sid, 0.5)], dtype=torch.float32)

        return (
            torch.from_numpy(logits_patch).float(),
            torch.from_numpy(mask_patch).long(),
            torch.tensor(sample_idx, dtype=torch.long),
            torch.tensor([y, x], dtype=torch.long),
            extra,
        )

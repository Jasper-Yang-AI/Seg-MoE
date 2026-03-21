from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from seg_moe.data.transforms import build_albu, normalize_image


class SegmentationDataset2D(Dataset):
    """2D segmentation dataset reading prepared PNGs and split indices.

    Each item returns:
      - image: float32 tensor [C,H,W]
      - mask: int64 tensor [H,W] with values 0..(M-1)
      - meta: dict with id/dataset/split
    """

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        dataset_cfg: Dict[str, Any],
        augs_cfg: Optional[Dict[str, Any]] = None,
        is_train: bool = False,
        limit: Optional[int] = None,
    ) -> None:
        self.samples = samples[: (limit or len(samples))]
        self.dataset_cfg = dataset_cfg
        self.num_classes = int(dataset_cfg["task"]["num_classes"])
        self.image_size = tuple(dataset_cfg["input"]["image_size"])  # H,W
        self.image_channels = int(dataset_cfg["input"].get("image_channels", 3))
        self.normalize_cfg = dict(dataset_cfg["input"].get("normalize", {}) or {})
        self.label_map = {int(k): int(v) for k, v in dataset_cfg["task"].get("label_map", {}).items()}

        self.aug = build_albu(augs_cfg, is_train) if augs_cfg else None

    def __len__(self) -> int:
        return len(self.samples)

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

    def _read_mask(self, path: str) -> np.ndarray:
        m = Image.open(path)
        if m.mode != "L":
            m = m.convert("L")
        arr = np.array(m, dtype=np.uint8)
        if self.label_map:
            mapped = np.zeros_like(arr, dtype=np.uint8)
            # default: unseen values map to themselves
            mapped[:] = arr
            for k, v in self.label_map.items():
                mapped[arr == k] = v
            arr = mapped
        return arr

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = self._read_image(s["image_path"])
        mask = self._read_mask(s["mask_path"])

        if self.aug is not None:
            augmented = self.aug(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

        # Normalize + to tensor
        img = normalize_image(img, self.normalize_cfg)
        img = np.transpose(img, (2, 0, 1))  # CHW
        img_t = torch.from_numpy(img).float()
        mask_t = torch.from_numpy(mask.astype(np.int64))

        meta = {
            "id": s["id"],
            "dataset": s.get("dataset"),
            "split": s.get("split"),
            "image_path": s.get("image_path"),
            "mask_path": s.get("mask_path"),
            "spacing_yx": s.get("spacing_yx"),
        }
        return img_t, mask_t, meta

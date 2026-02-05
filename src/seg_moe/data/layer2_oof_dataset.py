from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from seg_moe.data.oof import get_oof_prob_path, load_oof_manifest
from seg_moe.data.transforms import build_albu_for_layer2_with_probs, imagenet_normalize


class Layer2OOFDataset(Dataset):
    """Layer2 dataset that concatenates image with OOF layer1 probability maps.

    This prevents leakage: each sample's probs must be produced by a layer1 model
    that did NOT see that sample during training.

    Augmentations:
    - Spatial transforms are applied to (image, mask, probs) together.
    - Image-only transforms are applied to image only.

    The OOF probabilities are expected to be stored as npz with key 'probs':
      probs shape [K, M, H, W] (experts x classes x height x width)

    The model input becomes:
      x = concat([image[C,H,W], probs_flat[K*M,H,W]], dim=0)
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
    ) -> None:
        self.samples = samples[: (limit or len(samples))]
        self.dataset_cfg = dataset_cfg
        self.num_classes = int(dataset_cfg["task"]["num_classes"])
        self.image_channels = int(dataset_cfg["input"].get("image_channels", 3))
        self.label_map = {int(k): int(v) for k, v in dataset_cfg["task"].get("label_map", {}).items()}
        self.expected_num_experts = int(expected_num_experts) if expected_num_experts is not None else None

        self.oof_map = load_oof_manifest(oof_manifest_path)

        self.spatial_aug = None
        self.image_aug = None
        if augs_cfg is not None:
            self.spatial_aug, self.image_aug = build_albu_for_layer2_with_probs(augs_cfg, is_train)

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
            mapped[:] = arr
            for k, v in self.label_map.items():
                mapped[arr == k] = v
            arr = mapped
        return arr

    def _read_oof_probs(self, sample_id: str) -> np.ndarray:
        prob_path = get_oof_prob_path(self.oof_map, sample_id)
        if not prob_path.exists():
            raise FileNotFoundError(
                f"Missing OOF probability file for sample_id={sample_id}: {prob_path}. "
                "Run scripts/generate_layer1_oof.py first."
            )
        npz = np.load(prob_path)
        probs = npz["probs"].astype(np.float32)  # [K,M,H,W]
        if probs.ndim != 4:
            raise ValueError(f"Unexpected probs shape for {sample_id}: {probs.shape} (expected [K,M,H,W])")
        return probs

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        sample_id = s["id"]

        img = self._read_image(s["image_path"])  # HWC
        mask = self._read_mask(s["mask_path"])  # HW

        probs = self._read_oof_probs(sample_id)  # [K,M,H,W]
        k, m, h, w = probs.shape
        if self.expected_num_experts is not None and k != self.expected_num_experts:
            raise ValueError(
                f"OOF probs num_experts mismatch for sample_id={sample_id}: {k} vs expected {self.expected_num_experts}. "
                "Regenerate OOF cache with the same expert list used for layer2 training."
            )
        if m != self.num_classes:
            raise ValueError(
                f"OOF probs num_classes mismatch for sample_id={sample_id}: {m} vs dataset {self.num_classes}"
            )

        # Flatten probs to channels and convert to HWC for albumentations
        probs_flat = probs.reshape(k * m, h, w).transpose(1, 2, 0)  # HWC

        if self.spatial_aug is not None:
            out = self.spatial_aug(image=img, mask=mask, probs=probs_flat)
            img = out["image"]
            mask = out["mask"]
            probs_flat = out["probs"]

        if self.image_aug is not None:
            img = self.image_aug(image=img)["image"]

        # Normalize image (replicates grayscale->3ch internally)
        img = imagenet_normalize(img)
        img = np.transpose(img, (2, 0, 1))  # CHW
        img_t = torch.from_numpy(img).float()

        # probs: back to CHW
        probs_chw = np.transpose(probs_flat.astype(np.float32), (2, 0, 1))
        probs_t = torch.from_numpy(probs_chw).float()

        x = torch.cat([img_t, probs_t], dim=0)
        mask_t = torch.from_numpy(mask.astype(np.int64))

        rec = self.oof_map.get(str(sample_id))
        meta = {
            "id": sample_id,
            "dataset": s.get("dataset"),
            "split": s.get("split"),
            "image_path": s.get("image_path"),
            "mask_path": s.get("mask_path"),
            "oof_prob_path": str(rec.prob_path) if rec else None,
            "oof_sample_fold": int(rec.sample_fold) if rec else None,
            "oof_predictor_fold": int(rec.predictor_fold) if rec else None,
            "spacing_yx": s.get("spacing_yx"),
        }
        return x, mask_t, meta

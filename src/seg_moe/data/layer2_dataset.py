from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from seg_moe.data.dataset_2d import SegmentationDataset2D
from seg_moe.data.oof import OOFRecord, load_oof_manifest


class Layer2Dataset(torch.utils.data.Dataset):
    """Build I* input = concat(original image, layer1 probs).

    - base dataset returns image [C,H,W]
    - layer1 cache provides probs [K,M,H,W]

    Output image channels become C + K*M.
    """

    def __init__(
        self,
        base: SegmentationDataset2D,
        cache_dir: Optional[Path],
        num_experts: int,
        num_classes: int,
        *,
        oof_manifest_path: Optional[str | Path] = None,
    ):
        self.base = base
        self.cache_dir = cache_dir
        self.K = int(num_experts)
        self.M = int(num_classes)
        self.oof_map: Optional[dict[str, OOFRecord]] = None
        if oof_manifest_path is not None:
            self.oof_map = load_oof_manifest(oof_manifest_path)

    def _prob_path_for_sample(self, sample_id: str) -> Path:
        if self.oof_map is not None and sample_id in self.oof_map:
            return self.oof_map[sample_id].prob_path
        if self.cache_dir is not None:
            return self.cache_dir / f"{sample_id}.npz"
        raise KeyError(
            f"Missing layer1 probs for sample_id={sample_id}. "
            "Provide oof_manifest_path that contains this sample_id or set cache_dir to a layer1_probs directory."
        )

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx: int):
        img, mask, meta = self.base[idx]
        sid = meta["id"]
        prob_path = self._prob_path_for_sample(str(sid))
        if not prob_path.exists():
            raise FileNotFoundError(f"Missing layer1 probs file for sample_id={sid}: {prob_path}")
        npz = np.load(prob_path)
        probs = npz["probs"].astype(np.float32)  # [K,M,H,W]
        if probs.ndim != 4:
            raise ValueError(f"Unexpected probs shape for sample_id={sid}: {probs.shape} (expected [K,M,H,W])")
        if probs.shape[0] != self.K or probs.shape[1] != self.M:
            raise ValueError(
                f"Layer1 probs shape mismatch for sample_id={sid}: got {list(probs.shape)} expected [K={self.K},M={self.M},H,W]."
            )
        probs_flat = probs.reshape(self.K * self.M, probs.shape[-2], probs.shape[-1])
        probs_t = torch.from_numpy(probs_flat)
        x = torch.cat([img, probs_t], dim=0)
        return x, mask, meta

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from seg_moe.data.dataset_2d import SegmentationDataset2D


class Layer2Dataset(torch.utils.data.Dataset):
    """Build I* input = concat(original image, layer1 probs).

    - base dataset returns image [C,H,W]
    - layer1 cache provides probs [K,M,H,W]

    Output image channels become C + K*M.
    """

    def __init__(self, base: SegmentationDataset2D, cache_dir: Path, num_experts: int, num_classes: int):
        self.base = base
        self.cache_dir = cache_dir
        self.K = int(num_experts)
        self.M = int(num_classes)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx: int):
        img, mask, meta = self.base[idx]
        sid = meta["id"]
        npz = np.load(self.cache_dir / f"{sid}.npz")
        probs = npz["probs"].astype(np.float32)  # [K,M,H,W]
        probs_flat = probs.reshape(self.K * self.M, probs.shape[-2], probs.shape[-1])
        probs_t = torch.from_numpy(probs_flat)
        x = torch.cat([img, probs_t], dim=0)
        return x, mask, meta

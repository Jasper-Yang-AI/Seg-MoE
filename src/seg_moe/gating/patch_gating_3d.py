from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class PatchGating3DConfig:
    """Configuration placeholder for future 3D patch/region gating.

    Planned fields (not exhaustive):
    - num_experts: K
    - num_classes: M
    - patch_size: (d,h,w)
    - stride: (sd,sh,sw)
    - per_class: whether to output weights per-class
    """

    num_experts: int
    num_classes: int
    per_class: bool = False


class PatchGating3D(torch.nn.Module):
    """Patch/region-level dynamic gating for 3D fusion (reserved interface).

    This module is intentionally a stub for the later innovation.

    Expected future interface
    -------------------------
    forward(inputs) -> weights

    where inputs can be:
      - feature maps: [B, F, D, H, W]
      - or expert probs/logits: [B, K, M, D, H, W]

    and weights can be:
      - per-expert weights: [B, K, D', H', W']
      - optionally per-class: [B, K, M, D', H', W']

    Notes
    -----
    The 2D reproduction pipeline does not depend on this module.
    """

    def __init__(self, cfg: PatchGating3DConfig):
        super().__init__()
        self.cfg = cfg

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("PatchGating3D is a reserved stub; implement in the 3D roadmap phase.")

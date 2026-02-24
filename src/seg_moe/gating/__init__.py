"""Gating modules — patch-level dynamic expert fusion.

2D:
  - PatchConvGate2D: ConvNet gating network for 2D probability patches
  - PatchGatingConfig: gating configuration dataclass
3D:
  - PatchGating3D: reserved stub for 3D roadmap
"""
from seg_moe.gating.patch_gating_2d import (       # noqa: F401
    PatchConvGate2D,
    PatchGatingConfig,
    compute_load_balance_loss,
    compute_temperature,
)
from seg_moe.gating.patch_gating_3d import PatchGating3D  # noqa: F401

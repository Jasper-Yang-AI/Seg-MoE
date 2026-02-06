"""
Swin-UNetR Expert — MONAI SwinUNETR wrapped as BaseExpert.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from seg_moe.models.experts.base_expert import BaseExpert


class SwinUNETRExpert(BaseExpert):
    """Swin-UNetR 3D transformer expert."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 3,
        *,
        spatial_dims: int = 3,
        feature_size: int = 48,
        depths: Optional[List[int]] = None,
        num_heads: Optional[List[int]] = None,
        use_checkpoint: bool = False,
        expert_name: str = "swin-unetr-base",
        pretrained_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        from monai.networks.nets import SwinUNETR

        self._name = expert_name
        self._num_classes = out_channels

        depths = depths or [2, 2, 2, 2]
        num_heads = num_heads or [3, 6, 12, 24]

        self.model = SwinUNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            spatial_dims=spatial_dims,
            depths=depths,
            num_heads=num_heads,
            use_checkpoint=use_checkpoint,
        )

        if pretrained_path:
            self.load_checkpoint(pretrained_path, strict=False)
            print(f"[Swin-UNetR] Loaded pretrained from {pretrained_path}")

    @property
    def name(self) -> str:
        return self._name

    @property
    def num_classes(self) -> int:
        return self._num_classes

    def predict_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def build_swinunetr_expert(
    in_channels: int = 1,
    out_channels: int = 3,
    config: Optional[Dict[str, Any]] = None,
) -> SwinUNETRExpert:
    config = config or {}
    return SwinUNETRExpert(
        in_channels=in_channels,
        out_channels=out_channels,
        spatial_dims=config.get("spatial_dims", 3),
        feature_size=config.get("feature_size", 48),
        depths=config.get("depths", [2, 2, 2, 2]),
        num_heads=config.get("num_heads", [3, 6, 12, 24]),
        use_checkpoint=config.get("use_checkpoint", False),
        expert_name=config.get("name", "swin-unetr-base"),
        pretrained_path=config.get("pretrained_path"),
    )

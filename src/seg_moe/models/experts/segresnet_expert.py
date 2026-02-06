"""
SegResNet Expert — MONAI SegResNet (3D) wrapped as a BaseExpert.

SegResNet: Myronenko (2019) "3D MRI brain tumor segmentation using
autoencoder regularization" — MICCAI BraTS challenge winner.

Key properties:
  - Fully 3D residual encoder-decoder
  - Lightweight (~4.7 M params at init_filters=32, blocks_down=[1,2,2,4])
  - No external CUDA extensions (unlike Mamba-SSM)
  - Natively supports AMP (bf16/fp16)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from seg_moe.models.experts.base_expert import BaseExpert


class SegResNetExpert(BaseExpert):
    """SegResNet 3D expert with configurable hyper-parameters.

    All constructor parameters are also exposed in the YAML config
    ``configs/2d/models_sota.yaml`` → ``segresnet.config``.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 3,
        *,
        spatial_dims: int = 3,
        init_filters: int = 32,
        blocks_down: Optional[List[int]] = None,
        blocks_up: Optional[List[int]] = None,
        dropout_prob: float = 0.2,
        norm: str = "instance",       # instance | group | batch
        act: str = "relu",            # relu | leakyrelu | prelu
        expert_name: str = "segresnet-base",
        pretrained_path: Optional[str] = None,
    ) -> None:
        super().__init__()

        self._name = expert_name
        self._num_classes = out_channels

        # Lazy import so that MONAI is only required when actually used
        from monai.networks.nets import SegResNet

        blocks_down = blocks_down or [1, 2, 2, 4]
        blocks_up = blocks_up or [1, 1, 1]

        # Map string norm/act to MONAI tuple format
        _norm_map = {
            "instance": ("instance", {"affine": True}),
            "group":    ("group", {"num_groups": 8}),
            "batch":    ("batch", {"affine": True}),
        }
        _act_map = {
            "relu":      ("relu", {"inplace": True}),
            "leakyrelu": ("leakyrelu", {"negative_slope": 0.01, "inplace": True}),
            "prelu":     ("prelu", {}),
            "silu":      ("silu", {"inplace": True}),
        }

        norm_cfg = _norm_map.get(norm.lower(), norm)
        act_cfg = _act_map.get(act.lower(), act)

        self.model = SegResNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            init_filters=init_filters,
            blocks_down=blocks_down,
            blocks_up=blocks_up,
            dropout_prob=dropout_prob,
            norm=norm_cfg,
            act=act_cfg,
        )

        if pretrained_path:
            self.load_checkpoint(pretrained_path, strict=False)
            print(f"[SegResNet] Loaded pretrained from {pretrained_path}")

    # ---- BaseExpert implementation ----

    @property
    def name(self) -> str:
        return self._name

    @property
    def num_classes(self) -> int:
        return self._num_classes

    def predict_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass → logits [B, M, D, H, W] (no softmax)."""
        return self.model(x)


def build_segresnet_expert(
    in_channels: int = 1,
    out_channels: int = 3,
    config: Optional[Dict[str, Any]] = None,
) -> SegResNetExpert:
    """Convenience factory matching ``build_sota_model`` signature."""
    config = config or {}
    return SegResNetExpert(
        in_channels=in_channels,
        out_channels=out_channels,
        spatial_dims=config.get("spatial_dims", 3),
        init_filters=config.get("init_filters", 32),
        blocks_down=config.get("blocks_down", [1, 2, 2, 4]),
        blocks_up=config.get("blocks_up", [1, 1, 1]),
        dropout_prob=config.get("dropout_prob", 0.2),
        norm=config.get("norm", "instance"),
        act=config.get("act", "relu"),
        expert_name=config.get("name", "segresnet-base"),
        pretrained_path=config.get("pretrained_path"),
    )

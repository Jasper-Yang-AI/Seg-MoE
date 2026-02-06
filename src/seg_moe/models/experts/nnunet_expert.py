"""
nnUNet v2 Expert — PlainConvUNet wrapped as BaseExpert.

支持 deep_supervision=True 用于加载官方 nnUNet 训练权重。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from seg_moe.models.experts.base_expert import BaseExpert


class NnUNetExpert(BaseExpert):
    """nnUNet v2 CNN expert (PlainConvUNet from dynamic_network_architectures).

    When deep_supervision=True (required for loading official nnUNet weights),
    the model has segmentation heads at each decoder stage. forward/predict_logits
    always returns only the highest-resolution output.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 3,
        *,
        patch_size: Union[Tuple[int, ...], List[int]] = (96, 96, 96),
        n_stages: int = 6,
        features_per_stage: Optional[List[int]] = None,
        conv_op: str = "Conv3d",
        expert_name: str = "nnunet-v2",
        deep_supervision: bool = False,
        n_conv_per_stage_encoder: Optional[List[int]] = None,
        n_conv_per_stage_decoder: Optional[List[int]] = None,
        kernel_sizes: Optional[List[List[int]]] = None,
        strides: Optional[List[List[int]]] = None,
    ) -> None:
        super().__init__()

        self._name = expert_name
        self._num_classes = out_channels
        self.is_3d = len(patch_size) == 3
        self.deep_supervision = deep_supervision

        features_per_stage = features_per_stage or [32, 64, 128, 256, 320, 320]
        features_per_stage = features_per_stage[:n_stages]

        if n_conv_per_stage_encoder is None:
            n_conv_per_stage_encoder = [2] * n_stages
        if n_conv_per_stage_decoder is None:
            n_conv_per_stage_decoder = [2] * (n_stages - 1)

        from dynamic_network_architectures.architectures.unet import PlainConvUNet

        conv_op_class = nn.Conv3d if self.is_3d else nn.Conv2d
        norm_op = nn.InstanceNorm3d if self.is_3d else nn.InstanceNorm2d
        ndim = 3 if self.is_3d else 2

        if kernel_sizes is None:
            kernel_sizes = [[3] * ndim] * n_stages
        if strides is None:
            strides = [[1] * ndim] + [[2] * ndim] * (n_stages - 1)

        self.model = PlainConvUNet(
            input_channels=in_channels,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=conv_op_class,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_conv_per_stage=n_conv_per_stage_encoder,
            num_classes=out_channels,
            n_conv_per_stage_decoder=n_conv_per_stage_decoder,
            conv_bias=True,
            norm_op=norm_op,
            norm_op_kwargs={"eps": 1e-5, "affine": True},
            dropout_op=None,
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={"inplace": True},
            deep_supervision=deep_supervision,
        )
        n_params = sum(p.numel() for p in self.model.parameters())
        ds_tag = " +deep_supervision" if deep_supervision else ""
        print(f"[nnUNet] PlainConvUNet ({n_stages} stages, {n_params:,} params{ds_tag})")

    @property
    def name(self) -> str:
        return self._name

    @property
    def num_classes(self) -> int:
        return self._num_classes

    def predict_logits(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        if isinstance(out, (list, tuple)):
            return out[0]
        return out

    def set_deep_supervision_enabled(self, enabled: bool) -> None:
        """Toggle deep supervision (mirrors official nnUNet behavior)."""
        if hasattr(self.model, "decoder") and hasattr(self.model.decoder, "deep_supervision"):
            self.model.decoder.deep_supervision = enabled
            self.deep_supervision = enabled

    def train(self, mode: bool = True):
        """Override to manage deep supervision state (mirrors official nnUNet)."""
        super().train(mode)
        if self.deep_supervision:
            self.set_deep_supervision_enabled(mode)
        return self


def build_nnunet_expert(
    in_channels: int = 1,
    out_channels: int = 3,
    config: Optional[Dict[str, Any]] = None,
) -> NnUNetExpert:
    config = config or {}
    return NnUNetExpert(
        in_channels=in_channels,
        out_channels=out_channels,
        patch_size=config.get("patch_size", [96, 96, 96]),
        n_stages=config.get("n_stages", 6),
        features_per_stage=config.get("features_per_stage", [32, 64, 128, 256, 320, 320]),
        conv_op=config.get("conv_op", "Conv3d"),
        expert_name=config.get("name", "nnunet-v2"),
        deep_supervision=config.get("deep_supervision", False),
        n_conv_per_stage_encoder=config.get("n_conv_per_stage_encoder"),
        n_conv_per_stage_decoder=config.get("n_conv_per_stage_decoder"),
        kernel_sizes=config.get("conv_kernel_sizes") or config.get("kernel_sizes"),
        strides=config.get("pool_op_kernel_sizes") or config.get("strides"),
    )

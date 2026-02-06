"""
Lightweight wrapper for nnUNet v2 architecture (PlainConvUNet).

完全兼容官方 nnUNet v2:
  - 训练导入: deep_supervision=True 加载官方 1000-epoch 权重
  - 推理模式: 自动禁用深度监督 (与官方一致), forward() 直接输出单张量
  - 也可用于 Seg-MoE 自定义训练 (deep_supervision=False)

PlainConvUNet 来自 dynamic_network_architectures 包,
由 nnUNet v2 官方团队 (MIC-DKFZ) 维护.

For full nnUNet training pipeline, see: https://github.com/MIC-DKFZ/nnUNet
"""
from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn


class NnUNetWrapper(nn.Module):
    """nnUNet v2 PlainConvUNet wrapper, compatible with Seg-MoE pipeline.

    与官方 nnUNet v2 完全一致的架构参数和行为:
      - 使用 PlainConvUNet (dynamic_network_architectures)
      - 支持 deep_supervision 用于加载官方权重
      - eval() 模式自动禁用深度监督 (官方行为)
      - InstanceNorm + LeakyReLU + no dropout (官方默认)

    Args:
        in_channels: 输入通道数 (e.g., 1 for CT).
        num_classes: 输出类别数 (含背景).
        patch_size: 2D (H,W) 或 3D (D,H,W).
        n_stages: 编解码器阶段数.
        features_per_stage: 每阶段通道数 (来自 nnUNet plans).
        conv_op: "Conv2d" / "Conv3d".
        deep_supervision: True=使用深度监督头 (官方训练), forward() 时
            通过 train/eval 模式切换输出格式.
        n_conv_per_stage_encoder: 编码器每阶段卷积数.
        n_conv_per_stage_decoder: 解码器每阶段卷积数.
        kernel_sizes: 每阶段卷积核大小 (来自 nnUNet plans).
        strides: 每阶段步幅/池化核大小 (来自 nnUNet plans).
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        patch_size: Union[Tuple[int, ...], List[int]] = (256, 256),
        n_stages: int = 6,
        features_per_stage: Optional[List[int]] = None,
        conv_op: str = "Conv2d",
        deep_supervision: bool = False,
        n_conv_per_stage_encoder: Optional[List[int]] = None,
        n_conv_per_stage_decoder: Optional[List[int]] = None,
        kernel_sizes: Optional[List[List[int]]] = None,
        strides: Optional[List[List[int]]] = None,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.n_stages = n_stages
        self.is_3d = len(patch_size) == 3
        self._deep_supervision = deep_supervision

        if features_per_stage is None:
            features_per_stage = [32, 64, 128, 256, 320, 320]
        self.features_per_stage = features_per_stage[:n_stages]

        # 默认: 每级 2 个卷积 (官方默认)
        if n_conv_per_stage_encoder is None:
            n_conv_per_stage_encoder = [2] * n_stages
        if n_conv_per_stage_decoder is None:
            n_conv_per_stage_decoder = [2] * (n_stages - 1)

        try:
            from dynamic_network_architectures.architectures.unet import PlainConvUNet

            conv_op_class = nn.Conv3d if self.is_3d else nn.Conv2d
            norm_op = nn.InstanceNorm3d if self.is_3d else nn.InstanceNorm2d
            ndim = 3 if self.is_3d else 2

            # 默认 kernel_sizes 和 strides
            if kernel_sizes is None:
                kernel_sizes = [[3] * ndim] * n_stages
            if strides is None:
                strides = [[1] * ndim] + [[2] * ndim] * (n_stages - 1)

            # 与官方 nnUNet v2 ExperimentPlanner 完全一致的参数:
            # norm_op_kwargs={'eps': 1e-5, 'affine': True}
            # nonlin=LeakyReLU, nonlin_kwargs={'inplace': True}
            # dropout_op=None, conv_bias=True
            self.model = PlainConvUNet(
                input_channels=in_channels,
                n_stages=n_stages,
                features_per_stage=self.features_per_stage,
                conv_op=conv_op_class,
                kernel_sizes=kernel_sizes,
                strides=strides,
                n_conv_per_stage=n_conv_per_stage_encoder,
                num_classes=num_classes,
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
            print(f"[nnUNet] PlainConvUNet ({n_stages} stages, "
                  f"features={self.features_per_stage}, "
                  f"{n_params:,} params{ds_tag})")

        except ImportError as e:
            raise ImportError(
                f"Could not import PlainConvUNet ({e}). "
                "Install: pip install dynamic-network-architectures>=0.3 nnunetv2>=2.2"
            )

    def set_deep_supervision_enabled(self, enabled: bool) -> None:
        """Toggle deep supervision on/off (mirrors official nnUNet behavior).

        Official nnUNet disables deep supervision during validation and inference
        by setting decoder.deep_supervision = False.
        """
        if hasattr(self.model, "decoder") and hasattr(self.model.decoder, "deep_supervision"):
            self.model.decoder.deep_supervision = enabled
            self._deep_supervision = enabled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass — always returns single highest-resolution output.

        当 deep_supervision=True 且 model.training=True 时,
        PlainConvUNet 返回多尺度 list, 我们取 out[0].
        eval() 模式下官方行为是禁用 DS, 直接返回单张量.

        Args:
            x: Input [B, C, H, W] (2D) or [B, C, D, H, W] (3D).

        Returns:
            Logits [B, num_classes, H, W] or [B, num_classes, D, H, W].
        """
        out = self.model(x)
        # deep_supervision=True + training → list of multi-scale outputs
        if isinstance(out, (list, tuple)):
            return out[0]
        return out

    def train(self, mode: bool = True):
        """Override train to manage deep supervision state (mirrors official nnUNet)."""
        super().train(mode)
        if self._deep_supervision:
            # 训练模式: 开启深度监督; 评估模式: 关闭 (与官方一致)
            self.set_deep_supervision_enabled(mode)
        return self

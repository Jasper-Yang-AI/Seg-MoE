"""
Lightweight wrapper for nnUNet v2 architecture.

We extract only the model, not the full nnUNet training pipeline.
"""
from __future__ import annotations

from typing import List, Tuple, Union

import torch
import torch.nn as nn


class NnUNetWrapper(nn.Module):
    """Simplified nnUNet architecture wrapper.
    
    This provides a clean interface compatible with the Seg-MoE pipeline,
    while using nnUNet's proven architecture design.
    
    For full nnUNet training pipeline, see: https://github.com/MIC-DKFZ/nnUNet
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        patch_size: Union[Tuple[int, int], Tuple[int, int, int]] = (256, 256),
        n_stages: int = 6,
        features_per_stage: List[int] = None,
        conv_op: str = "Conv2d",
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.n_stages = n_stages
        self.is_3d = len(patch_size) == 3
        
        if features_per_stage is None:
            features_per_stage = [32, 64, 125, 256, 320, 320]
        
        self.features_per_stage = features_per_stage[:n_stages]
        
        # Import nnUNet components
        try:
            from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
            from nnunetv2.nets.UNetEncoder import UNetEncoder
            from nnunetv2.nets.UNetDecoder import UNetDecoder
            from dynamic_network_architectures.architectures.unet import PlainConvUNet
            
            # Use nnUNet's proven PlainConvUNet
            conv_op_class = nn.Conv3d if self.is_3d else nn.Conv2d
            norm_op = nn.InstanceNorm3d if self.is_3d else nn.InstanceNorm2d
            
            self.model = PlainConvUNet(
                input_channels=in_channels,
                n_stages=n_stages,
                features_per_stage=self.features_per_stage,
                conv_op=conv_op_class,
                kernel_sizes=[[3, 3]] * n_stages if not self.is_3d else [[3, 3, 3]] * n_stages,
                strides=[[1, 1]] * (n_stages - 1) + [[2, 2]] if not self.is_3d else [[1, 1, 1]] * (n_stages - 1) + [[2, 2, 2]],
                n_conv_per_stage=[2] * n_stages,
                num_classes=num_classes,
                n_conv_per_stage_decoder=[2] * (n_stages - 1),
                conv_bias=True,
                norm_op=norm_op,
                norm_op_kwargs={'eps': 1e-5, 'affine': True},
                dropout_op=None,
                nonlin=nn.LeakyReLU,
                nonlin_kwargs={'inplace': True},
            )
            
        except ImportError as e:
            # Fallback: simple U-Net if nnUNet not properly installed
            print(f"[nnUNet] Warning: Could not import nnUNetv2 components ({e}), using fallback U-Net")
            self.model = self._build_fallback_unet()
    
    def _build_fallback_unet(self) -> nn.Module:
        """Fallback to basic U-Net if nnUNet not available."""
        try:
            import segmentation_models_pytorch as smp
            return smp.Unet(
                encoder_name="resnet34",
                encoder_weights="imagenet",
                in_channels=self.in_channels,
                classes=self.num_classes,
            )
        except ImportError:
            raise ImportError(
                "Neither nnunetv2 nor segmentation_models_pytorch available. "
                "Install one of them: pip install nnunetv2>=2.2 OR pip install segmentation-models-pytorch"
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor [B, C, H, W] for 2D or [B, C, D, H, W] for 3D
            
        Returns:
            Logits tensor [B, num_classes, H, W] or [B, num_classes, D, H, W]
        """
        return self.model(x)

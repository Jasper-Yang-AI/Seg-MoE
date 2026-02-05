"""
Wrapper for SegMamba (Mamba-based segmentation model).

SegMamba: https://github.com/ge-xing/SegMamba
A linear-complexity model using state space models for efficient segmentation.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def check_segmamba_available() -> bool:
    """Check if SegMamba and its dependencies are available."""
    try:
        import mamba_ssm
        import causal_conv1d
        return True
    except ImportError:
        return False


class SegMambaWrapper(nn.Module):
    """Wrapper for SegMamba model with automatic fallback.
    
    If SegMamba/Mamba dependencies are not available, falls back to
    a lightweight U-Net architecture.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 4,
        img_size: int = 256,
        embed_dim: int = 96,
        depths: tuple = (2, 2, 9, 2),
        drop_path_rate: float = 0.2,
        **kwargs
    ):
        """Initialize SegMamba or fallback model.
        
        Args:
            in_channels: Number of input channels
            num_classes: Number of output classes
            img_size: Input image size
            embed_dim: Embedding dimension
            depths: Depths for each stage
            drop_path_rate: Drop path rate
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.img_size = img_size
        
        # Try to load SegMamba
        if check_segmamba_available():
            try:
                self.model = self._build_segmamba(
                    in_channels, num_classes, img_size, 
                    embed_dim, depths, drop_path_rate, **kwargs
                )
                self.using_fallback = False
                print(f"✓ SegMamba successfully loaded (handles {in_channels}-ch input)")
            except Exception as e:
                print(f"[SegMamba] Failed to load SegMamba: {e}")
                print("[SegMamba] Using fallback model")
                self.model = self._build_fallback(in_channels, num_classes)
                self.using_fallback = True
        else:
            print("[SegMamba] Mamba dependencies not available")
            print("[SegMamba] Using lightweight fallback model (EfficientNet-B0 UNet)")
            self.model = self._build_fallback(in_channels, num_classes)
            self.using_fallback = True
    
    def _build_segmamba(
        self, 
        in_channels: int,
        num_classes: int,
        img_size: int,
        embed_dim: int,
        depths: tuple,
        drop_path_rate: float,
        **kwargs
    ) -> nn.Module:
        """Build SegMamba model.
        
        Note: This is a simplified wrapper. For full SegMamba implementation,
        see: https://github.com/ge-xing/SegMamba
        """
        try:
            # Try to import from installed package
            from segmamba import SegMamba
            return SegMamba(
                in_chans=in_channels,
                out_chans=num_classes,
                img_size=img_size,
                embed_dim=embed_dim,
                depths=depths,
                drop_path_rate=drop_path_rate,
            )
        except ImportError:
            # If not installed as package, try local architecture
            try:
                from seg_moe.models.architectures.segmamba import SegMamba
                return SegMamba(
                    in_chans=in_channels,
                    out_chans=num_classes,
                    img_size=img_size,
                    embed_dim=embed_dim,
                    depths=depths,
                    drop_path_rate=drop_path_rate,
                )
            except ImportError:
                raise ImportError(
                    "SegMamba not found. Please install:\n"
                    "1. pip install causal-conv1d>=1.1.0 mamba-ssm>=1.0.0\n"
                    "2. Add SegMamba architecture to src/seg_moe/models/architectures/\n"
                    "See: https://github.com/ge-xing/SegMamba"
                )
    
    def _build_fallback(self, in_channels: int, num_classes: int) -> nn.Module:
        """Build lightweight fallback model using segmentation_models_pytorch.
        
        Uses EfficientNet-B0 encoder with U-Net decoder as fallback.
        """
        try:
            import segmentation_models_pytorch as smp
            
            # For 1-channel input, we'll use a simple adapter
            if in_channels == 1:
                model = smp.Unet(
                    encoder_name="efficientnet-b0",
                    encoder_weights="imagenet",
                    in_channels=3,  # EfficientNet expects 3 channels
                    classes=num_classes,
                    activation=None,
                )
                
                # Wrap with adapter
                return _ChannelAdapter(model, in_channels=in_channels)
            else:
                return smp.Unet(
                    encoder_name="efficientnet-b0",
                    encoder_weights=None,  # Can't use ImageNet weights for non-3ch
                    in_channels=in_channels,
                    classes=num_classes,
                    activation=None,
                )
        except ImportError:
            raise ImportError(
                "segmentation_models_pytorch required for fallback. "
                "Install with: pip install segmentation-models-pytorch"
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)


class _ChannelAdapter(nn.Module):
    """Adapter to convert 1-channel input to 3-channel for pretrained models."""
    
    def __init__(self, model: nn.Module, in_channels: int = 1):
        super().__init__()
        self.model = model
        self.in_channels = in_channels
        
        if in_channels == 1:
            # Simple channel expansion: repeat grayscale across RGB
            self.adapter = lambda x: x.repeat(1, 3, 1, 1)
        else:
            # For other channel counts, use learnable projection
            self.adapter = nn.Conv2d(in_channels, 3, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.adapter(x)
        return self.model(x)

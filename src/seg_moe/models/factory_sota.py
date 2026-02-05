"""
Factory for building SOTA segmentation models.

Supports:
- Swin-UNetR (Transformer-based, from MONAI)
- nnUNet v2 (Modern CNN, extracted architecture)
- SegMamba (Mamba-based, linear complexity)
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn


def build_sota_model(
    arch: str,
    in_channels: int,
    classes: int,
    config: Optional[Dict[str, Any]] = None,
    pretrained: bool = True,
) -> nn.Module:
    """Build SOTA model with unified interface.
    
    Args:
        arch: Architecture name ('swin_unetr', 'nnunet', 'segmamba')
        in_channels: Number of input channels
        classes: Number of output classes
        config: Architecture-specific config dict
        pretrained: Whether to use pretrained weights
        
    Returns:
        PyTorch model with forward(x) -> logits interface
    """
    arch = arch.lower().replace("-", "_").replace(" ", "_")
    config = config or {}
    
    if arch == "swin_unetr":
        return _build_swin_unetr(in_channels, classes, config, pretrained)
    elif arch == "nnunet" or arch == "nnunet_v2":
        return _build_nnunet(in_channels, classes, config)
    elif arch == "segmamba" or arch == "seg_mamba":
        return _build_segmamba(in_channels, classes, config, pretrained)
    else:
        raise ValueError(f"Unknown SOTA architecture: {arch}")


def _build_swin_unetr(
    in_channels: int,
    classes: int,
    config: Dict[str, Any],
    pretrained: bool = True,
) -> nn.Module:
    """Build Swin-UNetR from MONAI.
    
    Default config optimized for 2D medical images.
    For 3D, set img_size to [D, H, W] and spatial_dims=3.
    """
    try:
        from monai.networks.nets import SwinUNETR
    except ImportError:
        raise ImportError(
            "MONAI not installed. Install with: pip install monai>=1.3.0"
        )
    
    # Default 2D config
    spatial_dims = config.get("spatial_dims", 2)
    img_size = config.get("img_size", [256, 256])
    feature_size = config.get("feature_size", 48)
    depths = config.get("depths", [2, 2, 2, 2])
    num_heads = config.get("num_heads", [3, 6, 12, 24])
    use_checkpoint = config.get("use_checkpoint", False)
    
    # For 2D SwinUNETR, img_size should be tuple without the parameter name in some versions
    # Check MONAI version compatibility
    try:
        model = SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=classes,
            feature_size=feature_size,
            spatial_dims=spatial_dims,
            depths=depths,
            num_heads=num_heads,
            use_checkpoint=use_checkpoint,
        )
    except TypeError:
        # Older MONAI version or 2D compatibility issue
        # Try without img_size parameter
        print("[Swin-UNetR] img_size parameter not supported, using default")
        model = SwinUNETR(
            in_channels=in_channels,
            out_channels=classes,
            # img_size will be inferred from input
            feature_size=feature_size,
            depths=depths,
            num_heads=num_heads,
            use_checkpoint=use_checkpoint,
        )
    
    # Load pretrained weights if available
    if pretrained and config.get("pretrained_path"):
        try:
            state = torch.load(config["pretrained_path"], map_location="cpu")
            model.load_state_dict(state, strict=False)
            print(f"[Swin-UNetR] Loaded pretrained weights from {config['pretrained_path']}")
        except Exception as e:
            print(f"[Swin-UNetR] Warning: Could not load pretrained weights: {e}")
    
    return model


def _build_nnunet(
    in_channels: int,
    classes: int,
    config: Dict[str, Any],
) -> nn.Module:
    """Build nnUNet v2 architecture (lightweight wrapper).
    
    Note: We extract only the model architecture from nnunetv2,
    not the full training framework.
    """
    try:
        from seg_moe.models.wrappers.nnunet_wrapper import NnUNetWrapper
    except ImportError:
        raise ImportError(
            "nnUNet wrapper not found. Make sure nnunetv2 is installed: "
            "pip install nnunetv2>=2.2"
        )
    
    # Default config
    patch_size = config.get("patch_size", [256, 256])
    n_stages = config.get("n_stages", 6)
    features_per_stage = config.get("features_per_stage", [32, 64, 125, 256, 320, 320])
    conv_op = config.get("conv_op", "Conv2d")  # Conv2d or Conv3d
    
    model = NnUNetWrapper(
        in_channels=in_channels,
        num_classes=classes,
        patch_size=patch_size,
        n_stages=n_stages,
        features_per_stage=features_per_stage,
        conv_op=conv_op,
    )
    
    return model


def _build_segmamba(
    in_channels: int,
    classes: int,
    config: Dict[str, Any],
    pretrained: bool = True,
) -> nn.Module:
    """Build SegMamba (Mamba-based segmentation).
    
    SegMamba offers linear complexity for efficient segmentation.
    Requires: mamba-ssm and causal-conv1d
    Installation:
        pip install causal-conv1d>=1.1.0
        pip install mamba-ssm>=1.0.0
    """
    try:
        from seg_moe.models.wrappers.segmamba_wrapper import SegMambaWrapper
    except ImportError:
        raise ImportError(
            "SegMamba wrapper not found. This requires:\n"
            "1. Install dependencies: pip install causal-conv1d mamba-ssm\n"
            "2. (Optional) Add SegMamba architecture to src/seg_moe/models/architectures/\n"
            "See: https://github.com/ge-xing/SegMamba"
        )
    
    # Default config
    img_size = config.get("img_size", 256)
    embed_dim = config.get("embed_dim", 96)
    depths = config.get("depths", (2, 2, 9, 2))
    drop_path_rate = config.get("drop_path_rate", 0.2)
    
    model = SegMambaWrapper(
        in_channels=in_channels,
        num_classes=classes,
        img_size=img_size,
        embed_dim=embed_dim,
        depths=depths,
        drop_path_rate=drop_path_rate,
    )
    
    if pretrained and config.get("pretrained_path"):
        try:
            state = torch.load(config["pretrained_path"], map_location="cpu")
            model.load_state_dict(state, strict=False)
            print(f"[SegMamba] Loaded pretrained weights from {config['pretrained_path']}")
        except Exception as e:
            print(f"[SegMamba] Warning: Could not load pretrained weights: {e}")
    
    return model


def list_sota_experts(models_cfg: Dict[str, Any]) -> list[tuple[str, dict]]:
    """List SOTA expert configurations.
    
    Returns:
        List of (architecture_name, config_dict) tuples
    """
    if "sota_experts" not in models_cfg:
        return []
    
    experts = []
    for expert_cfg in models_cfg["sota_experts"]:
        arch = expert_cfg["architecture"]
        config = expert_cfg.get("config", {})
        experts.append((arch, config))
    
    return experts


def expert_name_sota(arch: str, variant: str = "") -> str:
    """Generate consistent naming for SOTA experts."""
    name = arch.lower().replace("_", "-")
    if variant:
        name = f"{name}-{variant}"
    return name

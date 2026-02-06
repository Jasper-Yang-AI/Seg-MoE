"""
Factory for building SOTA segmentation models.

Supports:
- Swin-UNetR (Transformer-based, from MONAI)
- nnUNet v2 (Modern CNN, extracted architecture)
- SegResNet (3D residual encoder-decoder, from MONAI)
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
        arch: Architecture name ('swin_unetr', 'nnunet', 'segresnet')
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
    elif arch in ("nnunet", "nnunet_v2"):
        return _build_nnunet(in_channels, classes, config)
    elif arch in ("segresnet", "seg_resnet"):
        return _build_segresnet(in_channels, classes, config, pretrained)
    else:
        raise ValueError(f"Unknown SOTA architecture: {arch}")


# ------------------------------------------------------------------
#  Swin-UNetR
# ------------------------------------------------------------------

def _build_swin_unetr(
    in_channels: int,
    classes: int,
    config: Dict[str, Any],
    pretrained: bool = True,
) -> nn.Module:
    """Build Swin-UNetR from MONAI.

    Default config optimized for 3D medical volumes.
    """
    try:
        from monai.networks.nets import SwinUNETR
    except ImportError:
        raise ImportError("MONAI not installed. Install with: pip install monai>=1.3.0")

    spatial_dims = config.get("spatial_dims", 3)
    feature_size = config.get("feature_size", 48)
    depths = config.get("depths", [2, 2, 2, 2])
    num_heads = config.get("num_heads", [3, 6, 12, 24])
    use_checkpoint = config.get("use_checkpoint", False)

    model = SwinUNETR(
        in_channels=in_channels,
        out_channels=classes,
        feature_size=feature_size,
        spatial_dims=spatial_dims,
        depths=depths,
        num_heads=num_heads,
        use_checkpoint=use_checkpoint,
    )

    if pretrained and config.get("pretrained_path"):
        try:
            state = torch.load(config["pretrained_path"], map_location="cpu", weights_only=True)
            model.load_state_dict(state, strict=False)
            print(f"[Swin-UNetR] Loaded pretrained weights from {config['pretrained_path']}")
        except Exception as e:
            print(f"[Swin-UNetR] Warning: Could not load pretrained weights: {e}")

    return model


# ------------------------------------------------------------------
#  nnUNet v2
# ------------------------------------------------------------------

def _build_nnunet(
    in_channels: int,
    classes: int,
    config: Dict[str, Any],
) -> nn.Module:
    """Build nnUNet v2 architecture (lightweight wrapper)."""
    try:
        from seg_moe.models.wrappers.nnunet_wrapper import NnUNetWrapper
    except ImportError:
        raise ImportError(
            "nnUNet wrapper not found. Make sure nnunetv2 is installed: "
            "pip install nnunetv2>=2.2"
        )

    patch_size = config.get("patch_size", [96, 96, 96])
    n_stages = config.get("n_stages", 6)
    features_per_stage = config.get("features_per_stage", [32, 64, 128, 256, 320, 320])
    conv_op = config.get("conv_op", "Conv3d")

    model = NnUNetWrapper(
        in_channels=in_channels,
        num_classes=classes,
        patch_size=patch_size,
        n_stages=n_stages,
        features_per_stage=features_per_stage,
        conv_op=conv_op,
    )
    return model


# ------------------------------------------------------------------
#  SegResNet (MONAI) — 替代原 SegMamba
# ------------------------------------------------------------------

def _build_segresnet(
    in_channels: int,
    classes: int,
    config: Dict[str, Any],
    pretrained: bool = True,
) -> nn.Module:
    """Build SegResNet (MONAI).

    SegResNet is a 3D encoder-decoder with residual blocks, lightweight,
    and natively available in MONAI without external CUDA extensions.

    Ref: Myronenko (2019) "3D MRI brain tumor segmentation using autoencoder
    regularization", MICCAI BraTS challenge winner.
    """
    try:
        from monai.networks.nets import SegResNet
    except ImportError:
        raise ImportError("MONAI not installed. Install with: pip install monai>=1.3.0")

    spatial_dims = config.get("spatial_dims", 3)
    init_filters = config.get("init_filters", 32)
    blocks_down = config.get("blocks_down", [1, 2, 2, 4])
    blocks_up = config.get("blocks_up", [1, 1, 1])
    dropout_prob = config.get("dropout_prob", 0.2)

    model = SegResNet(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=classes,
        init_filters=init_filters,
        blocks_down=blocks_down,
        blocks_up=blocks_up,
        dropout_prob=dropout_prob,
    )

    if pretrained and config.get("pretrained_path"):
        try:
            state = torch.load(config["pretrained_path"], map_location="cpu", weights_only=True)
            model.load_state_dict(state, strict=False)
            print(f"[SegResNet] Loaded pretrained weights from {config['pretrained_path']}")
        except Exception as e:
            print(f"[SegResNet] Warning: Could not load pretrained weights: {e}")

    return model


# ------------------------------------------------------------------
#  Listing & naming helpers
# ------------------------------------------------------------------

def list_sota_experts(models_cfg: Dict[str, Any]) -> list[tuple[str, dict]]:
    """List SOTA expert configurations.

    Returns:
        List of (architecture_name, full_expert_cfg_dict) tuples.
        Each expert_cfg contains: architecture, name, enabled, config, etc.
    """
    if "sota_experts" not in models_cfg:
        return []

    experts = []
    for expert_cfg in models_cfg["sota_experts"]:
        arch = expert_cfg["architecture"]
        experts.append((arch, expert_cfg))

    return experts


def expert_name_sota(arch: str, variant: str = "") -> str:
    """Generate consistent naming for SOTA experts."""
    name = arch.lower().replace("_", "-")
    if variant:
        name = f"{name}-{variant}"
    return name

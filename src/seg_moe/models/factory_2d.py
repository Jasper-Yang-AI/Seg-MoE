from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
#  MONAI 2D model builders
# ---------------------------------------------------------------------------

def _build_monai_swin_unetr_2d(in_channels: int, classes: int, params: Dict[str, Any]) -> nn.Module:
    """Build MONAI Swin-UNetR with spatial_dims=2."""
    try:
        from monai.networks.nets import SwinUNETR
    except ImportError:
        raise ImportError("MONAI not installed. Install with: pip install monai>=1.3.0")

    # Build kwargs, only pass parameters that SwinUNETR actually accepts
    import inspect
    valid_params = set(inspect.signature(SwinUNETR.__init__).parameters.keys())

    kwargs = {
        "in_channels": in_channels,
        "out_channels": classes,
        "spatial_dims": params.get("spatial_dims", 2),
        "feature_size": params.get("feature_size", 48),
        "depths": params.get("depths", [2, 2, 2, 2]),
        "num_heads": params.get("num_heads", [3, 6, 12, 24]),
        "use_checkpoint": params.get("use_checkpoint", False),
    }

    # img_size exists in older MONAI, removed in newer versions
    if "img_size" in valid_params and "img_size" in params:
        kwargs["img_size"] = tuple(params["img_size"])

    model = SwinUNETR(**kwargs)

    if params.get("pretrained_path"):
        try:
            state = torch.load(params["pretrained_path"], map_location="cpu", weights_only=True)
            model.load_state_dict(state, strict=False)
            print(f"[Swin-UNetR-2D] Loaded pretrained weights from {params['pretrained_path']}")
        except Exception as e:
            print(f"[Swin-UNetR-2D] Warning: Could not load pretrained weights: {e}")

    return model


def _build_monai_segresnet_2d(in_channels: int, classes: int, params: Dict[str, Any]) -> nn.Module:
    """Build MONAI SegResNet with spatial_dims=2."""
    try:
        from monai.networks.nets import SegResNet
    except ImportError:
        raise ImportError("MONAI not installed. Install with: pip install monai>=1.3.0")

    model = SegResNet(
        spatial_dims=params.get("spatial_dims", 2),
        in_channels=in_channels,
        out_channels=classes,
        init_filters=params.get("init_filters", 32),
        blocks_down=params.get("blocks_down", [1, 2, 2, 4]),
        blocks_up=params.get("blocks_up", [1, 1, 1]),
        dropout_prob=params.get("dropout_prob", 0.2),
    )

    if params.get("pretrained_path"):
        try:
            state = torch.load(params["pretrained_path"], map_location="cpu", weights_only=True)
            model.load_state_dict(state, strict=False)
            print(f"[SegResNet-2D] Loaded pretrained weights from {params['pretrained_path']}")
        except Exception as e:
            print(f"[SegResNet-2D] Warning: Could not load pretrained weights: {e}")

    return model


# ---------------------------------------------------------------------------
#  nnUNet 2D builder
# ---------------------------------------------------------------------------

def _build_nnunet_2d(in_channels: int, classes: int, params: Dict[str, Any]) -> nn.Module:
    """Build nnUNet v2 (PlainConvUNet) for 2D segmentation.

    Supports both custom training (deep_supervision=False) and
    loading official nnUNet weights (deep_supervision=True).
    """
    from seg_moe.models.wrappers.nnunet_wrapper import NnUNetWrapper

    patch_size = tuple(params.get("patch_size", [256, 256]))
    n_stages = params.get("n_stages", 6)
    features = params.get("features_per_stage", [32, 64, 128, 256, 320, 320])
    conv_op = params.get("conv_op", "Conv2d")
    deep_supervision = params.get("deep_supervision", False)

    wrapper = NnUNetWrapper(
        in_channels=in_channels,
        num_classes=classes,
        patch_size=patch_size,
        n_stages=n_stages,
        features_per_stage=features,
        conv_op=conv_op,
        deep_supervision=deep_supervision,
        n_conv_per_stage_encoder=params.get("n_conv_per_stage_encoder"),
        n_conv_per_stage_decoder=params.get("n_conv_per_stage_decoder"),
        kernel_sizes=params.get("conv_kernel_sizes") or params.get("kernel_sizes"),
        strides=params.get("pool_op_kernel_sizes") or params.get("strides"),
    )

    # Load pretrained weights (e.g., imported from official nnUNet)
    pretrained_path = params.get("pretrained_path")
    if pretrained_path:
        try:
            state = torch.load(pretrained_path, map_location="cpu", weights_only=True)
            # Seg-MoE checkpoint format: {"model": state_dict, ...}
            if "model" in state:
                wrapper.load_state_dict(state["model"], strict=False)
                print(f"[nnUNet-2D] Loaded pretrained weights from {pretrained_path}")
            else:
                wrapper.load_state_dict(state, strict=False)
                print(f"[nnUNet-2D] Loaded raw state_dict from {pretrained_path}")
        except Exception as e:
            print(f"[nnUNet-2D] Warning: Could not load pretrained weights: {e}")

    return wrapper


# ---------------------------------------------------------------------------
#  统一 experts_v2 接口 (models.yaml 格式)
# ---------------------------------------------------------------------------

def list_experts(models_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return list of expert config dicts from experts_v2 format.

    Each dict has keys: name, type, params.
    """
    return list(models_cfg.get("experts_v2", []))


def build_expert(
    expert_cfg: Dict[str, Any],
    in_channels: int,
    num_classes: int,
) -> nn.Module:
    """Build a single expert model from a config dict.

    Args:
        expert_cfg: dict with keys {name, type, params}
        in_channels: number of input channels
        num_classes: number of output classes (M)

    Returns:
        nn.Module with forward(x) -> logits [B, M, H, W]
    """
    etype = expert_cfg["type"].lower()
    params = expert_cfg.get("params", {})

    if etype == "nnunet":
        return _build_nnunet_2d(in_channels, num_classes, params)

    if etype == "monai_swin_unetr":
        return _build_monai_swin_unetr_2d(in_channels, num_classes, params)

    if etype == "monai_segresnet":
        return _build_monai_segresnet_2d(in_channels, num_classes, params)

    raise ValueError(f"Unknown expert type: {etype}. Supported: nnunet, monai_swin_unetr, monai_segresnet")


def expert_name(expert_cfg: Dict[str, Any]) -> str:
    """Return the expert name from config dict."""
    return expert_cfg["name"]




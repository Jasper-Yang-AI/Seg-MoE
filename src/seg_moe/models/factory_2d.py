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


def _build_monai_segresnet_ds_2d(in_channels: int, classes: int, params: Dict[str, Any]) -> nn.Module:
    """Build MONAI SegResNetDS with spatial_dims=2.

    Official MONAI Auto3DSeg recipe:
        SegResNetDS(init_filters=32, blocks_down=[1,2,2,4,4],
                    norm=BATCH, dsdepth=2, upsample_mode=deconv)

    For Seg-MoE inference, dsdepth should be 1 (single output).
    Training uses dsdepth=2 (deep supervision); import_segresnet_weights.py
    handles the extra deep-supervision head keys via strict=False.
    """
    try:
        from monai.networks.nets import SegResNetDS
    except ImportError:
        raise ImportError("MONAI >= 1.5.0 required for SegResNetDS. pip install monai>=1.5.0")

    model = SegResNetDS(
        spatial_dims=params.get("spatial_dims", 2),
        in_channels=in_channels,
        out_channels=classes,
        init_filters=params.get("init_filters", 32),
        blocks_down=params.get("blocks_down", [1, 2, 2, 4, 4]),
        norm=params.get("norm", "BATCH"),
        act=params.get("act", "relu"),
        dsdepth=params.get("dsdepth", 1),
        upsample_mode=params.get("upsample_mode", "deconv"),
    )

    if params.get("pretrained_path"):
        try:
            state = torch.load(params["pretrained_path"], map_location="cpu", weights_only=True)
            model.load_state_dict(state, strict=False)
            print(f"[SegResNetDS-2D] Loaded pretrained weights from {params['pretrained_path']}")
        except Exception as e:
            print(f"[SegResNetDS-2D] Warning: Could not load pretrained weights: {e}")

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

    if etype == "monai_segresnet_ds":
        return _build_monai_segresnet_ds_2d(in_channels, num_classes, params)

    raise ValueError(
        f"Unknown expert type: {etype}. "
        "Supported: nnunet, monai_swin_unetr, monai_segresnet, monai_segresnet_ds"
    )


def expert_name(expert_cfg: Dict[str, Any]) -> str:
    """Return the expert name from config dict."""
    return expert_cfg["name"]


# ---------------------------------------------------------------------------
#  Layer2 权重初始化: 从 Layer1 迁移权重 (B1)
# ---------------------------------------------------------------------------

def transfer_layer1_to_layer2(
    layer1_model: nn.Module,
    layer2_model: nn.Module,
    base_in_channels: int = 3,
    extra_in_channels: int = 9,
) -> nn.Module:
    """Transfer Layer1 weights to Layer2 model.

    Layer2 has a wider first conv (in_channels = base + extra).
    Strategy:
      1. Copy ALL shared weights from Layer1 to Layer2 (strict=False)
      2. For the first conv layer: copy the base_in_channels weights,
         initialize the extra channels with zeros (identity-like init)

    This allows Layer2 to start from a model that already segments well
    and only needs to learn how to use the probability channels.

    Args:
        layer1_model: Trained Layer1 model (in_channels = base_in_channels)
        layer2_model: Fresh Layer2 model (in_channels = base + extra)
        base_in_channels: Image channels (e.g. 3 for RGB)
        extra_in_channels: Probability channels (e.g. K*M = 9)

    Returns:
        layer2_model with transferred weights
    """
    l1_sd = layer1_model.state_dict()
    l2_sd = layer2_model.state_dict()

    transferred = 0
    stem_transferred = False

    for key in l2_sd:
        if key not in l1_sd:
            continue
        l1_param = l1_sd[key]
        l2_param = l2_sd[key]

        if l1_param.shape == l2_param.shape:
            # Same shape: direct copy
            l2_sd[key] = l1_param.clone()
            transferred += 1
        elif l1_param.ndim >= 2 and l2_param.ndim >= 2 and l1_param.shape[0] == l2_param.shape[0]:
            # First conv: out_channels match, in_channels differ
            # L1: [out, base_in, kH, kW], L2: [out, base_in + extra, kH, kW]
            if l1_param.shape[1] == base_in_channels and l2_param.shape[1] == base_in_channels + extra_in_channels:
                # Copy base channels, zero-init extra channels
                l2_sd[key] = torch.zeros_like(l2_param)
                l2_sd[key][:, :base_in_channels] = l1_param.clone()
                stem_transferred = True
                transferred += 1

    layer2_model.load_state_dict(l2_sd)

    total = len(l2_sd)
    print(f"[Layer2 init] Transferred {transferred}/{total} params from Layer1"
          f" (stem={'yes' if stem_transferred else 'no'})")
    return layer2_model




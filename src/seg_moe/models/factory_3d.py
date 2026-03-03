"""
3D model factory — mirrors factory_2d.py interface.

Supports config-driven construction of the three 3D experts:
  segresnet   → MONAI SegResNet (spatial_dims=3)
  swin_unetr  → MONAI SwinUNETR (spatial_dims=3)
  nnunet      → PlainConvUNet3D (dynamic_network_architectures)

Usage:
    from seg_moe.models.factory_3d import build_expert_3d, list_experts_3d, expert_name_3d
    experts = list_experts_3d(models_cfg)
    model = build_expert_3d(experts[0], in_channels=3, num_classes=4)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# MONAI SegResNet 3D
# ---------------------------------------------------------------------------

def _build_segresnet_3d(in_channels: int, classes: int, params: Dict[str, Any]) -> nn.Module:
    try:
        from monai.networks.nets import SegResNet
    except ImportError:
        raise ImportError("MONAI required: pip install monai>=1.3")

    model = SegResNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=classes,
        init_filters=params.get("init_filters", 32),
        blocks_down=params.get("blocks_down", [1, 2, 2, 4]),
        blocks_up=params.get("blocks_up", [1, 1, 1]),
        dropout_prob=params.get("dropout_prob", 0.2),
    )
    if params.get("pretrained_path"):
        _load_weights(model, params["pretrained_path"], tag="SegResNet-3D")
    return model


# ---------------------------------------------------------------------------
# MONAI SwinUNETR 3D
# ---------------------------------------------------------------------------

def _build_swinunetr_3d(in_channels: int, classes: int, params: Dict[str, Any]) -> nn.Module:
    try:
        from monai.networks.nets import SwinUNETR
        import inspect
        valid = set(inspect.signature(SwinUNETR.__init__).parameters.keys())
    except ImportError:
        raise ImportError("MONAI required: pip install monai>=1.3")

    kwargs: Dict[str, Any] = {
        "in_channels":    in_channels,
        "out_channels":   classes,
        "spatial_dims":   3,
        "feature_size":   params.get("feature_size", 48),
        "depths":         params.get("depths", [2, 2, 2, 2]),
        "num_heads":      params.get("num_heads", [3, 6, 12, 24]),
        "use_checkpoint": params.get("use_checkpoint", True),
    }
    if "patch_size" in valid:
        kwargs["patch_size"] = params.get("patch_size", 2)

    model = SwinUNETR(**kwargs)
    if params.get("pretrained_path"):
        _load_weights(model, params["pretrained_path"], tag="SwinUNETR-3D")
    return model


# ---------------------------------------------------------------------------
# nnUNet PlainConvUNet 3D
# ---------------------------------------------------------------------------

def _build_nnunet_3d(in_channels: int, classes: int, params: Dict[str, Any]) -> nn.Module:
    try:
        from dynamic_network_architectures.architectures.unet import PlainConvUNet
    except ImportError:
        raise ImportError(
            "dynamic_network_architectures required. "
            "pip install git+https://github.com/MIC-DKFZ/dynamic-network-architectures"
        )

    n_stages = int(params.get("n_stages", 5))
    features = list(params.get("features_per_stage", [32, 64, 128, 256, 320]))[:n_stages]
    n_enc    = params.get("n_conv_per_stage_encoder") or [2] * n_stages
    n_dec    = params.get("n_conv_per_stage_decoder") or [2] * (n_stages - 1)

    ks = [[3, 3, 3]] * n_stages
    st = [[1, 1, 1]] + [[2, 2, 2]] * (n_stages - 1)

    model = PlainConvUNet(
        input_channels=in_channels,
        n_stages=n_stages,
        features_per_stage=features,
        conv_op=nn.Conv3d,
        kernel_sizes=ks,
        strides=st,
        n_conv_per_stage=n_enc,
        num_classes=classes,
        n_conv_per_stage_decoder=n_dec,
        conv_bias=True,
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        dropout_op=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
        deep_supervision=bool(params.get("deep_supervision", False)),
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[nnUNet-3D] PlainConvUNet {n_stages} stages, {n_params:,} params")

    if params.get("pretrained_path"):
        _load_weights(model, params["pretrained_path"], tag="nnUNet-3D")
    return model


# ---------------------------------------------------------------------------
# Weight loading helper
# ---------------------------------------------------------------------------

def _load_weights(model: nn.Module, path: str, tag: str = "") -> None:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
        if "model" in state:
            state = state["model"]
        # Strip module. prefix
        state = {k.removeprefix("module."): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
        print(f"[{tag}] Loaded pretrained from {path}")
    except Exception as e:
        print(f"[{tag}] Warning: could not load weights from {path}: {e}")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_BUILDERS = {
    "segresnet":   _build_segresnet_3d,
    "seg_resnet":  _build_segresnet_3d,
    "swin_unetr":  _build_swinunetr_3d,
    "swinunetr":   _build_swinunetr_3d,
    "nnunet":      _build_nnunet_3d,
    "nnunet_v2":   _build_nnunet_3d,
}


# ---------------------------------------------------------------------------
# Public API (mirrors factory_2d.py)
# ---------------------------------------------------------------------------

def list_experts_3d(models_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return list of enabled expert config dicts from models_3d.yaml."""
    return [e for e in models_cfg.get("experts_3d", []) if e.get("enabled", True) is not False]


def expert_name_3d(ec: Dict[str, Any]) -> str:
    return str(ec["name"])


def build_expert_3d(
    ec: Dict[str, Any],
    in_channels: int = 1,
    num_classes: int = 4,
) -> nn.Module:
    """Build a 3D expert from config dict entry."""
    arch = str(ec["type"]).lower().replace("-", "_")
    builder = _BUILDERS.get(arch)
    if builder is None:
        raise ValueError(f"Unknown 3D expert type '{arch}'. Available: {sorted(_BUILDERS)}")
    params = dict(ec.get("params", {}))
    return builder(in_channels=in_channels, classes=num_classes, params=params)


def transfer_layer1_to_layer2_3d(
    l1_model: nn.Module,
    l2_model: nn.Module,
    base_in_channels: int,
    extra_in_channels: int,
) -> None:
    """Transfer Layer1 weights to Layer2, re-initialising the input stem.

    Copies all parameters except the first convolutional layer (which must
    handle extra input channels). The stem weights for the original channels
    are copied; extra channels are Xavier-initialised.
    """
    l1_sd = l1_model.state_dict()
    l2_sd = l2_model.state_dict()

    new_sd: Dict[str, torch.Tensor] = {}
    for k, v2 in l2_sd.items():
        if k not in l1_sd:
            new_sd[k] = v2
            continue
        v1 = l1_sd[k]
        if v1.shape == v2.shape:
            new_sd[k] = v1.clone()
        elif v2.dim() == 5 and v2.shape[1] != v1.shape[1]:
            # Conv weight: [Out, In, kD, kH, kW]
            # Copy base channels; Xavier-init extra channels
            out, total_in, *ks = v2.shape
            base = min(base_in_channels, v1.shape[1])
            w = torch.zeros_like(v2)
            w[:, :base] = v1[:, :base] if v1.shape[1] >= base else v1
            # Xavier for extra channels
            nn.init.xavier_uniform_(w[:, base:].reshape(out, -1).unsqueeze(-1).unsqueeze(-1))
            new_sd[k] = w
        else:
            new_sd[k] = v2  # shape mismatch: keep random init

    l2_model.load_state_dict(new_sd, strict=True)
    print(f"[transfer_layer1_to_layer2_3d] Transferred {sum(1 for k in new_sd if k in l1_sd)} param groups")

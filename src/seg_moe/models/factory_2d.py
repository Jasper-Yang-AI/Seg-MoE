from __future__ import annotations

from typing import Any, Dict

import torch


def build_smp_model(arch: str, encoder: str, in_channels: int, classes: int, encoder_weights: str | None) -> torch.nn.Module:
    import segmentation_models_pytorch as smp

    arch = arch.lower()
    encoder = encoder.lower()

    if arch == "unet":
        return smp.Unet(encoder_name=encoder, encoder_weights=encoder_weights, in_channels=in_channels, classes=classes)
    if arch == "linknet":
        return smp.Linknet(encoder_name=encoder, encoder_weights=encoder_weights, in_channels=in_channels, classes=classes)
    if arch == "fpn":
        return smp.FPN(encoder_name=encoder, encoder_weights=encoder_weights, in_channels=in_channels, classes=classes)
    if arch in ("unetplusplus", "unet++"):
        return smp.UnetPlusPlus(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )
    raise ValueError(f"Unknown architecture: {arch}")


def list_experts(models_cfg: Dict[str, Any]) -> list[tuple[str, str]]:
    arches = models_cfg["experts"]["architectures"]
    backs = models_cfg["experts"]["backbones"]
    experts: list[tuple[str, str]] = []
    for a in arches:
        for b in backs:
            experts.append((a, b))
    return experts


def expert_name(arch: str, backbone: str) -> str:
    return f"{arch}-{backbone}".lower()

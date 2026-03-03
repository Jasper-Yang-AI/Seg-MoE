"""
3D Patch-level Convolutional Gating Network for dynamic expert fusion.

Architecture mirrors patch_gating_2d.py but with 3D convolutions.

Design:
  Input:  Expert logit patches  [B, K*M, pD, pH, pW]
  Arch:   Shared 3D ConvNet backbone + Residual FC head (~40-80 K params)
  Output: Softmax gating weights [B, K] or [B, K, M] (per-class)
  Fusion: fused_logits = Σ_k w_k · logits_k  →  CE/Dice loss

Key features:
  - 3D convolutions for volumetric context
  - Temperature-annealed softmax (τ 2.0 → 0.5)
  - Load balancing regularization (prevent expert collapse)
  - Spatial smoothness regularization (TV-norm)
  - Gaussian-blended 3D sliding window inference
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Legacy alias (keeps old import paths working)
# ---------------------------------------------------------------------------

@dataclass
class PatchGating3DConfig:
    """Legacy dataclass — kept for backward compatibility."""
    num_experts: int
    num_classes: int
    per_class: bool = False


# ---------------------------------------------------------------------------
# New config dataclass
# ---------------------------------------------------------------------------

@dataclass
class PatchGatingConfig3D:
    """Configuration for the 3D patch-level gating network."""

    num_experts: int = 3
    num_classes: int = 4

    # Patch parameters (D, H, W)
    patch_size: Tuple[int, int, int] = (32, 32, 16)
    stride: Tuple[int, int, int] = (16, 16, 8)
    blend_mode: str = "gaussian"

    # Network architecture
    hidden_dim: int = 64
    dropout: float = 0.1
    per_class: bool = False
    use_residual_head: bool = True

    # Temperature annealing
    temperature_start: float = 2.0
    temperature_end: float = 0.5

    # Regularization
    load_balance_weight: float = 0.01
    spatial_smooth_weight: float = 0.0

    @property
    def in_channels(self) -> int:
        return self.num_experts * self.num_classes


# ---------------------------------------------------------------------------
# Temperature schedule
# ---------------------------------------------------------------------------

def compute_temperature_3d(
    epoch: int,
    max_epochs: int,
    t_start: float = 2.0,
    t_end: float = 0.5,
) -> float:
    """Exponential temperature annealing."""
    if max_epochs <= 1:
        return t_end
    ratio = epoch / (max_epochs - 1)
    return t_start * (t_end / t_start) ** ratio


# ---------------------------------------------------------------------------
# Regularization losses
# ---------------------------------------------------------------------------

def compute_load_balance_loss_3d(weights: torch.Tensor) -> torch.Tensor:
    """Load-balancing loss: K · Σ_k f_k²  (Shazeer 2017)."""
    K = weights.shape[1]
    usage = weights.mean(dim=0)
    return K * (usage ** 2).sum()


def compute_spatial_smooth_loss_3d(weights: torch.Tensor) -> torch.Tensor:
    """TV smoothness on sequential patch weight map."""
    if weights.shape[0] < 2:
        return weights.new_tensor(0.0)
    return (weights[1:] - weights[:-1]).abs().mean()


# ---------------------------------------------------------------------------
# Network building blocks
# ---------------------------------------------------------------------------

class _SharedBackbone3D(nn.Module):
    """3-layer 3D Conv + BN + GELU backbone → [B, h]."""

    def __init__(self, in_ch: int, hidden_dim: int) -> None:
        super().__init__()
        h = hidden_dim
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, h, 3, padding=1, bias=False),
            nn.BatchNorm3d(h),
            nn.GELU(),
            nn.Conv3d(h, h, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(h),
            nn.GELU(),
            nn.Conv3d(h, h, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(h),
            nn.GELU(),
            nn.AdaptiveAvgPool3d(1),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).flatten(1)   # [B, h]


class _ResidualHead(nn.Module):
    """FC head with residual shortcut."""

    def __init__(self, hidden_dim: int, out_dim: int, dropout: float = 0.1, use_residual: bool = True) -> None:
        super().__init__()
        h = hidden_dim
        self.use_residual = use_residual
        self.main = nn.Sequential(
            nn.Linear(h, h // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(h // 2, out_dim),
        )
        if use_residual:
            self.skip = nn.Linear(h, out_dim, bias=False)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        out = self.main(feat)
        if self.use_residual:
            out = out + self.skip(feat)
        return out


# ---------------------------------------------------------------------------
# Main gating network
# ---------------------------------------------------------------------------

class PatchConvGate3D(nn.Module):
    """3D Patch-level convolutional gating network.

    Input:  [B, K*M, pD, pH, pW]  — stacked expert logit patches
    Output: [B, K] softmax weights (or [B, K, M] if per_class=True)
    """

    def __init__(self, cfg: PatchGatingConfig3D) -> None:
        super().__init__()
        self.cfg = cfg
        K = cfg.num_experts
        M = cfg.num_classes
        h = cfg.hidden_dim
        out_dim = K * M if cfg.per_class else K

        self.backbone = _SharedBackbone3D(in_ch=K * M, hidden_dim=h)
        self.head = _ResidualHead(h, out_dim, cfg.dropout, cfg.use_residual_head)

    def forward(self, logit_flat: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Args:
            logit_flat: [B, K*M, pD, pH, pW]
            temperature: τ for softmax sharpness
        Returns:
            weights: [B, K] or [B, K, M]
        """
        B = logit_flat.shape[0]
        feat = self.backbone(logit_flat)    # [B, h]
        raw = self.head(feat)               # [B, K] or [B, K*M]

        if self.cfg.per_class:
            raw = raw.view(B, self.cfg.num_experts, self.cfg.num_classes)
            return F.softmax(raw / temperature, dim=1)   # [B, K, M]
        else:
            return F.softmax(raw / temperature, dim=1)   # [B, K]

    def fuse_logits(self, logits: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  [B, K, M, pD, pH, pW]
            weights: [B, K] or [B, K, M]
        Returns:
            fused:   [B, M, pD, pH, pW]
        """
        if weights.dim() == 2:
            w = weights[:, :, None, None, None, None]
        else:
            w = weights[:, :, :, None, None, None]
        return (logits * w).sum(dim=1)

    def weights_per_expert(self, weights: torch.Tensor) -> torch.Tensor:
        """Always return [B, K]."""
        if weights.dim() == 3:
            return weights.mean(dim=2)
        return weights


# ---------------------------------------------------------------------------
# Legacy stub (kept to avoid import errors from old code)
# ---------------------------------------------------------------------------

class PatchGating3D(nn.Module):
    """Backward-compat alias — use PatchConvGate3D for new code."""

    def __init__(self, cfg: PatchGating3DConfig):
        super().__init__()
        self.cfg = cfg

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "PatchGating3D is the legacy stub. Use PatchConvGate3D instead."
        )


# ---------------------------------------------------------------------------
# 3D Gaussian importance map
# ---------------------------------------------------------------------------

def _gaussian_kernel_3d(patch_size: Tuple[int, int, int], sigma_ratio: float = 0.125) -> torch.Tensor:
    pd, ph, pw = patch_size
    grids = []
    for size in (pd, ph, pw):
        sigma = size * sigma_ratio
        coords = torch.arange(size).float() - size // 2
        grids.append(torch.exp(-(coords ** 2) / (2 * sigma ** 2)))
    d_k, h_k, w_k = grids
    kernel = d_k[:, None, None] * h_k[None, :, None] * w_k[None, None, :]
    return (kernel / kernel.max()).clamp(min=0.01)


# ---------------------------------------------------------------------------
# Volume-level sliding-window fusion
# ---------------------------------------------------------------------------

def fuse_volume_sliding_window(
    model: PatchConvGate3D,
    logits_vol: torch.Tensor,
    patch_size: Tuple[int, int, int],
    stride: Tuple[int, int, int],
    num_classes: int,
    num_experts: int,
    temperature: float = 1.0,
    device: Optional[torch.device] = None,
    blend_mode: str = "gaussian",
) -> torch.Tensor:
    """Apply gating over a full volume using 3D sliding-window.

    Args:
        logits_vol: [K, M, D, H, W]
    Returns:
        fused: [M, D, H, W]
    """
    if device is None:
        device = next(model.parameters()).device

    K, M, D, H, W = logits_vol.shape
    pd, ph, pw = patch_size

    from seg_moe.data.gating_patch_dataset_3d import compute_patch_positions_3d
    positions = compute_patch_positions_3d((D, H, W), patch_size, stride)

    fused_vol = torch.zeros(M, D, H, W, device="cpu")
    weight_map = torch.zeros(1, D, H, W, device="cpu")
    importance = _gaussian_kernel_3d(patch_size) if blend_mode == "gaussian" else torch.ones(pd, ph, pw)

    model.eval()
    with torch.no_grad():
        for (d0, h0, w0) in positions:
            lp = logits_vol[:, :, d0:d0+pd, h0:h0+ph, w0:w0+pw]
            lp_in = lp.reshape(1, K * M, pd, ph, pw).to(device)
            w = model(lp_in, temperature=temperature)
            fp = model.fuse_logits(lp.unsqueeze(0).to(device), w).squeeze(0).cpu()
            fused_vol[:, d0:d0+pd, h0:h0+ph, w0:w0+pw] += fp * importance
            weight_map[:, d0:d0+pd, h0:h0+ph, w0:w0+pw] += importance

    return fused_vol / weight_map.clamp(min=1e-7)

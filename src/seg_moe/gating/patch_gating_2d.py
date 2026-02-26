"""
Patch-level Convolutional Gating Network for dynamic expert fusion.

Design (Seg-MoE thesis — core innovation):
  Input:  Expert logit patches  [B, K*M, pH, pW]
  Arch:   Shared ConvNet backbone + Residual FC head (~30-50 K params)
  Output: Softmax gating weights [B, K] or [B, K, M] (per-class)
  Fusion: fused_logits = Σ_k w_k · logits_k  →  CE/Dice loss (语义正确)

Key features:
  - Logits-only pipeline: 直接用 raw logits, 保留幅度信息, 无 softmax 压缩
  - Shared backbone: 3-layer ConvNet 提取空间上下文 [B, h, 1, 1]
  - Residual FC head: GAP → FC(h, out) + shortcut FC(h, out) 改善梯度流
  - Temperature-annealed softmax: τ 从 2.0 退火到 0.5
  - Load balancing regularization: 防止专家坍缩 (Shazeer et al. 2017)
  - Spatial smoothness regularization: TV-norm on weight map
  - Gaussian-blended overlap inference

References:
  - Shazeer et al. 2017  "Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer"
  - Riquelme et al. 2021 "Scaling Vision with Sparse Mixture of Experts" (V-MoE)
  - Dang et al. 2024     "Two-layer Ensemble of DL Models for Medical Image Segmentation"
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class PatchGatingConfig:
    """Configuration for the patch-level gating network."""

    num_experts: int = 3
    num_classes: int = 3

    # Patch parameters
    patch_size: int = 64
    stride: int = 32
    blend_mode: str = "gaussian"    # "gaussian" | "average"

    # Network architecture
    hidden_dim: int = 64
    dropout: float = 0.1
    per_class: bool = False         # True → [K, M] weights, False → [K]
    use_residual_head: bool = True  # Residual shortcut in FC head

    # Temperature annealing
    temperature_start: float = 2.0
    temperature_end: float = 0.5

    # Load balancing
    load_balance_weight: float = 0.01

    # Spatial smoothness regularization (TV on weight maps)
    spatial_smooth_weight: float = 0.0  # 0 = disabled; try 1e-3

    @property
    def in_channels(self) -> int:
        """Input channels: K * M (logits only)."""
        return self.num_experts * self.num_classes


# ---------------------------------------------------------------------------
# Temperature schedule
# ---------------------------------------------------------------------------

def compute_temperature(
    epoch: int,
    max_epochs: int,
    t_start: float = 2.0,
    t_end: float = 0.5,
) -> float:
    """Exponential temperature annealing: τ_start → τ_end."""
    if max_epochs <= 1:
        return t_end
    ratio = epoch / (max_epochs - 1)
    return t_start * (t_end / t_start) ** ratio


# ---------------------------------------------------------------------------
# Regularization losses
# ---------------------------------------------------------------------------

def compute_load_balance_loss(weights: torch.Tensor) -> torch.Tensor:
    """Load-balancing loss to prevent expert collapse.

    L = K · Σ_k f_k²    (Shazeer et al. 2017, Eq. 6 variant)

    Args:
        weights: [B, K] gating weights (each row sums to 1)
    Returns:
        scalar loss ≥ 1  (== 1 when perfectly uniform)
    """
    K = weights.shape[1]
    usage = weights.mean(dim=0)          # [K]
    return K * (usage ** 2).sum()


def compute_spatial_smooth_loss(weights: torch.Tensor) -> torch.Tensor:
    """Total Variation smoothness loss on per-patch weight map.

    Args:
        weights: [B, K] — per-patch gating weights (B = N_patches in sequence)
    Returns:
        scalar TV loss
    """
    if weights.shape[0] < 2:
        return weights.new_tensor(0.0)
    diff = (weights[1:] - weights[:-1]).abs()  # [B-1, K]
    return diff.mean()


# ---------------------------------------------------------------------------
# Network building blocks
# ---------------------------------------------------------------------------

class _SharedBackbone(nn.Module):
    """3-layer Conv + BN + GELU backbone → [B, h, 1, 1]."""

    def __init__(self, in_ch: int, hidden_dim: int) -> None:
        super().__init__()
        h = hidden_dim
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, h, 3, padding=1, bias=False),
            nn.BatchNorm2d(h),
            nn.GELU(),
            nn.Conv2d(h, h, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(h),
            nn.GELU(),
            nn.Conv2d(h, h, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(h),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),  # [B, h, 1, 1]
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).flatten(1)  # [B, h]


class _ResidualHead(nn.Module):
    """FC head with residual shortcut.

    main: h → h//2 → out_dim
    skip: h → out_dim  (when use_residual=True)
    """

    def __init__(
        self,
        hidden_dim: int,
        out_dim: int,
        dropout: float = 0.1,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        h = hidden_dim
        self.use_residual = use_residual
        self.main = nn.Sequential(
            nn.Linear(h, h // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h // 2, out_dim),
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
# Gating network
# ---------------------------------------------------------------------------

class PatchConvGate2D(nn.Module):
    """Patch-level convolutional gating network (logits-only pipeline).

    Architecture
    ------------
    Input: [B, K*M, pH, pW] — expert logit patches

    Backbone: Conv(K*M, h, 3) → Conv(h, h, s=2) → Conv(h, h, s=2) → GAP → [B, h]

    Head (residual):
            main: FC(h, h//2) + GELU + Dropout → FC(h//2, out_dim)
            skip: FC(h, out_dim)
            output: main + skip

    → softmax(output / τ, dim=K)  →  weights [B, K] or [B, K, M]

    Fusion: fused_logits = Σ_k w_k * logits_k  [B, M, pH, pW]
    Loss:   CE + Dice(fused_logits, gt)   ← semantically correct
    """

    def __init__(self, cfg: PatchGatingConfig):
        super().__init__()
        self.cfg = cfg
        K, M = cfg.num_experts, cfg.num_classes
        h = cfg.hidden_dim
        out_dim = K * M if cfg.per_class else K

        self.backbone = _SharedBackbone(cfg.in_channels, h)
        self.head = _ResidualHead(h, out_dim, cfg.dropout, cfg.use_residual_head)

    def forward(
        self,
        x: torch.Tensor,
        temperature: float | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, K*M, pH, pW]  expert logit patches
            temperature: softmax temperature override

        Returns:
            weights: [B, K] or [B, K, M]  (sums to 1 over K)
        """
        τ = temperature if temperature is not None else self.cfg.temperature_start
        feat = self.backbone(x)           # [B, h]
        gate_logits = self.head(feat)     # [B, out_dim]

        K, M = self.cfg.num_experts, self.cfg.num_classes
        if self.cfg.per_class:
            gate_logits = gate_logits.view(-1, K, M)
            return F.softmax(gate_logits / τ, dim=1)
        return F.softmax(gate_logits / τ, dim=1)   # [B, K]

    def fuse_logits(
        self,
        logits: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Weighted fusion of expert logit maps.

        fused = Σ_k w_k * logits_k  →  compatible with CE/Dice loss directly.

        Args:
            logits:  [B, K, M, pH, pW]
            weights: [B, K] or [B, K, M]
        Returns:
            fused:   [B, M, pH, pW]
        """
        if weights.dim() == 2:
            w = weights[:, :, None, None, None]   # [B, K, 1, 1, 1]
        else:
            w = weights[:, :, :, None, None]       # [B, K, M, 1, 1]
        return (w * logits).sum(dim=1)             # [B, M, pH, pW]

    def weights_per_expert(self, weights: torch.Tensor) -> torch.Tensor:
        """Collapse per-class weights to per-expert scalar for visualization.

        Args:
            weights: [B, K] or [B, K, M]
        Returns:
            [B, K]
        """
        if weights.dim() == 3:
            return weights.mean(dim=2)  # mean over M → [B, K]
        return weights


Design (Seg-MoE thesis — core innovation):
  Input:  Expert probability patches  [B, K*M, pH, pW]         (probs domain)
       OR Expert logits patches        [B, K*M, pH, pW]         (logits domain)
       OR Concatenated                 [B, K*2M, pH, pW]        (probs+logits)
  Arch:   Shared ConvNet backbone + Residual FC head (~30-50 K params)
  Output: Softmax gating weights  [B, K]  or  [B, K, M]  (per-class)

Key features:
  - input_channels: inferred from K, M, input_domain (flexible, no hardcode)
  - Shared backbone: 3-layer ConvNet extracts spatial context [B, h, 1, 1]
  - Residual FC head: GAP → FC(h, out) + shortcut FC(in, out) for gradient health
  - Temperature-annealed softmax: τ 从 2.0 退火到 0.5
  - Load balancing regularization: 防止专家坍缩 (Shazeer et al. 2017)
  - Spatial smoothness regularization: TV-norm on weight map (optional)
  - fuse_probs():  weighted sum over probs  → [B, M, pH, pW]
  - fuse_logits(): weighted sum over logits → [B, M, pH, pW] (semantically correct loss)
  - Gaussian-blended overlap inference

References:
  - Shazeer et al. 2017  "Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer"
  - Riquelme et al. 2021 "Scaling Vision with Sparse Mixture of Experts" (V-MoE)
  - Dang et al. 2024     "Two-layer Ensemble of DL Models for Medical Image Segmentation"
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class PatchGatingConfig:
    """Configuration for the patch-level gating network."""

    num_experts: int = 3
    num_classes: int = 3

    # Patch parameters
    patch_size: int = 64
    stride: int = 32
    blend_mode: str = "gaussian"    # "gaussian" | "average"

    # Input domain: controls in_channels = K*M ("probs"/"logits") or K*2M ("probs+logits")
    input_domain: str = "probs"     # "probs" | "logits" | "probs+logits"

    # Network architecture
    hidden_dim: int = 64
    dropout: float = 0.1
    per_class: bool = False         # True → [K, M] weights, False → [K]
    use_residual_head: bool = True  # Residual shortcut in FC head

    # Fusion domain for training loss
    fusion_domain: str = "logits"   # "logits" (semantically correct) | "probs" (legacy)

    # Temperature annealing
    temperature_start: float = 2.0
    temperature_end: float = 0.5

    # Load balancing
    load_balance_weight: float = 0.01

    # Spatial smoothness regularization (TV on weight maps)
    spatial_smooth_weight: float = 0.0  # 0 = disabled; try 1e-3

    @property
    def in_channels(self) -> int:
        """Input channels to the network based on input_domain."""
        km = self.num_experts * self.num_classes
        return km * 2 if self.input_domain == "probs+logits" else km


# ---------------------------------------------------------------------------
# Temperature schedule
# ---------------------------------------------------------------------------

def compute_temperature(
    epoch: int,
    max_epochs: int,
    t_start: float = 2.0,
    t_end: float = 0.5,
) -> float:
    """Exponential temperature annealing: τ_start → τ_end."""
    if max_epochs <= 1:
        return t_end
    ratio = epoch / (max_epochs - 1)
    return t_start * (t_end / t_start) ** ratio


# ---------------------------------------------------------------------------
# Load balancing loss
# ---------------------------------------------------------------------------

def compute_load_balance_loss(weights: torch.Tensor) -> torch.Tensor:
    """Load-balancing loss to prevent expert collapse.

    Penalises non-uniform average expert usage across the batch.
    Minimised when every expert is used equally (uniform *f* = 1/K).

    L = K · Σ_k f_k²    (Shazeer et al. 2017, Eq. 6 variant)

    Args:
        weights: [B, K] gating weights (each row sums to 1)
    Returns:
        scalar loss ≥ 1  (== 1 when perfectly uniform)
    """
    K = weights.shape[1]
    usage = weights.mean(dim=0)          # [K]
    return K * (usage ** 2).sum()


# ---------------------------------------------------------------------------
# Spatial smoothness regularization (TV-norm on weight map)
# ---------------------------------------------------------------------------

def compute_spatial_smooth_loss(weights: torch.Tensor) -> torch.Tensor:
    """Total Variation smoothness loss on per-patch weight map.

    Encourages spatial coherence of gating decisions across adjacent patches.
    Applied over the batch dimension (treats B as spatial positions).

    Args:
        weights: [B, K] — scalar per-patch gating weights (B = N_patches in sequence)
    Returns:
        scalar TV loss (mean absolute difference between consecutive entries)
    """
    if weights.shape[0] < 2:
        return weights.new_tensor(0.0)
    diff = (weights[1:] - weights[:-1]).abs()  # [B-1, K]
    return diff.mean()


# ---------------------------------------------------------------------------
# Gating network backbone
# ---------------------------------------------------------------------------

class _SharedBackbone(nn.Module):
    """3-layer Conv + BN + GELU backbone → [B, h, 1, 1]."""

    def __init__(self, in_ch: int, hidden_dim: int) -> None:
        super().__init__()
        h = hidden_dim
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, h, 3, padding=1, bias=False),
            nn.BatchNorm2d(h),
            nn.GELU(),
            nn.Conv2d(h, h, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(h),
            nn.GELU(),
            nn.Conv2d(h, h, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(h),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),  # [B, h, 1, 1]
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).flatten(1)  # [B, h]


class _ResidualHead(nn.Module):
    """FC head with residual shortcut from flattened input to output.

    main path:  h → h//2 → out_dim
    skip path:  in_features → out_dim  (1x1 linear shortcut, no bias)
    output: main + skip (before softmax)
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float = 0.1,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        h = hidden_dim
        self.use_residual = use_residual
        self.main = nn.Sequential(
            nn.Flatten(),
            nn.Linear(h, h // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h // 2, out_dim),
        )
        if use_residual:
            self.skip = nn.Linear(in_features, out_dim, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, feat: torch.Tensor, raw_flat: torch.Tensor) -> torch.Tensor:
        """
        feat     : [B, h]   — backbone output (GAP'd)
        raw_flat : [B, in_features]  — raw flattened input (for skip)
        Returns  : [B, out_dim] logits
        """
        out = self.main(feat)
        if self.use_residual:
            out = out + self.skip(raw_flat)
        return out


# ---------------------------------------------------------------------------
# Gating network
# ---------------------------------------------------------------------------

class PatchConvGate2D(nn.Module):
    """Patch-level convolutional gating network with shared backbone + residual head.

    Architecture
    ------------
    Backbone: Conv(C_in, h, 3) + BN + GELU
            → Conv(h, h, 3, s=2) + BN + GELU
            → Conv(h, h, 3, s=2) + BN + GELU
            → AdaptiveAvgPool2d(1) → Flatten  →  [B, h]

    Head (residual):
            main: FC(h, h//2) + GELU + Dropout → FC(h//2, out_dim)
            skip: FC(C_in*pH*pW, out_dim)   ← applied to flattened raw input

    Output: softmax([main + skip] / τ, dim=experts)

    ~30-50 K parameters for K=3, M=4, h=64.
    """

    def __init__(self, cfg: PatchGatingConfig):
        super().__init__()
        self.cfg = cfg
        in_ch = cfg.in_channels
        K, M = cfg.num_experts, cfg.num_classes
        h = cfg.hidden_dim

        self.backbone = _SharedBackbone(in_ch, h)

        out_dim = K * M if cfg.per_class else K
        # We use a proxy in_features = h for the shortcut (flattened GAP output)
        # This avoids depending on pH*pW at construction time.
        self.head = _ResidualHead(
            in_features=h,
            hidden_dim=h,
            out_dim=out_dim,
            dropout=cfg.dropout,
            use_residual=cfg.use_residual_head,
        )

    # ----- forward -----
    def forward(
        self,
        x: torch.Tensor,
        temperature: float | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, C_in, pH, pW]  gate input (probs / logits / probs+logits patches)
            temperature: softmax temperature (overrides cfg.temperature_start)

        Returns:
            weights: [B, K] or [B, K, M]  (sums to 1 over K)
        """
        τ = temperature if temperature is not None else self.cfg.temperature_start
        feat = self.backbone(x)              # [B, h]
        gate_logits = self.head(feat, feat)  # [B, out_dim]  (skip connects h→out)

        K, M = self.cfg.num_experts, self.cfg.num_classes
        if self.cfg.per_class:
            gate_logits = gate_logits.view(-1, K, M)
            return F.softmax(gate_logits / τ, dim=1)   # softmax over K dim
        return F.softmax(gate_logits / τ, dim=1)        # [B, K]

    # ----- fusion helpers -----
    def fuse_probs(
        self,
        probs: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Weighted fusion of expert **probability** maps.

        Args:
            probs:   [B, K, M, pH, pW]  softmax probs
            weights: [B, K] or [B, K, M]
        Returns:
            fused:   [B, M, pH, pW]  (still in prob space — do NOT pass to CE/Dice directly)
        """
        if weights.dim() == 2:
            w = weights[:, :, None, None, None]      # [B, K, 1, 1, 1]
        else:
            w = weights[:, :, :, None, None]          # [B, K, M, 1, 1]
        return (w * probs).sum(dim=1)                 # [B, M, pH, pW]

    def fuse_logits(
        self,
        logits: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Weighted fusion of expert **logit** maps.

        Mathematically cleaner than fusing probs: preserves calibration and
        allows standard CE / Dice losses without semantic mismatch.

        Args:
            logits:  [B, K, M, pH, pW]  raw logit maps
            weights: [B, K] or [B, K, M]
        Returns:
            fused:   [B, M, pH, pW]  (logit-space — compatible with CE/Dice loss_fn(logits, target))
        """
        if weights.dim() == 2:
            w = weights[:, :, None, None, None]      # [B, K, 1, 1, 1]
        else:
            w = weights[:, :, :, None, None]          # [B, K, M, 1, 1]
        return (w * logits).sum(dim=1)                # [B, M, pH, pW]

    def weights_per_expert(self, weights: torch.Tensor) -> torch.Tensor:
        """Collapse per-class weights to per-expert scalar for visualization.

        Args:
            weights: [B, K] or [B, K, M]
        Returns:
            [B, K]  — averaged (or passed-through) per-expert weight
        """
        if weights.dim() == 3:
            return weights.mean(dim=2)  # mean over class dim → [B, K]
        return weights                  # already [B, K]


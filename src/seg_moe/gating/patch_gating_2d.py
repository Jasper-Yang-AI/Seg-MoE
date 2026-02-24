"""
Patch-level Convolutional Gating Network for dynamic expert fusion.

Design (Seg-MoE thesis — core innovation):
  Input:  Expert probability patches  [B, K*M, pH, pW]
  Arch:   3-layer ConvNet + GAP + FC head (~25 K params)
  Output: Softmax gating weights  [B, K]  or  [B, K, M]  (per-class)

Key features:
  - Temperature-annealed softmax: τ 从 2.0 退火到 0.5, 训练初期均匀探索, 后期锐化
  - Load balancing regularization: 防止专家坍缩 (Shazeer et al. 2017)
  - Gaussian-blended overlap inference: stride < patch_size 时平滑拼接

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

    # Temperature annealing
    temperature_start: float = 2.0
    temperature_end: float = 0.5

    # Load balancing
    load_balance_weight: float = 0.01


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
# Gating network
# ---------------------------------------------------------------------------

class PatchConvGate2D(nn.Module):
    """Patch-level convolutional gating network.

    Architecture
    ------------
    Conv(KM, h, 3) + BN + GELU
    → Conv(h, h, 3, s=2) + BN + GELU       (down-sample ×2)
    → Conv(h, h, 3, s=2) + BN + GELU       (down-sample ×4)
    → AdaptiveAvgPool → Flatten
    → FC(h, h//2) + GELU + Dropout
    → FC(h//2, out_dim)
    → softmax(dim=experts) / τ

    ~25 K parameters for K=3, M=3, h=64.
    """

    def __init__(self, cfg: PatchGatingConfig):
        super().__init__()
        self.cfg = cfg
        K, M = cfg.num_experts, cfg.num_classes
        in_ch = K * M
        h = cfg.hidden_dim

        self.features = nn.Sequential(
            nn.Conv2d(in_ch, h, 3, padding=1, bias=False),
            nn.BatchNorm2d(h),
            nn.GELU(),
            nn.Conv2d(h, h, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(h),
            nn.GELU(),
            nn.Conv2d(h, h, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(h),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )

        out_dim = K * M if cfg.per_class else K
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(h, h // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(h // 2, out_dim),
        )

        self._init_weights()

    # ----- weight init -----
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ----- forward -----
    def forward(
        self,
        x: torch.Tensor,
        temperature: float | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, K*M, pH, pW]  expert probability patches
            temperature: softmax temperature (overrides cfg.temperature_start)

        Returns:
            weights: [B, K] or [B, K, M]  (sums to 1 over K)
        """
        τ = temperature if temperature is not None else self.cfg.temperature_start
        feat = self.features(x)
        logits = self.head(feat)

        K, M = self.cfg.num_experts, self.cfg.num_classes
        if self.cfg.per_class:
            logits = logits.view(-1, K, M)
            return F.softmax(logits / τ, dim=1)     # softmax over K
        return F.softmax(logits / τ, dim=1)          # [B, K]

    # ----- fusion helper -----
    def fuse_probs(
        self,
        probs: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Weighted fusion of expert probabilities.

        Args:
            probs:   [B, K, M, pH, pW]
            weights: [B, K] or [B, K, M]
        Returns:
            fused:   [B, M, pH, pW]
        """
        if weights.dim() == 2:
            w = weights[:, :, None, None, None]      # [B, K, 1, 1, 1]
        else:
            w = weights[:, :, :, None, None]          # [B, K, M, 1, 1]
        return (w * probs).sum(dim=1)

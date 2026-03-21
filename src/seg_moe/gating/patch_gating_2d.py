"""Patch-level 2D gating network with entropy-aware expert relation modeling."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PatchGatingConfig:
    num_experts: int = 3
    num_classes: int = 3
    patch_size: int = 64
    stride: int = 32
    blend_mode: str = "gaussian"
    hidden_dim: int = 64
    score_hidden_dim: int = 64
    dropout: float = 0.1
    per_class: bool = False
    use_residual_head: bool = True
    use_entropy: bool = True
    use_consensus_features: bool = True
    use_disagreement_features: bool = True
    use_confidence_features: bool = True
    temperature_start: float = 2.0
    temperature_end: float = 0.5
    load_balance_weight: float = 0.01
    spatial_smooth_weight: float = 0.0

    @property
    def expert_input_channels(self) -> int:
        return self.num_classes + (1 if self.use_entropy else 0)


def compute_temperature(
    epoch: int,
    max_epochs: int,
    t_start: float = 2.0,
    t_end: float = 0.5,
) -> float:
    if max_epochs <= 1:
        return t_end
    ratio = epoch / (max_epochs - 1)
    return t_start * (t_end / t_start) ** ratio


def compute_load_balance_loss(weights: torch.Tensor) -> torch.Tensor:
    experts = weights.shape[1]
    usage = weights.mean(dim=0)
    return experts * (usage ** 2).sum()


def compute_spatial_smooth_loss(
    weights: torch.Tensor,
    sample_ids: torch.Tensor | None = None,
    positions: torch.Tensor | None = None,
) -> torch.Tensor:
    if weights.shape[0] < 2:
        return weights.new_tensor(0.0)
    if sample_ids is None or positions is None:
        return (weights[1:] - weights[:-1]).abs().mean()

    losses: list[torch.Tensor] = []
    sample_ids = sample_ids.view(-1)
    positions = positions.view(weights.shape[0], -1)
    unique_ids = torch.unique(sample_ids)
    scale = 100000
    for sid in unique_ids:
        keep = sample_ids == sid
        if int(keep.sum()) < 2:
            continue
        local_weights = weights[keep]
        local_pos = positions[keep]
        key = local_pos[:, 0].long() * scale + local_pos[:, 1].long()
        order = torch.argsort(key)
        losses.append((local_weights[order][1:] - local_weights[order][:-1]).abs().mean())
    if not losses:
        return weights.new_tensor(0.0)
    return torch.stack(losses).mean()


class _SharedExpertEncoder2D(nn.Module):
    def __init__(self, in_ch: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _SpatialAttentionPool2D(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.score = nn.Conv2d(hidden_dim, 1, kernel_size=1, bias=True)

    def forward(self, feat_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, channels, height, width = feat_map.shape
        attn_logits = self.score(feat_map).view(batch, 1, height * width)
        attn = F.softmax(attn_logits, dim=-1)
        feat_flat = feat_map.view(batch, channels, height * width)
        pooled = torch.sum(feat_flat * attn, dim=-1)
        return pooled, attn.view(batch, 1, height, width)


class _ExpertScoreHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, dropout: float, use_residual: bool) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        self.skip = nn.Linear(in_dim, out_dim, bias=False) if use_residual else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.main(x)
        if self.skip is not None:
            out = out + self.skip(x)
        return out


class PatchConvGate2D(nn.Module):
    """Shared encoding + attention pooling + relation-aware expert scoring."""

    def __init__(self, cfg: PatchGatingConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = _SharedExpertEncoder2D(cfg.expert_input_channels, cfg.hidden_dim)
        self.pool = _SpatialAttentionPool2D(cfg.hidden_dim)

        score_input_dim = cfg.hidden_dim
        if cfg.use_consensus_features:
            score_input_dim += cfg.hidden_dim + 1
        if cfg.use_disagreement_features:
            score_input_dim += cfg.hidden_dim + 1
        if cfg.use_confidence_features:
            score_input_dim += 1

        score_out_dim = cfg.num_classes if cfg.per_class else 1
        self.score_head = _ExpertScoreHead(
            in_dim=score_input_dim,
            out_dim=score_out_dim,
            hidden_dim=cfg.score_hidden_dim,
            dropout=cfg.dropout,
            use_residual=cfg.use_residual_head,
        )

    def _reshape_expert_maps(self, tensor: torch.Tensor, name: str) -> torch.Tensor:
        if tensor.dim() == 5:
            batch, experts, classes, _, _ = tensor.shape
            if experts != self.cfg.num_experts or classes != self.cfg.num_classes:
                raise ValueError(
                    f"{name} shape mismatch: expected [B,{self.cfg.num_experts},{self.cfg.num_classes},H,W], "
                    f"got {tuple(tensor.shape)}"
                )
            return tensor
        if tensor.dim() == 4:
            batch, channels, height, width = tensor.shape
            expected_channels = self.cfg.num_experts * self.cfg.num_classes
            if channels != expected_channels:
                raise ValueError(
                    f"{name} must have {expected_channels} flattened channels, got {channels}"
                )
            return tensor.reshape(batch, self.cfg.num_experts, self.cfg.num_classes, height, width)
        raise ValueError(
            f"{name} must have shape [B,K,M,H,W] or legacy [B,K*M,H,W], got {tuple(tensor.shape)}"
        )

    def _compute_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=2)
        entropy = -(probs * torch.log(probs.clamp(min=1e-6))).sum(dim=2, keepdim=True)
        if self.cfg.num_classes > 1:
            entropy = entropy / torch.log(torch.tensor(float(self.cfg.num_classes), device=logits.device))
        return entropy

    def _encode_experts(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, experts, _, height, width = logits.shape
        probs = torch.softmax(logits, dim=2)
        entropy = self._compute_entropy(logits)
        expert_inputs = torch.cat([logits, entropy], dim=2) if self.cfg.use_entropy else logits

        enc_in = expert_inputs.reshape(batch * experts, expert_inputs.shape[2], height, width)
        feat_map = self.encoder(enc_in)
        pooled_feat, attn = self.pool(feat_map)
        feat = pooled_feat.reshape(batch, experts, -1)

        entropy_small = F.interpolate(
            entropy.reshape(batch * experts, 1, height, width),
            size=attn.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        confidence = (entropy_small * attn).flatten(2).sum(dim=-1).reshape(batch, experts, 1)
        return feat, probs, confidence

    def _build_relation_features(self, feat: torch.Tensor, probs: torch.Tensor, confidence: torch.Tensor) -> torch.Tensor:
        experts = feat.shape[1]
        parts = [feat]
        if experts > 1:
            others_mean = (feat.sum(dim=1, keepdim=True) - feat) / (experts - 1)
            probs_flat = probs.flatten(start_dim=3)
            pairwise_prob_diff = (probs_flat[:, :, None] - probs_flat[:, None, :]).abs().mean(dim=(-1, -2))
            disagreement_score = pairwise_prob_diff.sum(dim=-1, keepdim=True) / (experts - 1)

            norm_feat = F.normalize(feat, dim=-1)
            sim = torch.matmul(norm_feat, norm_feat.transpose(1, 2))
            consensus_score = (sim.sum(dim=-1, keepdim=True) - 1.0) / (experts - 1)
        else:
            others_mean = feat
            disagreement_score = feat.new_zeros(feat.shape[0], feat.shape[1], 1)
            consensus_score = feat.new_ones(feat.shape[0], feat.shape[1], 1)

        if self.cfg.use_consensus_features:
            parts.extend([others_mean, consensus_score])
        if self.cfg.use_disagreement_features:
            parts.extend([torch.abs(feat - others_mean), disagreement_score])
        if self.cfg.use_confidence_features:
            parts.append(confidence)
        return torch.cat(parts, dim=-1)

    def forward(self, logits: torch.Tensor, temperature: float | None = None) -> torch.Tensor:
        logits = self._reshape_expert_maps(logits, "logits")
        tau = temperature if temperature is not None else self.cfg.temperature_start
        feat, probs, confidence = self._encode_experts(logits)
        relation_feat = self._build_relation_features(feat, probs, confidence)

        batch, experts, _ = relation_feat.shape
        raw_scores = self.score_head(relation_feat.reshape(batch * experts, -1))
        if self.cfg.per_class:
            raw_scores = raw_scores.reshape(batch, experts, self.cfg.num_classes)
            return F.softmax(raw_scores / tau, dim=1)
        raw_scores = raw_scores.reshape(batch, experts)
        return F.softmax(raw_scores / tau, dim=1)

    def fuse_logits(self, logits: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        logits = self._reshape_expert_maps(logits, "logits")
        if weights.dim() == 2:
            weights = weights[:, :, None, None, None]
        else:
            weights = weights[:, :, :, None, None]
        return (logits * weights).sum(dim=1)

    def fuse_probs(self, probs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        probs = self._reshape_expert_maps(probs, "probs")
        if weights.dim() == 2:
            weights = weights[:, :, None, None, None]
        else:
            weights = weights[:, :, :, None, None]
        return (probs * weights).sum(dim=1)

    def weights_per_expert(self, weights: torch.Tensor) -> torch.Tensor:
        if weights.dim() == 3:
            return weights.mean(dim=2)
        return weights

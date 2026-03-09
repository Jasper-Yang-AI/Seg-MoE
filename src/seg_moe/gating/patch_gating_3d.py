"""3D patch-level gating network with entropy-aware expert relation modeling."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PatchGating3DConfig:
    num_experts: int
    num_classes: int
    per_class: bool = False


@dataclass
class PatchGatingConfig3D:
    num_experts: int = 3
    num_classes: int = 4
    patch_size: Tuple[int, int, int] = (32, 32, 16)
    stride: Tuple[int, int, int] = (16, 16, 8)
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


def compute_temperature_3d(
    epoch: int,
    max_epochs: int,
    t_start: float = 2.0,
    t_end: float = 0.5,
) -> float:
    if max_epochs <= 1:
        return t_end
    ratio = epoch / (max_epochs - 1)
    return t_start * (t_end / t_start) ** ratio


def compute_load_balance_loss_3d(weights: torch.Tensor) -> torch.Tensor:
    experts = weights.shape[1]
    usage = weights.mean(dim=0)
    return experts * (usage ** 2).sum()


def compute_spatial_smooth_loss_3d(
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
    scale_h = 100000
    scale_w = 1000
    for sid in unique_ids:
        keep = sample_ids == sid
        if int(keep.sum()) < 2:
            continue
        local_weights = weights[keep]
        local_pos = positions[keep]
        key = local_pos[:, 0].long() * scale_h * scale_w + local_pos[:, 1].long() * scale_w + local_pos[:, 2].long()
        order = torch.argsort(key)
        losses.append((local_weights[order][1:] - local_weights[order][:-1]).abs().mean())
    if not losses:
        return weights.new_tensor(0.0)
    return torch.stack(losses).mean()


class _SharedExpertEncoder3D(nn.Module):
    def __init__(self, in_ch: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm3d(hidden_dim),
            nn.GELU(),
            nn.Conv3d(hidden_dim, hidden_dim, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(hidden_dim),
            nn.GELU(),
            nn.Conv3d(hidden_dim, hidden_dim, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(hidden_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _SpatialAttentionPool3D(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.score = nn.Conv3d(hidden_dim, 1, kernel_size=1, bias=True)

    def forward(self, feat_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, channels, depth, height, width = feat_map.shape
        attn_logits = self.score(feat_map).view(batch, 1, depth * height * width)
        attn = F.softmax(attn_logits, dim=-1)
        feat_flat = feat_map.view(batch, channels, depth * height * width)
        pooled = torch.sum(feat_flat * attn, dim=-1)
        return pooled, attn.view(batch, 1, depth, height, width)


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


class PatchConvGate3D(nn.Module):
    def __init__(self, cfg: PatchGatingConfig3D) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = _SharedExpertEncoder3D(cfg.expert_input_channels, cfg.hidden_dim)
        self.pool = _SpatialAttentionPool3D(cfg.hidden_dim)

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

    def _compute_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=2)
        entropy = -(probs * torch.log(probs.clamp(min=1e-6))).sum(dim=2, keepdim=True)
        if self.cfg.num_classes > 1:
            entropy = entropy / torch.log(torch.tensor(float(self.cfg.num_classes), device=logits.device))
        return entropy

    def _encode_experts(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, experts, _, depth, height, width = logits.shape
        probs = torch.softmax(logits, dim=2)
        entropy = self._compute_entropy(logits)
        expert_inputs = torch.cat([logits, entropy], dim=2) if self.cfg.use_entropy else logits

        enc_in = expert_inputs.view(batch * experts, expert_inputs.shape[2], depth, height, width)
        feat_map = self.encoder(enc_in)
        pooled_feat, attn = self.pool(feat_map)
        feat = pooled_feat.view(batch, experts, -1)

        entropy_small = F.interpolate(
            entropy.view(batch * experts, 1, depth, height, width),
            size=attn.shape[-3:],
            mode="trilinear",
            align_corners=False,
        )
        confidence = (entropy_small * attn).flatten(2).sum(dim=-1).view(batch, experts, 1)
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

    def forward(self, logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        batch = logits.shape[0]
        feat, probs, confidence = self._encode_experts(logits)
        relation_feat = self._build_relation_features(feat, probs, confidence)
        experts = relation_feat.shape[1]
        raw_scores = self.score_head(relation_feat.view(batch * experts, -1))

        if self.cfg.per_class:
            raw_scores = raw_scores.view(batch, self.cfg.num_experts, self.cfg.num_classes)
            return F.softmax(raw_scores / temperature, dim=1)

        raw_scores = raw_scores.view(batch, experts)
        return F.softmax(raw_scores / temperature, dim=1)

    def fuse_logits(self, logits: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        if weights.dim() == 2:
            weights = weights[:, :, None, None, None, None]
        else:
            weights = weights[:, :, :, None, None, None]
        return (logits * weights).sum(dim=1)

    def weights_per_expert(self, weights: torch.Tensor) -> torch.Tensor:
        if weights.dim() == 3:
            return weights.mean(dim=2)
        return weights


class PatchGating3D(nn.Module):
    def __init__(self, cfg: PatchGating3DConfig):
        super().__init__()
        self.cfg = cfg

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("PatchGating3D is a legacy alias. Use PatchConvGate3D instead.")


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


def fuse_volume_sliding_window(
    model: PatchConvGate3D,
    logits_vol: torch.Tensor,
    patch_size: Tuple[int, int, int],
    stride: Tuple[int, int, int],
    num_classes: int,
    num_experts: int,
    temperature: float = 1.0,
    device: torch.device | None = None,
    blend_mode: str = "gaussian",
) -> torch.Tensor:
    if device is None:
        device = next(model.parameters()).device

    _, _, depth, height, width = logits_vol.shape
    pd, ph, pw = patch_size

    from seg_moe.data.gating_patch_dataset_3d import compute_patch_positions_3d

    positions = compute_patch_positions_3d((depth, height, width), patch_size, stride)
    fused_vol = torch.zeros(num_classes, depth, height, width, device="cpu")
    weight_map = torch.zeros(1, depth, height, width, device="cpu")
    importance = _gaussian_kernel_3d(patch_size) if blend_mode == "gaussian" else torch.ones(pd, ph, pw)

    model.eval()
    with torch.no_grad():
        for d0, h0, w0 in positions:
            patch_logits = logits_vol[:, :, d0:d0+pd, h0:h0+ph, w0:w0+pw]
            weights = model(patch_logits.unsqueeze(0).to(device), temperature=temperature)
            fused_patch = model.fuse_logits(patch_logits.unsqueeze(0).to(device), weights).squeeze(0).cpu()
            fused_vol[:, d0:d0+pd, h0:h0+ph, w0:w0+pw] += fused_patch * importance
            weight_map[:, d0:d0+pd, h0:h0+ph, w0:w0+pw] += importance

    return fused_vol / weight_map.clamp(min=1e-7)

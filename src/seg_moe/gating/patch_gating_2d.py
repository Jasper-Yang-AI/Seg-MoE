"""Patch-level 2D gating network with hierarchical semantic and anatomy context."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PatchGatingConfig:
    num_experts: int = 3
    num_classes: int = 3
    image_channels: int = 3
    patch_size: int = 64
    stride: int = 32
    blend_mode: str = "gaussian"
    hidden_dim: int = 64
    score_hidden_dim: int = 64
    context_hidden_dim: int = 32
    dropout: float = 0.1
    per_class: bool = False
    use_residual_head: bool = True
    use_entropy: bool = True
    use_consensus_features: bool = True
    use_disagreement_features: bool = True
    use_confidence_features: bool = True
    use_prior_agreement_features: bool = False
    use_layer1_semantics: bool = False
    use_image_context: bool = False
    use_position_channels: bool = False
    use_slice_position: bool = False
    use_context_film: bool = True
    temperature_start: float = 2.0
    temperature_end: float = 0.5
    load_balance_weight: float = 0.01
    spatial_smooth_weight: float = 0.0

    @property
    def expert_input_channels(self) -> int:
        return self.num_classes + (1 if self.use_entropy else 0)

    @property
    def context_input_channels(self) -> int:
        channels = 0
        if self.use_layer1_semantics:
            channels += 2 * self.num_classes + 1
        if self.use_image_context:
            channels += self.image_channels
        if self.use_position_channels:
            channels += 2
        return channels


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


class _ContextEncoder2D(nn.Module):
    def __init__(self, in_ch: int, hidden_dim: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )
        self.local_branch = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )
        self.global_branch = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, 1, bias=False),
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
        stem = self.stem(x)
        local_feat = self.local_branch(stem)
        global_feat = self.global_branch(stem)
        return self.fuse(torch.cat([local_feat, global_feat], dim=1))


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


class _ScalarEmbedding(nn.Module):
    def __init__(self, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PatchConvGate2D(nn.Module):
    """Hierarchical semantic-aware patch gate for 2D expert fusion."""

    def __init__(self, cfg: PatchGatingConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = _SharedExpertEncoder2D(cfg.expert_input_channels, cfg.hidden_dim)
        self.pool = _SpatialAttentionPool2D(cfg.hidden_dim)
        self.expert_feat_dim = cfg.hidden_dim * 2

        self.context_encoder = None
        self.context_pool = None
        self.context_feat_dim = 0
        if cfg.context_input_channels > 0:
            self.context_encoder = _ContextEncoder2D(cfg.context_input_channels, cfg.context_hidden_dim)
            self.context_pool = _SpatialAttentionPool2D(cfg.context_hidden_dim)
            self.context_feat_dim += cfg.context_hidden_dim * 2

        self.slice_embed = None
        if cfg.use_slice_position:
            slice_dim = max(8, cfg.context_hidden_dim // 2)
            self.slice_embed = _ScalarEmbedding(slice_dim)
            self.context_feat_dim += slice_dim

        self.context_mod = None
        if cfg.use_context_film and self.context_feat_dim > 0:
            self.context_mod = nn.Sequential(
                nn.Linear(self.context_feat_dim, self.expert_feat_dim * 2),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(self.expert_feat_dim * 2, self.expert_feat_dim * 2),
            )

        score_input_dim = self.expert_feat_dim
        if cfg.use_consensus_features:
            score_input_dim += self.expert_feat_dim + 1
        if cfg.use_disagreement_features:
            score_input_dim += self.expert_feat_dim + 1
        if cfg.use_confidence_features:
            score_input_dim += 1
        if cfg.use_prior_agreement_features:
            score_input_dim += 2
        if self.context_feat_dim > 0:
            score_input_dim += self.context_feat_dim

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
        logits = self._reshape_expert_maps(logits, "logits")
        batch, experts, _, height, width = logits.shape
        probs = torch.softmax(logits, dim=2)
        entropy = self._compute_entropy(logits)
        expert_inputs = torch.cat([logits, entropy], dim=2) if self.cfg.use_entropy else logits

        enc_in = expert_inputs.reshape(batch * experts, expert_inputs.shape[2], height, width)
        feat_map = self.encoder(enc_in)
        attn_feat, attn = self.pool(feat_map)
        avg_feat = F.adaptive_avg_pool2d(feat_map, output_size=1).flatten(1)
        feat = torch.cat([attn_feat, avg_feat], dim=1).view(batch, experts, -1)

        entropy_small = F.interpolate(
            entropy.reshape(batch * experts, 1, height, width),
            size=attn.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        confidence = (entropy_small * attn).flatten(2).sum(dim=-1).reshape(batch, experts, 1)
        return feat, probs, confidence

    def _build_context_input(self, extra: dict[str, torch.Tensor] | None) -> torch.Tensor | None:
        if not extra:
            return None
        parts: list[torch.Tensor] = []
        if self.cfg.use_image_context:
            image = extra.get("image")
            if image is None:
                raise ValueError("Gate config requires image context, but extra['image'] is missing")
            parts.append(image)
        if self.cfg.use_layer1_semantics:
            mean_map = extra.get("layer1_mean")
            entropy = extra.get("layer1_entropy")
            disagreement = extra.get("layer1_disagreement")
            if mean_map is None or entropy is None or disagreement is None:
                raise ValueError("Gate config requires Layer1 semantics, but semantic maps are missing")
            parts.extend([mean_map, entropy, disagreement])
        if self.cfg.use_position_channels:
            coords = extra.get("coords")
            if coords is None:
                raise ValueError("Gate config requires position channels, but extra['coords'] is missing")
            parts.append(coords)
        if not parts:
            return None
        return torch.cat(parts, dim=1)

    def _encode_context(self, extra: dict[str, torch.Tensor] | None) -> torch.Tensor | None:
        context_vec = None
        context_input = self._build_context_input(extra)
        if context_input is not None:
            if self.context_encoder is None or self.context_pool is None:
                raise RuntimeError("Context encoder is not initialized")
            feat_map = self.context_encoder(context_input)
            attn_feat, _ = self.context_pool(feat_map)
            avg_feat = F.adaptive_avg_pool2d(feat_map, output_size=1).flatten(1)
            context_vec = torch.cat([attn_feat, avg_feat], dim=1)

        if self.slice_embed is not None:
            if not extra or extra.get("slice_pos") is None:
                raise ValueError("Gate config requires slice position, but extra['slice_pos'] is missing")
            slice_pos = extra["slice_pos"]
            slice_vec = self.slice_embed(slice_pos.view(slice_pos.shape[0], 1))
            context_vec = slice_vec if context_vec is None else torch.cat([context_vec, slice_vec], dim=1)

        return context_vec

    def _apply_context_modulation(self, feat: torch.Tensor, context_vec: torch.Tensor | None) -> torch.Tensor:
        if context_vec is None or self.context_mod is None:
            return feat
        gamma, beta = torch.chunk(self.context_mod(context_vec), 2, dim=-1)
        gamma = torch.tanh(gamma)
        return feat * (1.0 + gamma[:, None, :]) + beta[:, None, :]

    def _build_relation_features(
        self,
        feat: torch.Tensor,
        probs: torch.Tensor,
        confidence: torch.Tensor,
        context_vec: torch.Tensor | None,
        extra: dict[str, torch.Tensor] | None,
    ) -> torch.Tensor:
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

        if self.cfg.use_prior_agreement_features and extra and extra.get("layer1_mean") is not None:
            prior = extra["layer1_mean"][:, None]  # [B,1,M,H,W]
            prior_gap = (probs - prior).abs().flatten(start_dim=2).mean(dim=-1, keepdim=True)
            prior_overlap = (probs * prior).flatten(start_dim=2).mean(dim=-1, keepdim=True)
            parts.extend([prior_gap, prior_overlap])

        if context_vec is not None:
            parts.append(context_vec[:, None, :].expand(-1, experts, -1))

        return torch.cat(parts, dim=-1)

    def forward(
        self,
        logits: torch.Tensor,
        extra: dict[str, torch.Tensor] | None = None,
        temperature: float | None = None,
    ) -> torch.Tensor:
        tau = temperature if temperature is not None else self.cfg.temperature_start
        feat, probs, confidence = self._encode_experts(logits)
        context_vec = self._encode_context(extra)
        feat = self._apply_context_modulation(feat, context_vec)
        relation_feat = self._build_relation_features(feat, probs, confidence, context_vec, extra)

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

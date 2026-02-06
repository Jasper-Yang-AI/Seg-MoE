from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, num_classes: int, smooth: float = 1.0) -> torch.Tensor:
    """Multiclass soft dice (supports 2D and 3D).

    logits: [B,C,H,W] or [B,C,D,H,W]
    target: [B,H,W] or [B,D,H,W] int64
    """
    probs = torch.softmax(logits, dim=1)
    target_1h = F.one_hot(target.clamp(min=0), num_classes=num_classes).float()
    # Move class dim from last to dim=1: [B,...,C] -> [B,C,...]
    ndim = target_1h.ndim
    perm = [0, ndim - 1] + list(range(1, ndim - 1))
    target_1h = target_1h.permute(*perm)
    # Sum over batch + spatial dims (all except class)
    spatial_dims = tuple(range(2, logits.ndim))
    dims = (0,) + spatial_dims
    intersection = torch.sum(probs * target_1h, dims)
    denom = torch.sum(probs + target_1h, dims)
    dice = (2.0 * intersection + smooth) / (denom + smooth)
    return 1.0 - dice.mean()


def ce_plus_dice(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    dice_smooth: float = 1.0,
    ce_weight: float = 1.0,
    dice_weight: float = 1.0,
    ignore_index: Optional[int] = None,
) -> torch.Tensor:
    ce = F.cross_entropy(logits, target, ignore_index=ignore_index if ignore_index is not None else -100)
    dl = soft_dice_loss(logits, target, num_classes=num_classes, smooth=dice_smooth)
    return ce_weight * ce + dice_weight * dl

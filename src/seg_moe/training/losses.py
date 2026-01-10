from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, num_classes: int, smooth: float = 1.0) -> torch.Tensor:
    """Multiclass soft dice.

    logits: [B,C,H,W]
    target: [B,H,W] int64
    """
    probs = torch.softmax(logits, dim=1)
    target_1h = F.one_hot(target.clamp(min=0), num_classes=num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
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

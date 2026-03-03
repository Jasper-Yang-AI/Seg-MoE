from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Distance map (for Boundary Loss)
# ---------------------------------------------------------------------------

def _compute_distance_map(target_1h: torch.Tensor) -> torch.Tensor:
    """Compute signed distance map from one-hot target.

    Uses efficient approximation via Euclidean distance transform.
    Replaces scipy.ndimage.distance_transform_edt for GPU tensors.

    Args:
        target_1h: [B, C, ...] float one-hot (2D or 3D)

    Returns:
        dist_map: [B, C, ...] signed distance (negative inside, positive outside)
    """
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError:
        raise ImportError("scipy is required for Boundary Loss. pip install scipy")

    B, C = target_1h.shape[:2]
    spatial_shape = target_1h.shape[2:]
    device = target_1h.device
    target_np = target_1h.detach().cpu().numpy()
    dist = torch.zeros_like(target_1h)

    for b in range(B):
        for c in range(C):
            fg = target_np[b, c]
            if fg.sum() == 0:
                # No foreground for this class -> all positive distance
                dist[b, c] = 1.0
                continue
            if fg.sum() == fg.size:
                # All foreground -> all negative distance
                dist[b, c] = -1.0
                continue
            # Distance from boundary: positive outside, negative inside
            pos_dist = distance_transform_edt(1 - fg)
            neg_dist = distance_transform_edt(fg)
            # Normalize by image diagonal for scale invariance
            diag = float(sum(int(s) ** 2 for s in spatial_shape) ** 0.5)
            signed = (pos_dist - neg_dist) / (diag + 1e-8)
            dist[b, c] = torch.from_numpy(signed).float()

    return dist.to(device)


def boundary_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Boundary Loss (Kervadec et al., MIDL 2019).

    Computes the dot product between softmax probabilities and signed
    distance maps derived from the ground truth, encouraging the
    predicted contour to align with the true boundary.

    logits: [B, C, H, W] or [B, C, D, H, W]
    target: [B, H, W] or [B, D, H, W] int64
    """
    probs = torch.softmax(logits, dim=1)
    target_1h = F.one_hot(target.clamp(min=0), num_classes=num_classes).float()
    ndim = target_1h.ndim
    perm = [0, ndim - 1] + list(range(1, ndim - 1))
    target_1h = target_1h.permute(*perm)

    dist_map = _compute_distance_map(target_1h)
    # Boundary loss = mean of (probs * dist_map) over foreground classes
    # Skip background (class 0) for boundary focus
    if num_classes > 1:
        bl = (probs[:, 1:] * dist_map[:, 1:]).mean()
    else:
        bl = (probs * dist_map).mean()
    return bl


# ---------------------------------------------------------------------------
# Soft Dice Loss
# ---------------------------------------------------------------------------


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


def ce_dice_boundary(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    dice_smooth: float = 1.0,
    ce_weight: float = 1.0,
    dice_weight: float = 1.0,
    boundary_weight: float = 0.5,
    ignore_index: Optional[int] = None,
) -> torch.Tensor:
    """CE + Dice + Boundary Loss.

    Layer2 uses this to focus on refining boundaries and small targets
    that Layer1 may have missed. The boundary term (Kervadec et al., MIDL 2019)
    directly optimizes boundary distance alignment.

    Args:
        boundary_weight: weight for boundary loss term (default 0.5)
    """
    ce = F.cross_entropy(logits, target, ignore_index=ignore_index if ignore_index is not None else -100)
    dl = soft_dice_loss(logits, target, num_classes=num_classes, smooth=dice_smooth)
    bl = boundary_loss(logits, target, num_classes=num_classes)
    return ce_weight * ce + dice_weight * dl + boundary_weight * bl


# ---------------------------------------------------------------------------
# Loss factory (config-driven)
# ---------------------------------------------------------------------------

def build_loss_fn(loss_cfg: dict, num_classes: int, *, ignore_index: int | None = None):
    """Build a loss function from config dict.

    Supported loss names:
      - "ce_plus_dice" (default)
      - "ce_dice_boundary" (Layer2 recommended)

    Args:
        loss_cfg: config dict with keys: name, dice_smooth, ce_weight, dice_weight, ...
        num_classes: number of segmentation classes
        ignore_index: override ignore_index (takes precedence over config)

    Returns a callable: loss_fn(logits, target) -> scalar tensor
    """
    name = str(loss_cfg.get("name", "ce_plus_dice")).lower()
    dice_smooth = float(loss_cfg.get("dice_smooth", 1.0))
    ce_weight = float(loss_cfg.get("ce_weight", 1.0))
    dice_weight = float(loss_cfg.get("dice_weight", 1.0))
    if ignore_index is None:
        ignore_index = loss_cfg.get("ignore_index")

    if name == "ce_dice_boundary":
        boundary_weight = float(loss_cfg.get("boundary_weight", 0.5))
        def _loss(logits, target):
            return ce_dice_boundary(
                logits, target, num_classes=num_classes,
                dice_smooth=dice_smooth, ce_weight=ce_weight,
                dice_weight=dice_weight, boundary_weight=boundary_weight,
                ignore_index=ignore_index)
        return _loss

    # Default: ce_plus_dice
    def _loss(logits, target):
        return ce_plus_dice(
            logits, target, num_classes=num_classes,
            dice_smooth=dice_smooth, ce_weight=ce_weight,
            dice_weight=dice_weight, ignore_index=ignore_index)
    return _loss

"""
3D Segmentation Evaluation Metrics.

Core metrics (Maier-Hein et al. 2024, Nature Methods):
  - Dice Similarity Coefficient (DSC)
  - IoU (Jaccard)
  - HD95 (95th percentile Hausdorff distance)
  - NSD (Normalised Surface Distance, τ=2mm)
  - Volume Similarity (VS)

Per-class, foreground only. Uses MONAI's compute_meandice if available,
falls back to pure-NumPy implementation.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Pure-NumPy helpers (no external deps)
# ---------------------------------------------------------------------------

def _dice_from_volumes(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> float:
    inter = (pred & gt).sum()
    union = pred.sum() + gt.sum()
    return float((2 * inter + eps) / (union + eps))


def _iou_from_volumes(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> float:
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    return float((inter + eps) / (union + eps))


def _hd95_3d(pred: np.ndarray, gt: np.ndarray, spacing_dhw: Tuple[float, float, float] = (1., 1., 1.)) -> float:
    """95th percentile Hausdorff distance (pure NumPy / scipy)."""
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError:
        return float("nan")

    if not pred.any() or not gt.any():
        return float("nan")

    # surface voxels via erosion
    from scipy.ndimage import binary_erosion
    pred_surf = pred ^ binary_erosion(pred)
    gt_surf = gt ^ binary_erosion(gt)

    if not pred_surf.any() or not gt_surf.any():
        return float("nan")

    dt_gt = distance_transform_edt(~gt_surf, sampling=spacing_dhw)
    dt_pred = distance_transform_edt(~pred_surf, sampling=spacing_dhw)

    pred_to_gt = dt_gt[pred_surf]
    gt_to_pred = dt_pred[gt_surf]
    all_dists = np.concatenate([pred_to_gt, gt_to_pred])
    return float(np.percentile(all_dists, 95))


def _nsd_3d(pred: np.ndarray, gt: np.ndarray,
            spacing_dhw: Tuple[float, float, float] = (1., 1., 1.),
            tolerance: float = 2.0) -> float:
    """Normalised Surface Distance (Nikolov et al. 2021)."""
    try:
        from scipy.ndimage import distance_transform_edt, binary_erosion
    except ImportError:
        return float("nan")

    if not pred.any() or not gt.any():
        return float("nan")

    pred_surf = pred ^ binary_erosion(pred)
    gt_surf   = gt   ^ binary_erosion(gt)
    if not pred_surf.any() or not gt_surf.any():
        return float("nan")

    dt_gt   = distance_transform_edt(~gt_surf,   sampling=spacing_dhw)
    dt_pred = distance_transform_edt(~pred_surf, sampling=spacing_dhw)

    pred_within = (dt_gt[pred_surf]   <= tolerance).sum()
    gt_within   = (dt_pred[gt_surf]   <= tolerance).sum()
    total       = pred_surf.sum() + gt_surf.sum()
    return float((pred_within + gt_within) / (total + 1e-7))


def _volume_similarity(pred: np.ndarray, gt: np.ndarray) -> float:
    """VS = 1 - |V_pred - V_gt| / (V_pred + V_gt)."""
    vp = pred.sum()
    vg = gt.sum()
    if vp + vg == 0:
        return float("nan")
    return float(1 - abs(vp - vg) / (vp + vg))


# ---------------------------------------------------------------------------
# Main metric function
# ---------------------------------------------------------------------------

def compute_segmentation_metrics_3d(
    pred_vol: np.ndarray,
    gt_vol: np.ndarray,
    num_classes: int,
    spacing_dhw: Tuple[float, float, float] = (1., 1., 1.),
    hd95: bool = True,
    nsd_tolerance: float = 2.0,
    compute_surface: bool = True,
) -> Dict[str, Any]:
    """Compute per-class 3D segmentation metrics.

    Args:
        pred_vol: [D, H, W] integer labels
        gt_vol:   [D, H, W] integer labels
        num_classes: total classes (including background=0)
        spacing_dhw: voxel spacing in mm (D, H, W)
        hd95: if True compute HD95 (slow); False skips it
        nsd_tolerance: NSD tolerance in mm

    Returns:
        Dict with per-class and mean metrics.
    """
    results: Dict[str, Any] = {}
    dice_list, iou_list, hd95_list, nsd_list, vs_list = [], [], [], [], []

    for c in range(1, num_classes):      # skip background
        p = (pred_vol == c)
        g = (gt_vol == c)

        if not g.any() and not p.any():
            # Both empty — skip this class for mean but record
            results[f"dice_c{c}"] = float("nan")
            results[f"iou_c{c}"]  = float("nan")
            results[f"hd95_c{c}"] = float("nan")
            results[f"nsd_c{c}"]  = float("nan")
            results[f"vs_c{c}"]   = float("nan")
            continue

        d  = _dice_from_volumes(p, g)
        io = _iou_from_volumes(p, g)
        vs = _volume_similarity(p, g)

        results[f"dice_c{c}"] = d
        results[f"iou_c{c}"]  = io
        results[f"vs_c{c}"]   = vs

        if compute_surface:
            h_val = _hd95_3d(p, g, spacing_dhw) if hd95 else float("nan")
            n_val = _nsd_3d(p, g, spacing_dhw, nsd_tolerance)
        else:
            h_val = n_val = float("nan")

        results[f"hd95_c{c}"] = h_val
        results[f"nsd_c{c}"]  = n_val

        dice_list.append(d)
        iou_list.append(io)
        vs_list.append(vs)
        if not np.isnan(h_val):
            hd95_list.append(h_val)
        if not np.isnan(n_val):
            nsd_list.append(n_val)

    def _safe_mean(lst: list) -> float:
        return float(np.nanmean(lst)) if lst else float("nan")

    results["dice_mean"] = _safe_mean(dice_list)
    results["iou_mean"]  = _safe_mean(iou_list)
    results["hd95_mean"] = _safe_mean(hd95_list)
    results["nsd_mean"]  = _safe_mean(nsd_list)
    results["vs_mean"]   = _safe_mean(vs_list)

    return results


# ---------------------------------------------------------------------------
# Batch convenience wrapper (for training loop)
# ---------------------------------------------------------------------------

def compute_dice_batch_3d(
    logits: torch.Tensor,
    mask: torch.Tensor,
    num_classes: int,
) -> Dict[str, float]:
    """Fast Dice for training loop (no surface metrics).

    Args:
        logits: [B, M, D, H, W]
        mask:   [B, D, H, W]  int64
    Returns:
        {'dice_mean': float, 'dice_c1': float, ...}
    """
    pred = logits.argmax(dim=1)   # [B, D, H, W]
    pred_np = pred.detach().cpu().numpy()
    mask_np = mask.detach().cpu().numpy()

    per_class: Dict[int, List[float]] = {c: [] for c in range(1, num_classes)}

    B = pred_np.shape[0]
    for b in range(B):
        for c in range(1, num_classes):
            p = (pred_np[b] == c)
            g = (mask_np[b] == c)
            if not g.any() and not p.any():
                continue
            per_class[c].append(_dice_from_volumes(p, g))

    results: Dict[str, float] = {}
    all_dice = []
    for c in range(1, num_classes):
        vals = per_class[c]
        v = float(np.mean(vals)) if vals else float("nan")
        results[f"dice_c{c}"] = v
        if not np.isnan(v):
            all_dice.append(v)

    results["dice_mean"] = float(np.mean(all_dice)) if all_dice else 0.0
    return results

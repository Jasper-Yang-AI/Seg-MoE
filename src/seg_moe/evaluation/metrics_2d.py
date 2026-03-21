"""
Comprehensive segmentation evaluation metrics.

References (科研标准级指标选择):
  - Maier-Hein et al. 2024, "Metrics Reloaded", Nature Methods
    → 推荐: DSC, NSD, HD95 为三大核心指标
  - Taha & Hanbury 2015, "Metrics for evaluating 3D medical image segmentation"
    → 系统综述 20+ 指标, 推荐 DSC + HD95 + VS
  - Isensee et al. 2021, "nnU-Net" (Nature Methods)
    → 使用 DSC + NSD 作为排名指标
  - MICCAI Challenge standard
    → DSC + HD95 为必选; NSD 逐渐普及

本模块输出指标:
  Per-class (foreground): Dice, IoU, HD95, NSD(τ=2), ASD, Sensitivity, Precision
  Aggregated:             nanmean over foreground classes
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from seg_moe.evaluation.surface_distance import surface_distances_2d


# ── Confusion matrix helpers ──────────────────────────────────────────

def dice_iou_from_confusion(
    tp: np.ndarray, fp: np.ndarray, fn: np.ndarray, eps: float = 1e-7,
) -> Tuple[np.ndarray, np.ndarray]:
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    return dice, iou


def sensitivity_precision_from_confusion(
    tp: np.ndarray, fp: np.ndarray, fn: np.ndarray, eps: float = 1e-7,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sensitivity (Recall) and Precision per class."""
    sens = (tp + eps) / (tp + fn + eps)
    prec = (tp + eps) / (tp + fp + eps)
    return sens, prec


def _safe_nanmean(values: List[float], default: float) -> float:
    if not values:
        return default
    arr = np.asarray(values, dtype=np.float64)
    if np.isnan(arr).all():
        return default
    return float(np.nanmean(arr))


# ── Main metric function ─────────────────────────────────────────────

def compute_segmentation_metrics_batch(
    probs: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    spacing_yx: Optional[Tuple[float, float]] = None,
    hd95: bool = True,
    nsd_tolerance: float = 2.0,
) -> Dict[str, Any]:
    """Compute comprehensive segmentation metrics for a batch.

    Parameters
    ----------
    probs : [B, C, H, W] tensor — softmax probabilities or one-hot
    target : [B, H, W] tensor — integer ground-truth labels
    num_classes : int
    spacing_yx : optional pixel spacing (mm) for distance metrics
    hd95 : if True use HD95 (default, MICCAI standard), else full HD
    nsd_tolerance : NSD tolerance τ in mm (default 2.0, Nikolov et al. 2021)

    Returns
    -------
    Dict with:
      Per-class:  dice_c{c}, iou_c{c}, hd95_c{c}, nsd_c{c}, asd_c{c},
                  sens_c{c}, prec_c{c}   (c = 1..C-1, foreground only)
      Aggregated: dice_mean, iou_mean, hd95_mean, nsd_mean, asd_mean,
                  sens_mean, prec_mean
    """
    pred = torch.argmax(probs, dim=1)
    pred_np = pred.detach().cpu().numpy()
    tgt_np = target.detach().cpu().numpy()

    # Accumulators per foreground class
    n_fg = num_classes - 1
    class_dices: List[List[float]] = [[] for _ in range(n_fg)]
    class_ious: List[List[float]] = [[] for _ in range(n_fg)]
    class_hds: List[List[float]] = [[] for _ in range(n_fg)]
    class_asds: List[List[float]] = [[] for _ in range(n_fg)]
    class_nsds: List[List[float]] = [[] for _ in range(n_fg)]
    class_sens: List[List[float]] = [[] for _ in range(n_fg)]
    class_precs: List[List[float]] = [[] for _ in range(n_fg)]

    for b in range(pred_np.shape[0]):
        tp = np.zeros((num_classes,), dtype=np.float64)
        fp = np.zeros((num_classes,), dtype=np.float64)
        fn = np.zeros((num_classes,), dtype=np.float64)
        for c in range(num_classes):
            p = pred_np[b] == c
            t = tgt_np[b] == c
            tp[c] = float(np.logical_and(p, t).sum())
            fp[c] = float(np.logical_and(p, ~t).sum())
            fn[c] = float(np.logical_and(~p, t).sum())

        d, j = dice_iou_from_confusion(tp, fp, fn)
        sn, pr = sensitivity_precision_from_confusion(tp, fp, fn)

        for ci, c in enumerate(range(1, num_classes)):
            class_dices[ci].append(float(d[c]))
            class_ious[ci].append(float(j[c]))
            class_sens[ci].append(float(sn[c]))
            class_precs[ci].append(float(pr[c]))

            # Surface distance metrics
            p_mask = pred_np[b] == c
            t_mask = tgt_np[b] == c

            # Handle special cases (Maier-Hein et al. 2024 recommendation):
            # Both empty → perfect (HD=0, NSD=1, ASD=0)
            # One empty → worst (HD=inf, NSD=0, ASD=inf)
            if p_mask.sum() == 0 and t_mask.sum() == 0:
                class_hds[ci].append(0.0)
                class_asds[ci].append(0.0)
                class_nsds[ci].append(1.0)
                continue
            if p_mask.sum() == 0 or t_mask.sum() == 0:
                class_hds[ci].append(np.nan)
                class_asds[ci].append(np.nan)
                class_nsds[ci].append(0.0)
                continue

            sd = surface_distances_2d(p_mask, t_mask, spacing_yx=spacing_yx)
            if sd is None:
                class_hds[ci].append(np.nan)
                class_asds[ci].append(np.nan)
                class_nsds[ci].append(np.nan)
            else:
                class_hds[ci].append(float(sd["hd95"] if hd95 else sd["hd"]))
                class_asds[ci].append(float(sd["mad"]))
                # NSD at specified tolerance
                nsd_key = "nsd_2" if abs(nsd_tolerance - 2.0) < 0.5 else "nsd_1"
                class_nsds[ci].append(float(sd.get(nsd_key, sd.get("nsd_2", 0.0))))

    # ── Build output dict ──
    result: Dict[str, Any] = {}

    # Per-class metrics
    all_dices, all_ious, all_hds, all_asds, all_nsds = [], [], [], [], []
    all_sens, all_precs = [], []

    for ci, c in enumerate(range(1, num_classes)):
        d_val = _safe_nanmean(class_dices[ci], 0.0)
        j_val = _safe_nanmean(class_ious[ci], 0.0)
        h_val = _safe_nanmean(class_hds[ci], float("nan"))
        a_val = _safe_nanmean(class_asds[ci], float("nan"))
        n_val = _safe_nanmean(class_nsds[ci], 0.0)
        s_val = _safe_nanmean(class_sens[ci], 0.0)
        p_val = _safe_nanmean(class_precs[ci], 0.0)

        result[f"dice_c{c}"] = d_val
        result[f"iou_c{c}"] = j_val
        result[f"hd95_c{c}"] = h_val
        result[f"asd_c{c}"] = a_val
        result[f"nsd_c{c}"] = n_val
        result[f"sens_c{c}"] = s_val
        result[f"prec_c{c}"] = p_val

        all_dices.append(d_val)
        all_ious.append(j_val)
        all_hds.append(h_val)
        all_asds.append(a_val)
        all_nsds.append(n_val)
        all_sens.append(s_val)
        all_precs.append(p_val)

    # Aggregated (nanmean over foreground classes)
    result["dice_mean"] = _safe_nanmean(all_dices, 0.0)
    result["iou_mean"] = _safe_nanmean(all_ious, 0.0)
    result["hd95_mean"] = _safe_nanmean(all_hds, float("nan"))
    result["asd_mean"] = _safe_nanmean(all_asds, float("nan"))
    result["nsd_mean"] = _safe_nanmean(all_nsds, 0.0)
    result["sens_mean"] = _safe_nanmean(all_sens, 0.0)
    result["prec_mean"] = _safe_nanmean(all_precs, 0.0)

    # Backward compat aliases
    result["hd_mean"] = result["hd95_mean"]
    result["mad_mean"] = result["asd_mean"]

    return result

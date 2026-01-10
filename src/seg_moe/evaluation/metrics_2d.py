from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch

from seg_moe.evaluation.surface_distance import surface_distances_2d


def dice_iou_from_confusion(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray, eps: float = 1e-7) -> tuple[np.ndarray, np.ndarray]:
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    return dice, iou


def compute_segmentation_metrics_batch(
    probs: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    spacing_yx: Optional[Tuple[float, float]] = None,
    hd95: bool = False,
) -> Dict[str, float]:
    """Compute metrics for a batch.

    probs: [B,C,H,W]
    target: [B,H,W]

    Returns mean across classes excluding background by default at caller.
    This function returns overall means across foreground classes (1..C-1).
    """

    pred = torch.argmax(probs, dim=1)
    pred_np = pred.detach().cpu().numpy()
    tgt_np = target.detach().cpu().numpy()

    dices = []
    ious = []
    hds = []
    mads = []

    for b in range(pred_np.shape[0]):
        tp = np.zeros((num_classes,), dtype=np.float64)
        fp = np.zeros((num_classes,), dtype=np.float64)
        fn = np.zeros((num_classes,), dtype=np.float64)
        for c in range(num_classes):
            p = pred_np[b] == c
            t = tgt_np[b] == c
            tp[c] = float(np.logical_and(p, t).sum())
            fp[c] = float(np.logical_and(p, np.logical_not(t)).sum())
            fn[c] = float(np.logical_and(np.logical_not(p), t).sum())

        d, j = dice_iou_from_confusion(tp, fp, fn)

        # distances (skip background=0)
        for c in range(1, num_classes):
            dices.append(float(d[c]))
            ious.append(float(j[c]))

            sd = surface_distances_2d(pred_np[b] == c, tgt_np[b] == c, spacing_yx=spacing_yx)
            if sd is None:
                # If both empty or one empty, define as nan and handle later
                hds.append(np.nan)
                mads.append(np.nan)
            else:
                hds.append(float(sd["hd95"] if hd95 else sd["hd"]))
                mads.append(float(sd["mad"]))

    return {
        "dice_mean": float(np.nanmean(dices)) if dices else 0.0,
        "iou_mean": float(np.nanmean(ious)) if ious else 0.0,
        "hd_mean": float(np.nanmean(hds)) if hds else float("nan"),
        "mad_mean": float(np.nanmean(mads)) if mads else float("nan"),
    }

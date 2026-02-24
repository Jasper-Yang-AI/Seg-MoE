from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree
from skimage.segmentation import find_boundaries


def _boundary_points(mask: np.ndarray) -> np.ndarray:
    b = find_boundaries(mask.astype(bool), mode="outer")
    pts = np.argwhere(b)
    return pts.astype(np.float64)


def _nearest_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # return for each point in a: distance to nearest in b
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.array([], dtype=np.float64)
    tree = cKDTree(b)
    d, _ = tree.query(a, k=1)
    return d.astype(np.float64)


def surface_distances_2d(
    pred_mask: np.ndarray,
    true_mask: np.ndarray,
    spacing_yx: Optional[Tuple[float, float]] = None,
) -> Optional[Dict[str, float]]:
    """Compute symmetric surface distances between two binary masks.

    References:
      - Maier-Hein et al. 2024, "Metrics Reloaded", Nature Methods
      - Nikolov et al. 2021, "Clinically Applicable Segmentation ...", Nature Machine Intelligence

    Returns
    -------
    None if both masks are empty or one is empty (undefined distances).
    Else dict with:
      - mad: symmetric average surface distance (ASD)
      - hd: symmetric Hausdorff distance (full)
      - hd95: symmetric 95th percentile Hausdorff
      - nsd_1: Normalized Surface Dice at τ=1mm (or 1 pixel if no spacing)
      - nsd_2: Normalized Surface Dice at τ=2mm (or 2 pixels if no spacing)
    """

    pred = pred_mask.astype(bool)
    true = true_mask.astype(bool)

    if pred.sum() == 0 and true.sum() == 0:
        return None
    if pred.sum() == 0 or true.sum() == 0:
        return None

    p_pts = _boundary_points(pred)
    t_pts = _boundary_points(true)

    if p_pts.shape[0] == 0 or t_pts.shape[0] == 0:
        return None

    # apply spacing (y,x)
    if spacing_yx is not None:
        sy, sx = float(spacing_yx[0]), float(spacing_yx[1])
        p_pts = p_pts * np.array([sy, sx], dtype=np.float64)
        t_pts = t_pts * np.array([sy, sx], dtype=np.float64)

    d_pt = _nearest_distances(p_pts, t_pts)
    d_tp = _nearest_distances(t_pts, p_pts)

    if d_pt.size == 0 or d_tp.size == 0:
        return None

    all_d = np.concatenate([d_pt, d_tp], axis=0)

    mad = float(all_d.mean())
    hd = float(all_d.max())
    hd95 = float(np.percentile(all_d, 95))

    # ── NSD (Normalized Surface Dice) ──
    # Nikolov et al. 2021; Maier-Hein et al. 2024
    # NSD(τ) = |{p ∈ ∂P : d(p, ∂T) ≤ τ}| + |{t ∈ ∂T : d(t, ∂P) ≤ τ}|
    #          / (|∂P| + |∂T|)
    nsd_1 = _compute_nsd(d_pt, d_tp, tau=1.0)
    nsd_2 = _compute_nsd(d_pt, d_tp, tau=2.0)

    return {"mad": mad, "hd": hd, "hd95": hd95, "nsd_1": nsd_1, "nsd_2": nsd_2}


def _compute_nsd(d_pt: np.ndarray, d_tp: np.ndarray, tau: float) -> float:
    """Normalized Surface Dice at tolerance τ.

    Fraction of boundary points within distance τ of the other boundary.
    """
    n_pt = d_pt.size
    n_tp = d_tp.size
    if n_pt + n_tp == 0:
        return 0.0
    within_pt = float(np.sum(d_pt <= tau))
    within_tp = float(np.sum(d_tp <= tau))
    return (within_pt + within_tp) / (n_pt + n_tp)

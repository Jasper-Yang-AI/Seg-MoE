from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple, Union

import numpy as np


ArrayLikePreds = Union[np.ndarray, Iterable[Tuple[np.ndarray, np.ndarray]]]


def _flatten_preds_gt(preds: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if preds.ndim < 3:
        raise ValueError(f"oof_preds must have at least 3 dims [...,K,M], got {preds.shape}")
    K = preds.shape[-2]
    M = preds.shape[-1]
    preds2 = preds.reshape(-1, K, M)
    gt2 = gt.reshape(-1)
    if preds2.shape[0] != gt2.shape[0]:
        raise ValueError(f"oof_preds and gt pixel counts mismatch: {preds2.shape[0]} vs {gt2.shape[0]}")
    return preds2, gt2


def _maybe_sample_global(
    preds: np.ndarray,
    gt: np.ndarray,
    *,
    rng: np.random.Generator,
    pixel_sample_ratio: Optional[float],
) -> tuple[np.ndarray, np.ndarray]:
    if pixel_sample_ratio is None:
        return preds, gt
    r = float(pixel_sample_ratio)
    if not (0.0 < r <= 1.0):
        raise ValueError(f"pixel_sample_ratio must be in (0,1], got {r}")
    if r >= 1.0:
        return preds, gt

    n = preds.shape[0]
    keep = max(1, int(round(n * r)))
    idx = rng.choice(n, size=keep, replace=False)
    return preds[idx], gt[idx]


def _maybe_sample_per_class(
    preds: np.ndarray,
    gt: np.ndarray,
    *,
    rng: np.random.Generator,
    num_classes: int,
    max_pixels_per_class: Optional[int],
) -> tuple[np.ndarray, np.ndarray]:
    if max_pixels_per_class is None:
        return preds, gt

    max_pos = int(max_pixels_per_class)
    if max_pos <= 0:
        raise ValueError(f"max_pixels_per_class must be positive, got {max_pos}")

    n = preds.shape[0]
    keep_mask = np.zeros((n,), dtype=bool)
    all_idx = np.arange(n)

    for m in range(int(num_classes)):
        pos_idx = all_idx[gt == m]
        if pos_idx.size == 0:
            continue
        pos_keep = pos_idx if pos_idx.size <= max_pos else rng.choice(pos_idx, size=max_pos, replace=False)

        neg_idx = all_idx[gt != m]
        neg_keep_n = min(int(pos_keep.size), int(neg_idx.size))
        neg_keep = neg_idx if neg_keep_n == neg_idx.size else rng.choice(neg_idx, size=neg_keep_n, replace=False)

        keep_mask[pos_keep] = True
        keep_mask[neg_keep] = True

    idx = np.where(keep_mask)[0]
    if idx.size == 0:
        return preds, gt
    return preds[idx], gt[idx]


def fit_from_oof(
    oof_preds: ArrayLikePreds,
    gt: Optional[np.ndarray] = None,
    *,
    bounds: tuple[float, float] = (0.0, 1.0),
    method: str = "bvls",
    pixel_sample_ratio: Optional[float] = None,
    max_pixels_per_class: Optional[int] = None,
    seed: int = 42,
) -> np.ndarray:
    """Paper-style per-class least squares weight fitting.

    oof_preds can be:
    - np.ndarray with shape [...,K,M] and gt with shape [...]
    - iterable yielding (preds_chunk, gt_chunk)

    Returns W with shape [K,M].
    """

    rng = np.random.default_rng(int(seed))

    if isinstance(oof_preds, np.ndarray):
        if gt is None:
            raise ValueError("gt is required when oof_preds is a numpy array")
        preds_flat, gt_flat = _flatten_preds_gt(oof_preds, gt)
    else:
        if gt is not None:
            raise ValueError("gt must be None when oof_preds is an iterable of (preds, gt) pairs")
        preds_list = []
        gt_list = []
        for p_chunk, g_chunk in oof_preds:
            p2, g2 = _flatten_preds_gt(np.asarray(p_chunk), np.asarray(g_chunk))
            p2, g2 = _maybe_sample_global(p2, g2, rng=rng, pixel_sample_ratio=pixel_sample_ratio)
            preds_list.append(p2)
            gt_list.append(g2)
        if not preds_list:
            raise ValueError("oof_preds iterable produced no data")
        preds_flat = np.concatenate(preds_list, axis=0)
        gt_flat = np.concatenate(gt_list, axis=0)

    preds_flat, gt_flat = _maybe_sample_global(preds_flat, gt_flat, rng=rng, pixel_sample_ratio=pixel_sample_ratio)

    _, K, M = preds_flat.shape
    if gt_flat.min() < 0 or gt_flat.max() >= M:
        raise ValueError(f"gt contains labels outside [0,{M-1}]")

    preds_flat, gt_flat = _maybe_sample_per_class(
        preds_flat,
        gt_flat,
        rng=rng,
        num_classes=M,
        max_pixels_per_class=max_pixels_per_class,
    )

    method = str(method).lower()
    l, u = float(bounds[0]), float(bounds[1])

    X_all = preds_flat.astype(np.float64)  # [N,K,M]
    w = np.zeros((K, M), dtype=np.float64)

    if method == "ls":
        for m in range(M):
            Xm = X_all[:, :, m]
            ym = (gt_flat == m).astype(np.float64)
            sol, *_ = np.linalg.lstsq(Xm, ym, rcond=None)
            w[:, m] = sol
    elif method == "nnls":
        from scipy.optimize import nnls

        for m in range(M):
            Xm = X_all[:, :, m]
            ym = (gt_flat == m).astype(np.float64)
            sol, _ = nnls(Xm, ym)
            w[:, m] = sol
    elif method == "bvls":
        from scipy.optimize import lsq_linear

        for m in range(M):
            Xm = X_all[:, :, m]
            ym = (gt_flat == m).astype(np.float64)
            res = lsq_linear(Xm, ym, bounds=(l, u))
            w[:, m] = res.x
    else:
        raise ValueError(f"Unknown method: {method} (use 'ls', 'nnls', or 'bvls')")

    return w.astype(np.float32)


def fuse(preds: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Linear fusion without dividing by sum(W)."""
    if preds.ndim < 2:
        raise ValueError(f"preds must have at least 2 dims [...,K,M], got {preds.shape}")
    if W.ndim != 2:
        raise ValueError(f"W must have shape [K,M], got {W.shape}")
    if preds.shape[-2] != W.shape[0] or preds.shape[-1] != W.shape[1]:
        raise ValueError(f"preds[...,K,M] and W[K,M] mismatch: preds={preds.shape}, W={W.shape}")
    return np.einsum("...km,km->...m", preds.astype(np.float64), W.astype(np.float64)).astype(np.float32)


def predict(preds: np.ndarray, W: np.ndarray) -> np.ndarray:
    scores = fuse(preds, W)
    return np.argmax(scores, axis=-1).astype(np.int64)


@dataclass
class OLEWeights:
    # shape: [K, M]
    w: np.ndarray


class OLECombiner:
    """One-Layer Ensemble weight-based combiner.

    For each class m, learn a non-negative bounded weight vector w_m (length K).
    Fusion (paper-style): score[m] = sum_k w[k,m] * p_k[m] (per-pixel).
    Note: We intentionally DO NOT divide by sum(w) and do not renormalize.

    Two modes are supported:
    - lsq_bounded: paper-style bounded least squares (bvls, bounds=[0,1])
    - sgd_conv1x1: legacy/debug SGD approximation (still bounded to [0,1])

    Notes
    -----
    This is a practical, runnable approximation aligned with the paper's weight-based combining.
    """

    def __init__(self, mode: str = "lsq_bounded", max_iter: int = 2000, lr: float = 1e-2, seed: int = 42):
        self.mode = mode
        self.max_iter = int(max_iter)
        self.lr = float(lr)
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.weights: Optional[OLEWeights] = None

    def fit(self, probs: np.ndarray, target: np.ndarray, num_classes: int) -> OLEWeights:
        """Fit weights.

        Parameters
        ----------
        probs: [N, K, M] flattened per-pixel or per-sample class probs
        target: [N] int labels (0..M-1)
        """
        N, K, M = probs.shape
        assert M == num_classes
        y = np.eye(M, dtype=np.float64)[target.astype(int)]  # [N,M]
        X = probs.astype(np.float64)  # [N,K,M]

        w = np.zeros((K, M), dtype=np.float64)

        if self.mode == "lsq_bounded":
            # Strict paper-style per-class bounded least squares on OOF pixels.
            w = fit_from_oof(X.astype(np.float32), target.astype(np.int64), bounds=(0.0, 1.0), method="bvls", seed=self.seed).astype(np.float64)
        elif self.mode == "sgd_conv1x1":
            # SGD on mean squared error with projection to [0,1]
            w = self.rng.uniform(0.0, 1.0, size=(K, M)).astype(np.float64)
            for _ in range(self.max_iter):
                pred = np.einsum("nkm,km->nm", X, w)  # [N,M]
                grad = (2.0 / N) * np.einsum("nkm,nm->km", X, (pred - y))
                w -= self.lr * grad
                w = np.clip(w, 0.0, 1.0)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        self.weights = OLEWeights(w=w.astype(np.float32))
        return self.weights

    def predict(self, probs: np.ndarray) -> np.ndarray:
        """Fuse K expert probs into fused scores.

        probs: [..., K, M] -> [..., M]
        """
        if self.weights is None:
            raise RuntimeError("OLECombiner not fitted")
        w = self.weights.w.astype(np.float64)  # [K,M]
        p = probs.astype(np.float64)
        fused = np.einsum("...km,km->...m", p, w)
        return fused.astype(np.float32)

    def export_weights_table(self, expert_names: list[str], class_names: list[str]) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("OLECombiner not fitted")
        assert self.weights.w.shape == (len(expert_names), len(class_names))
        return self.weights.w

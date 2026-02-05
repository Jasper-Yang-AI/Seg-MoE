from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class OLEWeights:
    # shape: [K, M]
    w: np.ndarray


class OLECombiner:
    """One-Layer Ensemble weight-based combiner.

    For each class m, learn a non-negative bounded weight vector w_m (length K).
    Fusion: p_fused[m] = sum_k w[k,m] * p_k[m] / sum_k w[k,m] (per-pixel).

    Two modes are supported:
    - lsq_bounded: bounded least squares via scipy.optimize.lsq_linear
    - sgd_conv1x1: simple SGD on 1x1 conv weights (here implemented as numpy SGD)

    Notes
    -----
    This is a practical, runnable approximation aligned with the paper's weight-based combining.
    """

    def __init__(self, mode: str = "lsq_bounded", max_iter: int = 2000, lr: float = 1e-2, seed: int = 42):
        self.mode = mode
        self.max_iter = int(max_iter)
        self.lr = float(lr)
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
            from scipy.optimize import lsq_linear

            for m in range(M):
                Xm = X[:, :, m]  # [N,K]
                ym = y[:, m]  # [N]
                res = lsq_linear(Xm, ym, bounds=(0.0, 1.0), max_iter=self.max_iter)
                w[:, m] = res.x
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
        """Fuse K expert probs into fused probs.

        probs: [..., K, M] -> [..., M]
        """
        if self.weights is None:
            raise RuntimeError("OLECombiner not fitted")
        w = self.weights.w.astype(np.float64)  # [K,M]
        p = probs.astype(np.float64)
        num = np.einsum("...km,km->...m", p, w)
        den = np.sum(w, axis=0, keepdims=False)  # [M]
        den = np.maximum(den, 1e-8)
        fused = num / den
        # renormalize
        fused = fused / np.maximum(fused.sum(axis=-1, keepdims=True), 1e-8)
        return fused.astype(np.float32)

    def export_weights_table(self, expert_names: list[str], class_names: list[str]) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("OLECombiner not fitted")
        assert self.weights.w.shape == (len(expert_names), len(class_names))
        return self.weights.w

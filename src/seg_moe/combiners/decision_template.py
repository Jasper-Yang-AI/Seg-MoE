from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, Union

import numpy as np


ArrayLikePreds = Union[np.ndarray, Iterable[Tuple[np.ndarray, np.ndarray]]]


def _flatten_preds_gt(preds: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if preds.ndim < 3:
        raise ValueError(f"probs must have at least 3 dims [...,K,M], got {preds.shape}")
    k_experts = preds.shape[-2]
    num_classes = preds.shape[-1]
    preds2 = preds.reshape(-1, k_experts, num_classes)
    gt2 = gt.reshape(-1)
    if preds2.shape[0] != gt2.shape[0]:
        raise ValueError(f"probs and gt pixel counts mismatch: {preds2.shape[0]} vs {gt2.shape[0]}")
    return preds2, gt2


def _iter_flat_chunks(
    probs: ArrayLikePreds,
    target: Optional[np.ndarray] = None,
) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    if isinstance(probs, np.ndarray):
        if target is None:
            raise ValueError("target is required when probs is a numpy array")
        yield _flatten_preds_gt(probs, target)
        return

    if target is not None:
        raise ValueError("target must be None when probs is an iterable of (preds, gt) pairs")

    yielded = False
    for p_chunk, g_chunk in probs:
        yielded = True
        yield _flatten_preds_gt(np.asarray(p_chunk), np.asarray(g_chunk))

    if not yielded:
        raise ValueError("probs iterable produced no data")


@dataclass
class DecisionTemplateState:
    # templates: [M, K, M] (class -> (experts x class-prob-vector))
    templates: np.ndarray


class DecisionTemplateCombiner:
    """Decision Template (DT) combiner.

    Training:
      For each true class c, compute the mean decision profile of pixels with label c.
      Decision profile at a pixel is a concatenation of per-expert class probability vectors.

    Inference:
      For a pixel, compute distance to each class template and choose argmin.

    Notes
    -----
    This implementation uses squared Euclidean distance.
    """

    def __init__(self):
        self.state: Optional[DecisionTemplateState] = None

    def fit(
        self,
        probs: ArrayLikePreds,
        target: Optional[np.ndarray],
        num_classes: int,
    ) -> DecisionTemplateState:
        """Fit templates.

        probs: [N,K,M]
        target: [N]
        """
        templates: Optional[np.ndarray] = None
        counts = np.zeros((num_classes,), dtype=np.int64)
        global_sum: Optional[np.ndarray] = None
        global_count = 0

        for probs_flat, target_flat in _iter_flat_chunks(probs, target):
            if probs_flat.size == 0:
                continue

            _, k_experts, m_classes = probs_flat.shape
            if m_classes != num_classes:
                raise ValueError(f"Expected {num_classes} classes, got {m_classes}")

            if templates is None or global_sum is None:
                templates = np.zeros((num_classes, k_experts, num_classes), dtype=np.float64)
                global_sum = np.zeros((k_experts, num_classes), dtype=np.float64)

            global_sum += probs_flat.sum(axis=0, dtype=np.float64)
            global_count += int(probs_flat.shape[0])

            for c in range(num_classes):
                yc = target_flat == c
                counts[c] += int(np.sum(yc))
                if np.any(yc):
                    templates[c] += probs_flat[yc].sum(axis=0, dtype=np.float64)

        if templates is None or global_sum is None or global_count == 0:
            raise ValueError("No OOF pixels found for fitting Decision Template")

        global_mean = global_sum / float(global_count)
        for c in range(num_classes):
            if counts[c] == 0:
                templates[c] = global_mean
            else:
                templates[c] /= float(counts[c])

        self.state = DecisionTemplateState(templates=templates.astype(np.float32))
        return self.state

    def predict(self, probs: np.ndarray) -> np.ndarray:
        """Return soft probability-like scores (negative distance → similarity).

        Converts squared Euclidean distances to class templates into
        probability-like scores via softmax over negative distances:
          score[c] = exp(-dist[c] / τ) / Σ_c' exp(-dist[c'] / τ)

        This allows downstream evaluation to use soft predictions
        rather than hard argmax, preserving information for metrics.

        probs: [...,K,M]
        returns: [...,M] float32 probability-like scores
        """
        if self.state is None:
            raise RuntimeError("DecisionTemplateCombiner not fitted")
        templates = self.state.templates.astype(np.float64)  # [M,K,M]
        x = probs.astype(np.float64)
        # dist[...,c] = sum_{k,m} (x[...,k,m] - T[c,k,m])^2
        dist = np.sum((x[..., None, :, :] - templates[None, ...]) ** 2, axis=(-2, -1))
        # Convert distances to similarities via softmax over negative distances
        # Use temperature τ = mean(dist) for numerical stability
        tau = np.maximum(np.mean(dist), 1e-8)
        neg_dist = -dist / tau
        neg_dist -= neg_dist.max(axis=-1, keepdims=True)  # stable softmax
        exp_nd = np.exp(neg_dist)
        scores = exp_nd / np.maximum(exp_nd.sum(axis=-1, keepdims=True), 1e-8)
        return scores.astype(np.float32)

    def predict_hard(self, probs: np.ndarray) -> np.ndarray:
        """Return predicted class indices (legacy behavior).

        probs: [...,K,M]
        returns: [...] int64
        """
        if self.state is None:
            raise RuntimeError("DecisionTemplateCombiner not fitted")
        templates = self.state.templates.astype(np.float64)
        x = probs.astype(np.float64)
        dist = np.sum((x[..., None, :, :] - templates[None, ...]) ** 2, axis=(-2, -1))
        return np.argmin(dist, axis=-1).astype(np.int64)

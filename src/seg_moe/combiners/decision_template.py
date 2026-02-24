from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


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

    def fit(self, probs: np.ndarray, target: np.ndarray, num_classes: int) -> DecisionTemplateState:
        """Fit templates.

        probs: [N,K,M]
        target: [N]
        """
        N, K, M = probs.shape
        assert M == num_classes

        templates = np.zeros((M, K, M), dtype=np.float64)
        counts = np.zeros((M,), dtype=np.int64)

        for c in range(M):
            idx = np.where(target == c)[0]
            counts[c] = idx.size
            if idx.size == 0:
                continue
            templates[c] = probs[idx].mean(axis=0)

        # handle empty class by global mean
        global_mean = probs.mean(axis=0)
        for c in range(M):
            if counts[c] == 0:
                templates[c] = global_mean

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

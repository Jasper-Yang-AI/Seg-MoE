"""
Majority Voting Combiner — 最基础的多数投票集成基线.

References:
  - Kittler et al. 1998, "On combining classifiers" (IEEE TPAMI)
    → Majority Vote 为所有集成方法的 lower bound baseline
  - Dang et al. 2024, "Two-layer Ensemble of Deep Learning Models"
    → 论文 Table 4 对比 simple average / OLE / DT / WE-CLPSO,
      但缺少 majority voting baseline (本文补充)

Hard majority voting:
  每个像素, 每个专家预测一个类别 (argmax of probs), 取出现次数最多的类别.
  若票数相同, 选择 softmax 概率之和最大的类别 (soft tie-breaking).

Soft majority voting (predict_proba):
  将 per-expert one-hot 投票做平均, 返回概率向量.
  等价于对硬预测做平均 (区别于对 softmax probs 做平均).
"""
from __future__ import annotations

from typing import Optional

import numpy as np


class MajorityVotingCombiner:
    """Majority Voting (hard + soft tie-breaking).

    No fitting required — purely a prediction-time combiner.
    Included as the most basic ensemble baseline for fair comparison.
    """

    def __init__(self) -> None:
        self.num_classes: Optional[int] = None

    def fit(
        self,
        probs: np.ndarray,
        target: np.ndarray,
        num_classes: int,
    ) -> None:
        """No-op fit (for API consistency with other combiners)."""
        self.num_classes = num_classes

    def predict(self, probs: np.ndarray) -> np.ndarray:
        """Hard majority voting with soft tie-breaking.

        Parameters
        ----------
        probs : [..., K, M]  per-expert softmax probabilities

        Returns
        -------
        fused : [..., M]  probability-like vector (averaged one-hot votes)
        """
        K = probs.shape[-2]
        M = probs.shape[-1]

        # Hard votes: each expert's argmax
        votes = np.argmax(probs, axis=-1)  # [..., K]

        # Count votes per class
        prefix_shape = probs.shape[:-2]
        flat_votes = votes.reshape(-1, K)
        N = flat_votes.shape[0]

        # One-hot encode votes → average to get vote fractions
        vote_counts = np.zeros((N, M), dtype=np.float64)
        for k in range(K):
            for i in range(N):
                vote_counts[i, flat_votes[i, k]] += 1.0

        # Soft tie-breaking: add tiny amount of softmax probs sum for tie resolution
        flat_probs = probs.reshape(N, K, M).astype(np.float64)
        prob_sum = flat_probs.sum(axis=1)  # [N, M]
        # Normalize prob_sum to be << 1 vote (max contribution = 0.1)
        max_prob_sum = np.maximum(prob_sum.max(axis=-1, keepdims=True), 1e-8)
        tie_breaker = 0.1 * prob_sum / max_prob_sum

        fused = vote_counts + tie_breaker
        # Normalize to probabilities
        fused = fused / np.maximum(fused.sum(axis=-1, keepdims=True), 1e-8)

        return fused.reshape(*prefix_shape, M).astype(np.float32)

    def predict_hard(self, probs: np.ndarray) -> np.ndarray:
        """Return argmax class indices directly.

        Parameters
        ----------
        probs : [..., K, M]

        Returns
        -------
        pred : [...]  int64 class indices
        """
        fused = self.predict(probs)
        return np.argmax(fused, axis=-1).astype(np.int64)

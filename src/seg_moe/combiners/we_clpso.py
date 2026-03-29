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
class PSOState:
    w: np.ndarray  # [K,M]
    best_score: float


class WECLPSOCombiner:
    """WE-CLPSO (simplified runnable) weight optimization via PSO.

    The paper uses CLPSO. Here we implement a practical approximation:
    - Particle Swarm Optimization with per-dimension "comprehensive learning"-like mixing.

    Objective (default): maximize mean Dice on validation pixels.
    For efficiency and simplicity, we optimize weights on flattened pixels.

    Constraints: weights in [0,1].
    """

    def __init__(
        self,
        n_particles: int = 20,
        iters: int = 100,
        w_inertia: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
        seed: int = 42,
    ):
        self.n_particles = int(n_particles)
        self.iters = int(iters)
        self.w_inertia = float(w_inertia)
        self.c1 = float(c1)
        self.c2 = float(c2)
        self.rng = np.random.default_rng(seed)
        self.state: Optional[PSOState] = None

    @staticmethod
    def _fuse_probs(probs: np.ndarray, w: np.ndarray) -> np.ndarray:
        # probs: [N,K,M], w: [K,M] -> fused: [N,M]
        num = np.einsum("nkm,km->nm", probs, w)
        den = np.maximum(np.sum(w, axis=0, keepdims=False), 1e-8)
        fused = num / den
        fused = fused / np.maximum(fused.sum(axis=-1, keepdims=True), 1e-8)
        return fused

    @staticmethod
    def _fuse_scores_multi(probs: np.ndarray, weights: np.ndarray) -> np.ndarray:
        # probs: [N,K,M], weights: [P,K,M] -> scores: [P,N,M]
        probs32 = probs.astype(np.float32, copy=False)
        weights32 = weights.astype(np.float32, copy=False)
        num = np.einsum("nkm,pkm->pnm", probs32, weights32, optimize=True)
        den = np.maximum(np.sum(weights32, axis=1), 1e-8)
        return num / den[:, None, :]

    @staticmethod
    def _dice_mean_from_pred(pred: np.ndarray, target: np.ndarray, num_classes: int) -> float:
        dices = []
        for c in range(1, num_classes):
            p = pred == c
            t = target == c
            tp = np.logical_and(p, t).sum()
            fp = np.logical_and(p, np.logical_not(t)).sum()
            fn = np.logical_and(np.logical_not(p), t).sum()
            d = (2 * tp + 1e-7) / (2 * tp + fp + fn + 1e-7)
            dices.append(float(d))
        return float(np.mean(dices)) if dices else 0.0

    def _infer_shape(
        self,
        probs: ArrayLikePreds,
        target: Optional[np.ndarray],
    ) -> tuple[int, int]:
        for probs_flat, _ in _iter_flat_chunks(probs, target):
            return int(probs_flat.shape[1]), int(probs_flat.shape[2])
        raise ValueError("No OOF pixels found for WE-CLPSO fitting")

    def _score_particles(
        self,
        probs: ArrayLikePreds,
        target: Optional[np.ndarray],
        weights: np.ndarray,
        num_classes: int,
    ) -> np.ndarray:
        n_particles = weights.shape[0]
        tp = np.zeros((n_particles, num_classes), dtype=np.int64)
        fp = np.zeros((n_particles, num_classes), dtype=np.int64)
        fn = np.zeros((n_particles, num_classes), dtype=np.int64)

        for probs_flat, target_flat in _iter_flat_chunks(probs, target):
            if probs_flat.size == 0:
                continue

            pred = np.argmax(self._fuse_scores_multi(probs_flat, weights), axis=-1)  # [P,N]
            target_view = target_flat[None, :]
            for c in range(1, num_classes):
                pred_c = pred == c
                true_c = target_view == c
                tp[:, c] += np.logical_and(pred_c, true_c).sum(axis=1, dtype=np.int64)
                fp[:, c] += np.logical_and(pred_c, np.logical_not(true_c)).sum(axis=1, dtype=np.int64)
                fn[:, c] += np.logical_and(np.logical_not(pred_c), true_c).sum(axis=1, dtype=np.int64)

        if num_classes <= 1:
            return np.zeros((n_particles,), dtype=np.float64)

        dices = (2.0 * tp[:, 1:] + 1e-7) / (2.0 * tp[:, 1:] + fp[:, 1:] + fn[:, 1:] + 1e-7)
        return dices.mean(axis=1).astype(np.float64)

    def fit(
        self,
        probs: ArrayLikePreds,
        target: Optional[np.ndarray],
        num_classes: int,
    ) -> PSOState:
        """Optimize weights.

        probs: [N,K,M]
        target: [N]
        """
        k_experts, m_classes = self._infer_shape(probs, target)
        assert m_classes == num_classes

        dim = k_experts * num_classes
        # init
        X = self.rng.uniform(0.0, 1.0, size=(self.n_particles, dim)).astype(np.float64)
        V = self.rng.normal(0.0, 0.1, size=(self.n_particles, dim)).astype(np.float64)

        pbest = X.copy()
        pbest_score = np.full((self.n_particles,), -np.inf, dtype=np.float64)

        # evaluate initial
        weights = X.reshape(self.n_particles, k_experts, num_classes)
        pbest_score = self._score_particles(probs, target, weights, num_classes=num_classes)

        g_idx = int(np.argmax(pbest_score))
        gbest = pbest[g_idx].copy()
        gbest_score = float(pbest_score[g_idx])

        # PSO loop
        for _ in range(self.iters):
            r1 = self.rng.uniform(size=(self.n_particles, dim))
            r2 = self.rng.uniform(size=(self.n_particles, dim))

            V = self.w_inertia * V + self.c1 * r1 * (pbest - X) + self.c2 * r2 * (gbest[None, :] - X)
            X = X + V
            X = np.clip(X, 0.0, 1.0)

            # "comprehensive learning"-like dimension mixing: randomly copy some dims from random pbest
            mix_prob = 0.05
            mix_mask = self.rng.uniform(size=(self.n_particles, dim)) < mix_prob
            donors = self.rng.integers(0, self.n_particles, size=(self.n_particles,))
            for i in range(self.n_particles):
                m = mix_mask[i]
                if np.any(m):
                    X[i, m] = pbest[donors[i], m]

            scores = self._score_particles(
                probs,
                target,
                X.reshape(self.n_particles, k_experts, num_classes),
                num_classes=num_classes,
            )
            improved = scores > pbest_score
            pbest_score[improved] = scores[improved]
            pbest[improved] = X[improved].copy()

            g_idx = int(np.argmax(pbest_score))
            if float(pbest_score[g_idx]) > gbest_score:
                gbest_score = float(pbest_score[g_idx])
                gbest = pbest[g_idx].copy()

        w_best = gbest.reshape(k_experts, num_classes).astype(np.float32)
        self.state = PSOState(w=w_best, best_score=float(gbest_score))
        return self.state

    def predict(self, probs: np.ndarray) -> np.ndarray:
        if self.state is None:
            raise RuntimeError("WECLPSOCombiner not fitted")
        fused = self._fuse_probs(probs.astype(np.float64), self.state.w.astype(np.float64))
        return fused.astype(np.float32)

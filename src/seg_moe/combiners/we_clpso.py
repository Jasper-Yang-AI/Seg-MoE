from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


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

    def fit(self, probs: np.ndarray, target: np.ndarray, num_classes: int) -> PSOState:
        """Optimize weights.

        probs: [N,K,M]
        target: [N]
        """
        N, K, M = probs.shape
        assert M == num_classes

        dim = K * M
        # init
        X = self.rng.uniform(0.0, 1.0, size=(self.n_particles, dim)).astype(np.float64)
        V = self.rng.normal(0.0, 0.1, size=(self.n_particles, dim)).astype(np.float64)

        pbest = X.copy()
        pbest_score = np.full((self.n_particles,), -np.inf, dtype=np.float64)

        def score(x_flat: np.ndarray) -> float:
            w = x_flat.reshape(K, M)
            fused = self._fuse_probs(probs.astype(np.float64), w)
            pred = np.argmax(fused, axis=-1)
            return self._dice_mean_from_pred(pred, target, num_classes=num_classes)

        # evaluate initial
        for i in range(self.n_particles):
            s = score(X[i])
            pbest_score[i] = s

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

            for i in range(self.n_particles):
                s = score(X[i])
                if s > pbest_score[i]:
                    pbest_score[i] = s
                    pbest[i] = X[i].copy()

            g_idx = int(np.argmax(pbest_score))
            if float(pbest_score[g_idx]) > gbest_score:
                gbest_score = float(pbest_score[g_idx])
                gbest = pbest[g_idx].copy()

        w_best = gbest.reshape(K, M).astype(np.float32)
        self.state = PSOState(w=w_best, best_score=float(gbest_score))
        return self.state

    def predict(self, probs: np.ndarray) -> np.ndarray:
        if self.state is None:
            raise RuntimeError("WECLPSOCombiner not fitted")
        fused = self._fuse_probs(probs.astype(np.float64), self.state.w.astype(np.float64))
        return fused.astype(np.float32)

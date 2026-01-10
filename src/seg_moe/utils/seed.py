from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np


def seed_everything(seed: int, deterministic: bool = True, cudnn_benchmark: bool = False) -> None:
    """Seed python/numpy/torch for reproducibility.

    Notes
    -----
    - torch determinism can reduce performance.
    - Some CUDA ops are nondeterministic depending on platform/driver.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            # Some CUDA ops used in segmentation (e.g. certain loss kernels) may not have
            # a deterministic implementation on all platforms. Using warn_only keeps the
            # pipeline runnable while still surfacing nondeterministic ops.
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except TypeError:
                # Older PyTorch versions don't support warn_only.
                torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
    except Exception:
        # Torch might not be installed yet.
        pass


@dataclass(frozen=True)
class SeedConfig:
    seed: int
    torch_deterministic: bool = True
    cudnn_benchmark: bool = False

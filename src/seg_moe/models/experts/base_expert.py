"""
Base Expert interface.

All 3D heterogeneous experts MUST implement this ABC so that
the training / caching / fusion pipeline can treat them uniformly.
"""
from __future__ import annotations

import abc
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class BaseExpert(nn.Module, abc.ABC):
    """Abstract base class for a segmentation expert.

    Every expert MUST expose:
      - name          (str)
      - num_classes   (int)
      - predict_logits(x) -> [B, M, D, H, W]  (raw logits, no softmax)
      - load_checkpoint(path)
    """

    # ---- abstract properties / methods ----

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Short unique name, e.g. 'nnunet-v2', 'swin-unetr-base'."""
        ...

    @property
    @abc.abstractmethod
    def num_classes(self) -> int:
        ...

    @abc.abstractmethod
    def predict_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits [B, M, *spatial] — NO softmax.

        For 3D: x is [B, C, D, H, W], output is [B, M, D, H, W].
        """
        ...

    # ---- convenience (default implementations) ----

    def load_checkpoint(self, path: str | Path, strict: bool = True) -> None:
        """Load model weights (state_dict) from *path*.

        Handles common patterns:
          - Plain state_dict
          - Dict with 'model' key (engine.py convention)
          - DDP/DP 'module.' prefix stripping
        """
        state = torch.load(str(path), map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        # strip 'module.' prefix (DataParallel / DDP compat)
        state = {
            (k[len("module."):] if k.startswith("module.") else k): v
            for k, v in state.items()
        }
        self.load_state_dict(state, strict=strict)

    def save_checkpoint(
        self, path: str | Path, meta: Optional[Dict[str, Any]] = None
    ) -> None:
        """Save model state_dict + optional meta to *path*."""
        data: Dict[str, Any] = {"model": self.state_dict()}
        if meta:
            data.update(meta)
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, str(p))

    # ---- forward delegates to predict_logits ----

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Default: forward == predict_logits.

        Override if you need additional behaviour (e.g. deep supervision).
        """
        return self.predict_logits(x)

"""Checkpoint helpers for trusted local model files.

PyTorch 2.6 changed ``torch.load`` to default to ``weights_only=True``.
Some Seg-MoE checkpoints include NumPy scalar metadata, which can trigger
weights-only unpickling failures even though the checkpoints are valid.

These helpers keep the safer path first, then fall back to
``weights_only=False`` only for trusted checkpoints when that specific
PyTorch 2.6 compatibility error is encountered.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import torch


def normalize_state_dict_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Strip an optional ``module.`` prefix from state_dict keys."""
    return {k.removeprefix("module."): v for k, v in state_dict.items()}


def _is_weights_only_compat_error(exc: BaseException) -> bool:
    msg = str(exc)
    return (
        isinstance(exc, pickle.UnpicklingError)
        or "Weights only load failed" in msg
        or "Unsupported global" in msg
    )


def load_trusted_torch_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device | None = "cpu",
) -> Any:
    """Load a trusted checkpoint across PyTorch versions.

    Use this only for checkpoints you trust, since the fallback path may
    deserialize arbitrary Python objects when required for backward
    compatibility with older Seg-MoE checkpoints.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)
    except Exception as exc:
        if not _is_weights_only_compat_error(exc):
            raise
        return torch.load(path, map_location=map_location, weights_only=False)


def extract_model_state_dict(payload: Any) -> dict[str, Any]:
    """Return a normalized model state_dict from a checkpoint payload."""
    if isinstance(payload, dict) and "model" in payload and isinstance(payload["model"], dict):
        payload = payload["model"]
    if not isinstance(payload, dict):
        raise TypeError(f"Expected checkpoint payload to be a dict, got {type(payload)!r}")
    return normalize_state_dict_keys(payload)


def load_trusted_model_state_dict(
    path: str | Path,
    *,
    map_location: str | torch.device | None = "cpu",
) -> dict[str, Any]:
    """Load and normalize a trusted checkpoint's model state_dict."""
    return extract_model_state_dict(
        load_trusted_torch_checkpoint(path, map_location=map_location)
    )

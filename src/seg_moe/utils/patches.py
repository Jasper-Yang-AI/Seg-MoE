"""
2D Patch split / merge utilities for gating network.

Supports:
  - Non-overlapping patches (stride == patch_size)
  - Overlapping patches (stride < patch_size) with Gaussian blending
  - Edge-aligned patches for non-divisible image sizes

Used by:
  - GatingPatchDataset (training)
  - gating_inference.py (dynamic fusion inference)
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Patch positions
# ---------------------------------------------------------------------------

def compute_patch_positions(
    H: int,
    W: int,
    patch_size: int,
    stride: int,
) -> List[Tuple[int, int]]:
    """Compute top-left (y, x) positions for overlapping patches.

    Patches that would extend beyond the image border are shifted inward.
    Deduplicates positions for small images / large stride.
    """
    positions: list[tuple[int, int]] = []

    ys = list(range(0, H - patch_size + 1, stride))
    # ensure last row is covered
    if len(ys) == 0 or ys[-1] + patch_size < H:
        ys.append(max(0, H - patch_size))

    xs = list(range(0, W - patch_size + 1, stride))
    if len(xs) == 0 or xs[-1] + patch_size < W:
        xs.append(max(0, W - patch_size))

    for y in ys:
        for x in xs:
            positions.append((y, x))

    # deduplicate (preserving order)
    return list(dict.fromkeys(positions))


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------

def split_into_patches_2d(
    x: np.ndarray | torch.Tensor,
    patch_size: int,
    stride: int,
) -> Tuple[list, List[Tuple[int, int]]]:
    """Split ``[C, H, W]`` tensor / array into patches.

    Returns
    -------
    patches : list of ``[C, pH, pW]``
    positions : list of ``(y, x)`` top-left coordinates
    """
    if isinstance(x, torch.Tensor):
        _, H, W = x.shape
    else:
        _, H, W = x.shape

    positions = compute_patch_positions(H, W, patch_size, stride)
    patches = []
    for y, px in positions:
        patches.append(x[:, y : y + patch_size, px : px + patch_size])
    return patches, positions


# ---------------------------------------------------------------------------
# Gaussian kernel for blending
# ---------------------------------------------------------------------------

def _gaussian_kernel_2d(size: int, sigma: float | None = None) -> np.ndarray:
    """2-D Gaussian kernel, peak = 1, for overlap blending."""
    if sigma is None:
        sigma = size / 4.0
    ax = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return kernel / kernel.max()


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def merge_patches_2d(
    patches: list,
    positions: List[Tuple[int, int]],
    full_shape: Tuple[int, int],
    patch_size: int,
    blend_mode: str = "gaussian",
) -> np.ndarray:
    """Merge patches back into a full image with blending.

    Parameters
    ----------
    patches : list of ``[C, pH, pW]`` or ``[pH, pW]`` arrays / tensors
    positions : list of ``(y, x)`` (same order as patches)
    full_shape : ``(H, W)``
    patch_size : int
    blend_mode : ``"gaussian"`` or ``"average"``

    Returns
    -------
    merged : ``[C, H, W]`` or ``[H, W]`` float32 ndarray
    """
    H, W = full_shape

    # determine output shape
    sample = patches[0]
    if isinstance(sample, torch.Tensor):
        sample = sample.detach().cpu().numpy()
    has_channels = sample.ndim == 3

    if has_channels:
        C = sample.shape[0]
        out = np.zeros((C, H, W), dtype=np.float64)
    else:
        out = np.zeros((H, W), dtype=np.float64)
    weight_map = np.zeros((H, W), dtype=np.float64)

    kernel = (
        _gaussian_kernel_2d(patch_size)
        if blend_mode == "gaussian"
        else np.ones((patch_size, patch_size), dtype=np.float64)
    )

    for patch, (y, x) in zip(patches, positions):
        if isinstance(patch, torch.Tensor):
            patch = patch.detach().cpu().numpy()
        patch = patch.astype(np.float64)

        if has_channels:
            out[:, y : y + patch_size, x : x + patch_size] += patch * kernel[None]
        else:
            out[y : y + patch_size, x : x + patch_size] += patch * kernel

        weight_map[y : y + patch_size, x : x + patch_size] += kernel

    weight_map = np.maximum(weight_map, 1e-8)
    if has_channels:
        out /= weight_map[None]
    else:
        out /= weight_map

    return out.astype(np.float32)

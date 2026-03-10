from __future__ import annotations

from typing import Iterable, Tuple


def parse_3d_size(size: Iterable[int | float]) -> Tuple[int, int, int]:
    """Convert config order [H, W, D] to tensor order [D, H, W]."""
    values = [int(v) for v in size]
    if len(values) != 3:
        raise ValueError(f"3D size must have exactly 3 values, got {values!r}")
    h, w, d = values
    return (d, h, w)
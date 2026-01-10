from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Sample:
    image_path: str
    mask_path: str
    id: str
    dataset: str
    split: str
    spacing_yx: Optional[tuple[float, float]] = None  # for metrics in mm if available

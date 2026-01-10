from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from seg_moe.utils.io import load_jsonl


def load_split_index(path: str | Path) -> List[Dict[str, Any]]:
    return load_jsonl(path)


def filter_by_split(rows: Iterable[Dict[str, Any]], split: str) -> List[Dict[str, Any]]:
    return [r for r in rows if r.get("split") == split]


def list_splits(rows: Iterable[Dict[str, Any]]) -> List[str]:
    return sorted({r.get("split") for r in rows})


def infer_num_classes(dataset_cfg: Dict[str, Any]) -> int:
    return int(dataset_cfg["task"]["num_classes"])


def infer_image_size(dataset_cfg: Dict[str, Any]) -> tuple[int, int]:
    size = dataset_cfg["input"]["image_size"]
    return int(size[0]), int(size[1])


def infer_image_channels(dataset_cfg: Dict[str, Any]) -> int:
    # If using imagenet normalization, output is always 3 channels (grayscale gets replicated)
    norm_mode = dataset_cfg.get("input", {}).get("normalize", {}).get("mode", "imagenet")
    if norm_mode == "imagenet":
        return 3
    return int(dataset_cfg["input"].get("image_channels", 3))

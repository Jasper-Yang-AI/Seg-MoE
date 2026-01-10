#!/usr/bin/env python
"""
通用 MSD (Medical Segmentation Decathlon) 数据 prepare 脚本。

用法（以 Task03 为例）：
    python scripts/prepare_msd.py --config configs/2d/datasets/msd_task03_liver.yaml

或一键新建 Task 配置后 prepare（如 Task04）：
    1. 复制 configs/2d/datasets/msd_template.yaml -> msd_task04_xxx.yaml
    2. 修改 name, paths.*, raw_structure.* 等
    3. python scripts/prepare_msd.py --config configs/2d/datasets/msd_task04_xxx.yaml
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from seg_moe.utils.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Prepare MSD Task data: 3D NIfTI -> 2D slices + index_all.jsonl"
    )
    ap.add_argument("--config", required=True, help="Dataset YAML config (e.g., msd_task03_liver.yaml)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    print(f"[prepare_msd] dataset: {cfg['name']}")
    print(f"[prepare_msd] raw_dir: {cfg['paths']['raw_dir']}")

    # Delegate to generic NIfTI slicer
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "prepare_nifti_slices.py"),
        "--config",
        args.config,
    ]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse

from seg_moe.utils.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    _ = load_config(args.config)

    import subprocess
    import sys

    cmd = [
        sys.executable,
        "scripts/prepare_nifti_slices.py",
        "--config",
        args.config,
    ]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()

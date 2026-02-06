#!/usr/bin/env python
"""
Cache expert logits for all 3D experts.

For each case in the specified split, runs sliding-window inference
through each expert and saves float16 logits + meta to disk.

Usage:
    python scripts/inference/cache_expert_logits.py \
        --exp configs/3d/exp_msd03_liver.yaml \
        --split val_fold0 \
        --fold 0 \
        --which best

Output structure:
    runs/<exp>/cache/logits/<expert_name>/<split>/<case_id>.npz
      → logits: float16 [M, D, H, W]
      → meta: {case_id, shape, spacing, affine}
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from seg_moe.models.experts.factory import ExpertFactory
from seg_moe.utils.config import load_config
from seg_moe.utils.io import ensure_dir


def _sliding_window_inference(model, x, roi_size, sw_batch_size=4, overlap=0.5):
    try:
        from monai.inferers import sliding_window_inference
        return sliding_window_inference(x, roi_size, sw_batch_size, model, overlap=overlap)
    except ImportError:
        return model(x)


def cache_logits_for_expert(
    expert_name: str,
    expert: torch.nn.Module,
    cases: List[Dict[str, Any]],
    cache_dir: Path,
    device: torch.device,
    roi_size: tuple = (96, 96, 96),
    sw_batch_size: int = 4,
    skip_existing: bool = True,
) -> int:
    """Cache logits for one expert, all cases.

    Each case dict must have:  id, image (tensor or path), optionally spacing/affine.
    Returns number of cases cached.
    """
    expert.eval()
    expert.to(device)
    out_dir = ensure_dir(cache_dir / expert_name)
    cached = 0

    with torch.no_grad():
        for case in cases:
            case_id = str(case["id"])
            out_path = out_dir / f"{case_id}.npz"
            if skip_existing and out_path.exists():
                continue

            img = case["image"]  # expected [1, C, D, H, W]
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img).float()
            if img.ndim == 4:
                img = img.unsqueeze(0)
            img = img.to(device)

            logits = _sliding_window_inference(expert, img, roi_size, sw_batch_size)
            logits_np = logits.cpu().numpy()[0].astype(np.float16)  # [M, D, H, W]

            meta = {
                "case_id": case_id,
                "shape": list(logits_np.shape),
            }
            if "spacing" in case:
                meta["spacing"] = list(case["spacing"])
            if "affine" in case:
                meta["affine"] = case["affine"].tolist() if hasattr(case["affine"], "tolist") else case["affine"]

            np.savez_compressed(str(out_path), logits=logits_np, meta=meta)
            cached += 1

    return cached


def cache_logits_smoke(
    experts_cfg: Dict[str, Any],
    num_classes: int = 3,
    in_channels: int = 1,
    n_cases: int = 4,
    patch_size: tuple = (96, 96, 96),
    cache_dir: Path = Path("runs/smoke_cache"),
):
    """Smoke test: cache random logits for all experts."""
    factory = ExpertFactory(experts_cfg)
    experts = factory.build_all(in_channels=in_channels, classes=num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    D, H, W = patch_size
    cases = []
    for i in range(n_cases):
        cases.append({
            "id": f"case_{i:04d}",
            "image": torch.randn(1, in_channels, D, H, W),
        })

    for expert in experts:
        n = cache_logits_for_expert(
            expert_name=expert.name,
            expert=expert,
            cases=cases,
            cache_dir=cache_dir,
            device=device,
            roi_size=patch_size,
        )
        print(f"  [{expert.name}] cached {n}/{len(cases)} cases → {cache_dir / expert.name}")

    return cache_dir, [e.name for e in experts], cases


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Cache 3D expert logits")
    ap.add_argument("--exp", required=True)
    ap.add_argument("--split", default="val_fold0")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--which", default="best", choices=["best", "last"])
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="Smoke test with random data")
    args = ap.parse_args()

    exp_cfg = load_config(args.exp)
    experts_cfg = load_config(exp_cfg.get("experts_config", "configs/3d/experts.yaml"))

    num_classes = int(exp_cfg.get("dataset", {}).get("num_classes", 3))
    in_channels = int(exp_cfg.get("dataset", {}).get("in_channels", 1))

    exp_name = exp_cfg.get("exp_name", "segmoe_3d")
    cache_dir = Path(exp_cfg.get("output", {}).get("cache_dir",
                     f"runs/{exp_name}/cache").replace("${exp_name}", exp_name)) / "logits" / args.split

    if args.smoke:
        cache_logits_smoke(experts_cfg, num_classes, in_channels, cache_dir=cache_dir)
        print(f"\nSmoke cache done → {cache_dir}")
        return

    # TODO: plug in real 3D data loader here
    raise NotImplementedError("Real 3D data loading not yet implemented. Use --smoke for testing.")


if __name__ == "__main__":
    main()

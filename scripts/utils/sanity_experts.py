#!/usr/bin/env python
"""
Sanity check: build all 3 experts and verify output shapes match.

Usage:
    python scripts/utils/sanity_experts.py
    python scripts/utils/sanity_experts.py --experts-config configs/3d/experts.yaml
"""
from __future__ import annotations

import argparse
import sys

import torch

from seg_moe.models.experts.factory import ExpertFactory
from seg_moe.utils.config import load_config


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts-config", default="configs/3d/experts.yaml")
    ap.add_argument("--in-channels", type=int, default=1)
    ap.add_argument("--num-classes", type=int, default=3)
    ap.add_argument("--patch-size", type=int, nargs=3, default=[96, 96, 96],
                    help="Spatial size D H W. SwinUNETR needs >=64 per dim.")
    ap.add_argument("--batch-size", type=int, default=1)
    args = ap.parse_args()

    min_dim = min(args.patch_size)
    if min_dim < 64:
        print(f"WARNING: patch-size {args.patch_size} may be too small for "
              f"SwinUNETR (minimum ~64 per dim). Consider using 64+.")

    cfg = load_config(args.experts_config)
    factory = ExpertFactory(cfg)
    experts = factory.build_all(in_channels=args.in_channels, classes=args.num_classes)

    if not experts:
        print("ERROR: no experts built")
        return 1

    B = args.batch_size
    C = args.in_channels
    D, H, W = args.patch_size
    x = torch.randn(B, C, D, H, W)

    device = torch.device("cpu")  # keep on CPU for sanity check
    shapes = []
    dtypes = []
    passed = 0

    print(f"\nInput: x.shape = {list(x.shape)}")
    print(f"Expected output: [B={B}, M={args.num_classes}, D={D}, H={H}, W={W}]\n")
    print(f"{'Expert':<25s} {'Shape':<30s} {'dtype':<12s} {'Params':>12s}  Status")
    print("-" * 90)

    for exp in experts:
        exp.to(device)
        exp.eval()
        with torch.no_grad():
            out = exp.predict_logits(x.to(device))

        n_params = sum(p.numel() for p in exp.parameters())
        shape = list(out.shape)
        dt = str(out.dtype)
        expected = [B, args.num_classes, D, H, W]

        ok = (shape == expected)
        status = "OK" if ok else f"FAIL (expected {expected})"
        print(f"{exp.name:<25s} {str(shape):<30s} {dt:<12s} {n_params:>12,d}  {status}")

        shapes.append(shape)
        dtypes.append(dt)
        if ok:
            passed += 1

    print("-" * 90)
    all_same = len(set(str(s) for s in shapes)) == 1 and len(set(dtypes)) == 1
    print(f"\nAll shapes identical: {all_same}")
    print(f"Passed: {passed}/{len(experts)}")

    if passed == len(experts) and all_same:
        print("\n✓ All experts produce consistent [B, M, D, H, W] logits.")
        return 0
    else:
        print("\n✗ Some experts failed shape check.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

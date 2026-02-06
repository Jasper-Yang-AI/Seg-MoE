"""
Sanity check: build all 2D experts and verify output shapes match.

Usage:
    python scripts/utils/sanity_experts_2d.py
    python scripts/utils/sanity_experts_2d.py --models configs/2d/models.yaml
"""
from __future__ import annotations

import argparse
import sys

import torch

from seg_moe.models.factory_2d import build_expert, expert_name, list_experts
from seg_moe.utils.config import load_config


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="configs/2d/models.yaml",
                    help="Path to experts_v2 YAML config")
    ap.add_argument("--in-channels", type=int, default=3)
    ap.add_argument("--num-classes", type=int, default=3)
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=2)
    args = ap.parse_args()

    models_cfg = load_config(args.models)
    expert_cfgs = list_experts(models_cfg)

    if not expert_cfgs:
        print("ERROR: no experts defined in config")
        return 1

    B, C, H, W = args.batch_size, args.in_channels, args.img_size, args.img_size
    x = torch.randn(B, C, H, W)
    expected = [B, args.num_classes, H, W]
    device = torch.device("cpu")

    print(f"\nInput:    x.shape = {list(x.shape)}")
    print(f"Expected: {expected}\n")
    print(f"{'Expert':<25s} {'Shape':<25s} {'Params':>12s}  Status")
    print("-" * 70)

    passed = 0
    shapes = []
    for ec in expert_cfgs:
        name = expert_name(ec)
        try:
            model = build_expert(ec, in_channels=C, num_classes=args.num_classes)
            model.to(device).eval()
            with torch.no_grad():
                out = model(x.to(device))
            shape = list(out.shape)
            n_params = sum(p.numel() for p in model.parameters())
            ok = shape == expected
            status = "OK" if ok else f"FAIL (got {shape})"
            print(f"{name:<25s} {str(shape):<25s} {n_params:>12,d}  {status}")
            shapes.append(shape)
            if ok:
                passed += 1
        except Exception as e:
            print(f"{name:<25s} {'ERROR':<25s} {'':>12s}  {e}")
            shapes.append(None)

    print("-" * 70)
    valid = [s for s in shapes if s is not None]
    all_same = len(set(str(s) for s in valid)) == 1 if valid else False
    print(f"\nAll shapes identical: {all_same}")
    print(f"Passed: {passed}/{len(expert_cfgs)}")

    if passed == len(expert_cfgs) and all_same:
        print("\n✓ All 2D experts produce consistent [B, M, H, W] logits.")
        return 0
    else:
        print("\n✗ Some experts failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

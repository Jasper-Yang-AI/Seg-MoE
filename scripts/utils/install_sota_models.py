#!/usr/bin/env python
"""Quick installation script for SOTA expert dependencies."""
import subprocess
import sys


def run_command(cmd, description):
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}")
    try:
        subprocess.run(cmd, shell=True, check=True)
        print(f"  OK: {description}")
        return True
    except subprocess.CalledProcessError:
        print(f"  FAIL: {description}")
        return False


def main():
    print("""
  Seg-MoE SOTA Models Installation
  nnUNet v2 + Swin-UNetR + SegResNet
""")
    results = {}
    results["monai"] = run_command("pip install monai>=1.3.0", "Installing MONAI (Swin-UNetR + SegResNet)")
    results["nnunet"] = run_command("pip install nnunetv2>=2.2 dynamic-network-architectures>=0.3",
                                    "Installing nnUNet v2")
    results["support"] = run_command("pip install timm>=0.9.0 einops>=0.7.0", "Installing support libraries")

    print(f"\n{'='*60}")
    print("Installation Summary")
    print(f"{'='*60}")
    for comp, ok in results.items():
        print(f"  {comp:20s}: {'OK' if ok else 'FAIL'}")

    # Verify SegResNet
    try:
        from monai.networks.nets import SegResNet
        m = SegResNet(spatial_dims=3, in_channels=1, out_channels=3)
        n = sum(p.numel() for p in m.parameters())
        print(f"\n  SegResNet verification: {n:,} params — OK")
    except Exception as e:
        print(f"\n  SegResNet verification FAILED: {e}")

    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())

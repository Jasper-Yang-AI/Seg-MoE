#!/usr/bin/env python
"""
Test script to verify SOTA model installations.
"""
import sys
from pathlib import Path

import torch


def test_swin_unetr():
    """Test Swin-UNetR availability."""
    try:
        from monai.networks.nets import SwinUNETR
        
        model = SwinUNETR(
            img_size=(256, 256),
            in_channels=1,
            out_channels=4,
            spatial_dims=2,
            feature_size=48,
        )
        
        # Test forward pass
        x = torch.rand(1, 1, 256, 256)
        with torch.no_grad():
            y = model(x)
        
        params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"✓ Swin-UNetR: Available ({params:.1f}M params)")
        return True
        
    except ImportError as e:
        print(f"✗ Swin-UNetR: Not available - {e}")
        print("  Install with: pip install monai>=1.3.0")
        return False
    except Exception as e:
        print(f"⚠ Swin-UNetR: Import OK but error: {e}")
        return False


def test_nnunet():
    """Test nnUNet availability."""
    try:
        from seg_moe.models.wrappers.nnunet_wrapper import NnUNetWrapper
        
        model = NnUNetWrapper(
            in_channels=1,
            num_classes=4,
            patch_size=(256, 256),
            n_stages=6,
        )
        
        # Test forward pass
        x = torch.rand(1, 1, 256, 256)
        with torch.no_grad():
            y = model(x)
        
        params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"✓ nnUNet v2: Available ({params:.1f}M params)")
        return True
        
    except ImportError as e:
        print(f"✗ nnUNet v2: Not available - {e}")
        print("  Install with: pip install nnunetv2>=2.2")
        return False
    except Exception as e:
        print(f"⚠ nnUNet v2: Import OK but error: {e}")
        return False


def test_vm_unet():
    """Test VM-UNet availability."""
    try:
        from seg_moe.models.wrappers.vm_unet_wrapper import VMUNetWrapper, check_mamba_available
        
        if not check_mamba_available():
            print("⚠ VM-UNet: Mamba not installed (using fallback)")
            print("  Install with: pip install causal-conv1d mamba-ssm")
            return False
        
        model = VMUNetWrapper(
            in_channels=1,
            num_classes=4,
            img_size=256,
        )
        
        # Test forward pass
        x = torch.rand(1, 1, 256, 256)
        with torch.no_grad():
            y = model(x)
        
        params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"✓ VM-UNet: Available ({params:.1f}M params)")
        return True
        
    except ImportError as e:
        print(f"✗ VM-UNet: Not available - {e}")
        print("  See docs/INSTALL_SOTA_MODELS.md for setup instructions")
        return False
    except Exception as e:
        print(f"⚠ VM-UNet: Import OK but error: {e}")
        return False


def test_factory():
    """Test factory can build models."""
    try:
        from seg_moe.models.factory_sota import build_sota_model
        
        print("\nTesting factory_sota.py:")
        
        # Test Swin-UNetR via factory
        try:
            model = build_sota_model("swin_unetr", in_channels=1, classes=4)
            print("  ✓ Factory can build Swin-UNetR")
        except Exception as e:
            print(f"  ✗ Factory Swin-UNetR failed: {e}")
        
        # Test nnUNet via factory
        try:
            model = build_sota_model("nnunet", in_channels=1, classes=4)
            print("  ✓ Factory can build nnUNet")
        except Exception as e:
            print(f"  ✗ Factory nnUNet failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Factory test failed: {e}")
        return False


def main():
    print("=" * 50)
    print("Testing SOTA Models Installation")
    print("=" * 50)
    print()
    
    results = []
    
    results.append(("Swin-UNetR", test_swin_unetr()))
    results.append(("nnUNet v2", test_nnunet()))
    results.append(("VM-UNet", test_vm_unet()))
    
    print()
    test_factory()
    
    print()
    print("=" * 50)
    print("Summary")
    print("=" * 50)
    
    available = sum(1 for _, ok in results if ok)
    total = len(results)
    
    print(f"Models ready: {available}/{total}")
    
    # Note: Fallback models are acceptable for testing
    print()
    print("Note: Models using fallback implementations are still functional.")
    print("Install optional dependencies for full features:")
    print("  - MONAI: pip install monai>=1.3.0")
    print("  - nnUNet: pip install nnunetv2>=2.2")
    print("  - VM-UNet (Mamba): pip install causal-conv1d mamba-ssm")
    
    if available == total:
        print("\n🎉 All SOTA models are available!")
        return 0
    elif available >= 1:
        print("\n✅ Core models available, ready for training!")
        print("   You can start experimenting with available models.")
        return 0
    else:
        print("\n❌ Please install missing models")
        print("   See: docs/INSTALL_SOTA_MODELS.md")
        return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""
Test SegMamba integration and dependencies.
"""
import sys
import torch


def test_mamba_dependencies():
    """Test if Mamba dependencies are available."""
    print("1. Testing Mamba dependencies...")
    try:
        import mamba_ssm
        print("   ✓ mamba-ssm imported successfully")
        has_mamba = True
    except ImportError as e:
        print(f"   ✗ mamba-ssm not available: {e}")
        has_mamba = False
    
    try:
        import causal_conv1d
        print("   ✓ causal-conv1d imported successfully")
        has_causal = True
    except ImportError as e:
        print(f"   ✗ causal-conv1d not available: {e}")
        has_causal = False
    
    return has_mamba and has_causal


def test_segmamba_wrapper():
    """Test SegMamba wrapper."""
    print("\n2. Testing SegMamba wrapper...")
    try:
        from seg_moe.models.wrappers.segmamba_wrapper import SegMambaWrapper, check_segmamba_available
        print("   ✓ SegMambaWrapper imported successfully")
        
        if check_segmamba_available():
            print("   ✓ Mamba dependencies available (will use SegMamba)")
        else:
            print("   ⚠ Mamba dependencies not available (will use fallback)")
        
        return True
    except ImportError as e:
        print(f"   ✗ Failed to import SegMambaWrapper: {e}")
        return False


def test_model_creation():
    """Test creating SegMamba model."""
    print("\n3. Testing model creation...")
    try:
        from seg_moe.models.wrappers.segmamba_wrapper import SegMambaWrapper
        
        model = SegMambaWrapper(
            in_channels=1,
            num_classes=4,
            img_size=256,
            embed_dim=96,
            depths=(2, 2, 9, 2),
        )
        
        # Test forward pass
        x = torch.rand(1, 1, 256, 256)
        with torch.no_grad():
            y = model(x)
        
        assert y.shape == (1, 4, 256, 256), f"Unexpected output shape: {y.shape}"
        
        params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"   ✓ Model created successfully ({params:.1f}M params)")
        
        if model.using_fallback:
            print("   ⚠ Using fallback model (EfficientNet-B0 UNet)")
        else:
            print("   ✓ Using full SegMamba with Mamba")
        
        return True
    except Exception as e:
        print(f"   ✗ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_factory_integration():
    """Test factory can build SegMamba."""
    print("\n4. Testing factory integration...")
    try:
        from seg_moe.models.factory_sota import build_sota_model
        
        model = build_sota_model(
            arch="segmamba",
            in_channels=1,
            classes=4,
            config={
                "img_size": 256,
                "embed_dim": 96,
                "depths": [2, 2, 9, 2],
            }
        )
        
        # Test forward pass
        x = torch.rand(1, 1, 256, 256)
        with torch.no_grad():
            y = model(x)
        
        assert y.shape == (1, 4, 256, 256), f"Unexpected output shape: {y.shape}"
        print("   ✓ Factory can build SegMamba")
        return True
    except Exception as e:
        print(f"   ✗ Factory integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_load():
    """Test loading SegMamba from config."""
    print("\n5. Testing config loading...")
    try:
        import yaml
        from pathlib import Path
        
        config_path = Path("configs/2d/models_sota.yaml")
        if not config_path.exists():
            print("   ⚠ Config file not found, skipping test")
            return True
        
        with open(config_path, encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Check if SegMamba is in config
        experts = config.get("sota_experts", [])
        segmamba_config = None
        for expert in experts:
            if expert.get("architecture") == "segmamba":
                segmamba_config = expert
                break
        
        if segmamba_config:
            print(f"   ✓ SegMamba found in config: {segmamba_config['name']}")
            print(f"   ✓ Enabled: {segmamba_config.get('enabled', False)}")
            return True
        else:
            print("   ✗ SegMamba not found in config")
            return False
    except Exception as e:
        print(f"   ✗ Config loading failed: {e}")
        return False


def main():
    print("=" * 60)
    print("Testing SegMamba Integration")
    print("=" * 60)
    print()
    
    results = {}
    results["dependencies"] = test_mamba_dependencies()
    results["wrapper"] = test_segmamba_wrapper()
    results["model"] = test_model_creation()
    results["factory"] = test_factory_integration()
    results["config"] = test_config_load()
    
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:20s}: {status}")
    
    print(f"\nTests passed: {passed}/{total}")
    
    if not results["dependencies"]:
        print("\n⚠ Note: Mamba dependencies not installed")
        print("SegMamba will use fallback model (EfficientNet-B0 UNet)")
        print("\nTo install Mamba:")
        print("  pip install causal-conv1d>=1.1.0")
        print("  pip install mamba-ssm>=1.0.0")
        print("\nFor full SegMamba architecture:")
        print("  https://github.com/ge-xing/SegMamba")
    
    if results["wrapper"] and results["model"] and results["factory"]:
        print("\n✅ SegMamba integration successful!")
        print("   You can start training with:")
        print("   python scripts/train_2d_experts.py --models configs/2d/models_sota.yaml")
        return 0
    else:
        print("\n❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

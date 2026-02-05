#!/usr/bin/env python
"""
Quick installation script for SOTA models.
"""
import subprocess
import sys


def run_command(cmd, description):
    """Run a command and report status."""
    print(f"\n{'='*60}")
    print(f"📦 {description}")
    print(f"{'='*60}")
    print(f"Running: {cmd}")
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=False,
            text=True,
        )
        print(f"✅ {description} - SUCCESS")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - FAILED")
        print(f"Error: {e}")
        return False


def main():
    print("""
╔══════════════════════════════════════════════════════════╗
║     Seg-MoE SOTA Models Installation                     ║
║     nnUNet v2 + Swin-UNetR + VM-UNet                     ║
╚══════════════════════════════════════════════════════════╝
""")
    
    results = {}
    
    # Step 1: Install MONAI (for Swin-UNetR)
    results["monai"] = run_command(
        "pip install monai>=1.3.0",
        "Installing MONAI (Swin-UNetR)"
    )
    
    # Step 2: Install nnUNet
    results["nnunet"] = run_command(
        "pip install nnunetv2>=2.2",
        "Installing nnUNet v2"
    )
    
    # Step 3: Install supporting libraries
    results["support"] = run_command(
        "pip install timm>=0.9.0 einops>=0.7.0",
        "Installing support libraries"
    )
    
    # Step 4: Try Mamba (optional)
    print(f"\n{'='*60}")
    print("Installing Mamba (optional - requires CUDA)")
    print(f"{'='*60}")
    print("This may fail if CUDA is not properly configured.")
    print("VM-UNet will use a fallback model if Mamba is unavailable.")
    
    user_input = input("\nAttempt Mamba installation? [y/N]: ").lower()
    if user_input == 'y':
        results["mamba_causal"] = run_command(
            "pip install causal-conv1d>=1.1.0",
            "Installing causal-conv1d"
        )
        results["mamba"] = run_command(
            "pip install mamba-ssm>=1.0.0",
            "Installing mamba-ssm"
        )
    else:
        print("⏭Skipping Mamba installation")
        results["mamba"] = False
    
    # Summary
    print(f"\n{'='*60}")
    print("Installation Summary")
    print(f"{'='*60}")
    
    for component, success in results.items():
        status = "OK" if success else "❌ Failed"
        print(f"{component:20s}: {status}")
    
    # Test installation
    print(f"\n{'='*60}")
    print("Testing installation...")
    print(f"{'='*60}")
    
    test_result = subprocess.run(
        [sys.executable, "scripts/test_sota_models.py"],
        capture_output=False,
    )
    
    if test_result.returncode == 0:
        print("Installation complete and verified!")
        print("\nNext steps:")
        print("1. Review configs/2d/models_sota.yaml")
        print("2. Enable models you want to use")
        print("3. Run: python scripts/train_2d_experts.py --models configs/2d/models_sota.yaml")
    else:
        print("Installation completed with some issues")
        print("See docs/INSTALL_SOTA_MODELS.md for troubleshooting")
    
    return test_result.returncode


if __name__ == "__main__":
    sys.exit(main())

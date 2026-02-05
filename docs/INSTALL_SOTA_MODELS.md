# Installing SOTA Models for Seg-MoE

This guide walks through installing the three heterogeneous expert models.

## 🎯 Overview

- **Swin-UNetR** (Transformer): Easy - via MONAI ✅
- **nnUNet v2** (CNN): Medium - via pip + wrapper ⚠️
- **SegMamba** (Mamba): Medium - automatic fallback 🔧

---

## 1. Swin-UNetR (Recommended to start)

### Installation

```bash
pip install monai>=1.3.0
# Or with all extras:
pip install "monai[all]"
```

### Verification

```python
from monai.networks.nets import SwinUNETR
model = SwinUNETR(
    img_size=(256, 256),
    in_channels=1,
    out_channels=4,
    spatial_dims=2,
)
print(f"✓ Swin-UNetR loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
```

### Pretrained Weights (Optional)

MONAI provides weights pretrained on large medical datasets:

```bash
# Download from MONAI Model Zoo
# https://github.com/Project-MONAI/MONAI-extra-test-data/releases
wget https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/swin_unetr.pt
```

---

## 2. nnUNet v2

### Installation

```bash
pip install nnunetv2>=2.2
```

**Dependencies**: nnUNet will auto-install `dynamic-network-architectures`.

### Verification

```python
from seg_moe.models.wrappers.nnunet_wrapper import NnUNetWrapper
model = NnUNetWrapper(
    in_channels=1,
    num_classes=4,
    patch_size=(256, 256),
)
print(f"✓ nnUNet loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
```

### Note on nnUNet

- We extract **only the architecture**, not the full training pipeline
- nnUNet has its own data preprocessing; we bypass it for consistency
- For full nnUNet workflow, see: https://github.com/MIC-DKFZ/nnUNet

---

## 3. SegMamba (Mamba-based)

### Overview

SegMamba provides efficient segmentation with linear complexity using state space models.
**Features**: Automatic fallback to EfficientNet-B0 UNet if Mamba dependencies unavailable.

### Prerequisites

⚠️ **Mamba requires CUDA** and may need compilation!

### Installation Steps

#### Step 1: Install Mamba dependencies

```bash
# Install causal-conv1d (required for Mamba)
pip install causal-conv1d>=1.1.0

# Install mamba-ssm
pip install mamba-ssm>=1.0.0
```

**Troubleshooting**: If installation fails, you may need:
- CUDA Toolkit matching your PyTorch version
- Proper C++ compiler (MSVC on Windows, GCC on Linux)

#### Step 2: Verify installation

```python
from seg_moe.models.wrappers.segmamba_wrapper import SegMambaWrapper, check_segmamba_available

if check_segmamba_available():
    print("✓ Mamba dependencies available")
    model = SegMambaWrapper(
        in_channels=1,
        num_classes=4,
        img_size=256,
    )
    print(f"✓ SegMamba loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
else:
    print("⚠ Mamba not available - SegMamba will use fallback (EfficientNet-B0)")
    model = SegMamba Wrapper(
        in_channels=1,
        num_classes=4,
        img_size=256,
    )
    print(f"✓ Fallback model loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
```

#### Optional: Full SegMamba Architecture

For the complete SegMamba implementation:

```bash
# Clone SegMamba repository
git clone https://github.com/ge-xing/SegMamba.git

# Install (if available as package)
pip install segmamba
```

**Note**: Our wrapper works with or without the full SegMamba implementation:
- **With Mamba**: Uses state space model backbone
- **Without Mamba**: Uses EfficientNet-B0 UNet as lightweight fallback

---

## Quick Installation (All models)

### For active development:

```bash
# Install base requirements
pip install -r requirements.txt

# Install SOTA extras
pip install -e ".[sota]"

# Enable SegMamba (optional, if CUDA available)
pip install causal-conv1d>=1.1.0 mamba-ssm>=1.0.0
```

### Minimal install (Swin-UNetR only):

```bash
pip install monai>=1.3.0
```

---

## Testing Installation

Run the test script:

```bash
python scripts/test_sota_models.py
```

Expected output:
```
Testing SOTA Models Installation
=================================
✓ Swin-UNetR: Available (45.2M params)
✓ nnUNet v2: Available (31.2M params)
⚠ SegMamba: Mamba not installed (using fallback 6.3M params)

Summary: 3/3 models ready (1 using fallback)
```

---

## Pretrained Weights

### Swin-UNetR
- **MONAI Model Zoo**: https://github.com/Project-MONAI/MONAI/wiki/Model-Zoo
- Weights trained on CT/MRI datasets

### nnUNet
- Can use models trained via official nnUNet pipeline
- Convert checkpoints: see `scripts/convert_nnunet_weights.py` (TODO)

### SegMamba
- **GitHub**: https://github.com/ge-xing/SegMamba
- Research paper on efficient medical image segmentation
- Automatic fallback ensures functionality without full installation

---

## Troubleshooting

### Issue: Mamba installation fails

**Solution**: Check CUDA compatibility
```bash
python -c "import torch; print(torch.version.cuda)"
```
Ensure `causal-conv1d` and `mamba-ssm` support your CUDA version.

### Issue: nnUNet imports fail

**Solution**: Verify installation
```bash
pip list | grep nnunet
nnunetv2 should be >= 2.2
```

### Issue: MONAI takes too long to install

**Solution**: Install minimal version
```bash
pip install monai  # Without [all] extras
```

---

## Next Steps

After installation:
1. Update `configs/2d/models_sota.yaml` - enable models
2. Test with: `python scripts/train_2d_experts.py --exp configs/2d/exp/exp_acdc.yaml --models configs/2d/models_sota.yaml`
3. Compare SOTA vs. original 9 experts

---

## References

- Swin-UNetR: https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR
- nnUNet v2: https://github.com/MIC-DKFZ/nnUNet
- SegMamba: https://github.com/ge-xing/SegMamba
- Mamba: https://github.com/state-spaces/mamba

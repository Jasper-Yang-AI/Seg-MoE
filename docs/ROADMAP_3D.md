# ROADMAP: 3D Patch/Region-Level Dynamic Gating (Planned)

This project currently reproduces the **2D** setting in:
- Dang et al. (Springer 2024) *Two-layer Ensemble of Deep Learning Models for Medical Image Segmentation*

The 3D extension is **not implemented yet** (by design), but the codebase reserves stable interfaces so you can add:

## Goals

1. Support 3D datasets (NIfTI/MetaImage volumes) with consistent preprocessing
2. Train 3D experts (e.g., 3D UNet variants)
3. Implement **patch/region-level dynamic gating**:
   - Compute patch descriptors from 3D feature maps
   - Predict gating weights per patch (and optionally per class)
   - Fuse expert logits/probabilities in a spatially varying manner

## Planned Components

### 1) 3D data pipeline

- Add 3D dataset entries under `configs/3d/`
- Implement `seg_moe.data.dataset_3d.*` similar to the 2D dataset
- Add preprocessing scripts to export cached 3D patches or on-the-fly patch sampling

### 2) 3D experts

- New factory: `seg_moe.models.factory_3d.build_model_3d(cfg)`
- Maintain a consistent interface:
  - `forward(x) -> logits` with shape `[B, C, D, H, W]`

### 3) Patch gating module

- File reserved: `src/seg_moe/gating/patch_gating_3d.py`
- Stable interface:
  - `PatchGating3D.forward(feature_or_probs) -> weights`
  - `weights` shape supports:
    - per-expert weights: `[B, K, D', H', W']`
    - optional per-class weights: `[B, K, M, D', H', W']`

### 4) Evaluation & submission

- Extend surface-distance metrics to 3D (distance transforms in 3D)
- Add dataset-specific submission exporters if required

## Notes

- The 2D reproducibility rules (seed, deterministic flags, config-driven pipelines) should carry over to 3D.
- Any 3D implementation must keep cache formats and run directory structure consistent with 2D.

#!/usr/bin/env python
"""
Prepare multi-modal prostate MRI → 2D RGB PNG slices.

科研级多模态 MRI → RGB 映射策略 (References):
  - Isensee et al. 2021, "nnU-Net" (Nature Methods)
    → per-channel z-score normalization for MRI
  - Litjens et al. 2014, "Evaluation of prostate segmentation algorithms"
    → Multi-parametric MRI (T2w, ADC, DWI) as multi-channel input
  - Reinke et al. 2024, "Understanding metric-related pitfalls" (Nature Methods)
    → Per-modality normalization is essential when value ranges differ

策略:
  每个模态 (0000/0001/0002) 独立做 per-volume percentile clipping [p0.5, p99.5]
  → 归一化到 [0, 255] uint8 → stack 为 RGB (H,W,3) PNG
  这样保证各模态贡献均等, 且兼容 ImageNet pretrained weights.

数据结构:
    E:\\nnunetv2_WebUI\\nnUNet_raw\\Dataset002_ProstateCrop_seg/
    imagesTr/  case_0000.nii.gz, case_0001.nii.gz, case_0002.nii.gz ...
    labelsTr/  case.nii.gz ...
    imagesTs/  ...
    labelsTs/  ...

Usage:
    python scripts/data/prepare_prostate.py \\
        --config configs/2d/datasets/prostate_local.yaml
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

from seg_moe.utils.config import load_config
from seg_moe.utils.io import ensure_dir, save_jsonl


# ── NIfTI reader ──────────────────────────────────────────────────────

def _read_nii(path: Path) -> Tuple[np.ndarray, Tuple[float, float]]:
    import SimpleITK as sitk

    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)  # [Z, Y, X]
    spacing = img.GetSpacing()  # (x, y, z)
    sx, sy = float(spacing[0]), float(spacing[1])
    if arr.ndim == 2:
        arr = arr[None]
    return arr, (sy, sx)  # spacing_yx


# ── Per-modality percentile normalization ─────────────────────────────

def _percentile_normalize_uint8(
    vol: np.ndarray,
    plow: float = 0.5,
    phigh: float = 99.5,
) -> np.ndarray:
    """Per-volume percentile clipping → [0, 255] uint8.

    Robust against outliers (e.g., bright artifact voxels in T2w/DWI).
    Uses foreground-only percentiles when most of the volume is zero-background.
    """
    v = vol.astype(np.float32)

    # Compute percentiles on foreground (non-zero) voxels if available
    fg = v[v > 0]
    if fg.size > 100:
        lo = float(np.percentile(fg, plow))
        hi = float(np.percentile(fg, phigh))
    else:
        lo = float(np.percentile(v, plow))
        hi = float(np.percentile(v, phigh))

    if hi - lo < 1e-6:
        hi = lo + 1.0

    v = np.clip(v, lo, hi)
    v = (v - lo) / (hi - lo)
    return (v * 255.0).clip(0, 255).astype(np.uint8)


# ── Resize ────────────────────────────────────────────────────────────

def _resize2d(arr: np.ndarray, size_hw: Tuple[int, int], is_mask: bool) -> np.ndarray:
    H, W = size_hw
    if arr.shape[0] == H and arr.shape[1] == W:
        return arr
    pil = Image.fromarray(arr)
    pil = pil.resize((W, H), resample=Image.NEAREST if is_mask else Image.BILINEAR)
    return np.array(pil)


# ── Case discovery ────────────────────────────────────────────────────

def _discover_cases(
    img_dir: Path,
    lab_dir: Path,
    n_modalities: int = 3,
) -> List[Dict]:
    """Discover multi-modal cases from nnU-Net style naming.

    Images: {case_id}_{MMMM}.nii.gz   (MMMM = 0000, 0001, 0002)
    Labels: {case_id}.nii.gz
    """
    # Find all label files → case IDs
    lab_files = sorted(list(lab_dir.rglob("*.nii.gz")) + list(lab_dir.rglob("*.nii")))
    lab_files = [p for p in lab_files if not p.name.startswith("._")]

    cases = []
    for lf in lab_files:
        # Strip extensions to get case_id
        stem = lf.name
        for ext in [".nii.gz", ".nii"]:
            if stem.endswith(ext):
                stem = stem[: -len(ext)]
                break
        case_id = stem

        # Find modality files
        mod_paths: List[Optional[Path]] = []
        all_found = True
        for m in range(n_modalities):
            mod_name = f"{case_id}_{m:04d}"
            cand_gz = img_dir / f"{mod_name}.nii.gz"
            cand_nii = img_dir / f"{mod_name}.nii"
            if cand_gz.exists():
                mod_paths.append(cand_gz)
            elif cand_nii.exists():
                mod_paths.append(cand_nii)
            else:
                all_found = False
                break
        if not all_found or len(mod_paths) != n_modalities:
            continue

        cases.append({
            "case_id": case_id,
            "label_path": lf,
            "modality_paths": mod_paths,
        })
    return cases


# ── Patient ID extraction ─────────────────────────────────────────────

def _extract_patient_id(case_id: str) -> str:
    """Extract patient-level ID for group-stratified splits.

    e.g. 'njmu_0000666593_N1973' → 'njmu_0000666593_N1973' (whole case is patient)
    For multi-timepoint data, override this function.
    """
    return case_id


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Prepare multi-modal prostate MRI → 2D RGB PNG slices"
    )
    ap.add_argument("--config", required=True, help="Dataset YAML config")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    name = cfg["name"]
    raw_dir = Path(cfg["paths"]["raw_dir"])
    proc_dir = Path(cfg["paths"]["processed_dir"])
    splits_dir = Path(cfg["paths"]["splits_dir"])

    ensure_dir(proc_dir / "images")
    ensure_dir(proc_dir / "masks")
    ensure_dir(splits_dir)

    H, W = int(cfg["input"]["image_size"][0]), int(cfg["input"]["image_size"][1])
    num_classes = int(cfg["task"]["num_classes"])
    n_modalities = int(cfg["input"].get("n_modalities", 3))

    ncfg = cfg.get("nifti", {}) or {}
    slice_policy = str(ncfg.get("slice_policy", "non_empty"))
    min_fg = int(ncfg.get("min_fg_pixels", 10))

    pclip = ncfg.get("intensity", {}).get("percentile_clip", [0.5, 99.5])
    p_low, p_high = float(pclip[0]), float(pclip[1])

    label_map = {int(a): int(b) for a, b in (cfg["task"].get("label_map") or {}).items()}

    # ── Discover train + test sets ──
    rs = cfg.get("raw_structure", {}) or {}

    split_dirs = []
    if rs.get("images_tr_dir") and rs.get("labels_tr_dir"):
        split_dirs.append(("raw_train", raw_dir / rs["images_tr_dir"], raw_dir / rs["labels_tr_dir"]))
    if rs.get("images_ts_dir") and rs.get("labels_ts_dir"):
        split_dirs.append(("raw_test", raw_dir / rs["images_ts_dir"], raw_dir / rs["labels_ts_dir"]))

    if not split_dirs:
        raise ValueError("No image/label dirs found in raw_structure config")

    all_rows = []
    stats_slices = 0
    stats_skipped = 0

    for raw_split, img_dir, lab_dir in split_dirs:
        if not img_dir.exists():
            print(f"[WARN] Missing {img_dir}, skip")
            continue
        if not lab_dir.exists():
            print(f"[WARN] Missing {lab_dir}, skip")
            continue

        cases = _discover_cases(img_dir, lab_dir, n_modalities)
        print(f"[{name}] {raw_split}: found {len(cases)} cases in {img_dir}")

        for case in tqdm(cases, desc=f"processing {raw_split}"):
            case_id = case["case_id"]
            patient_id = _extract_patient_id(case_id)

            # Read label volume
            lab_zyx, _ = _read_nii(case["label_path"])
            lab = lab_zyx.astype(np.int64)
            if label_map:
                mapped = lab.copy()
                for a, b in label_map.items():
                    mapped[lab == a] = b
                lab = mapped

            # Read and normalize each modality independently
            mods_u8 = []
            spacings = []
            for mp in case["modality_paths"]:
                vol, spacing_yx = _read_nii(mp)
                mods_u8.append(_percentile_normalize_uint8(vol, p_low, p_high))
                spacings.append(spacing_yx)
            spacing_yx = spacings[0]  # use first modality's spacing

            # Verify all modalities have same shape
            shapes = [m.shape for m in mods_u8]
            if len(set(shapes)) > 1:
                print(f"[WARN] Shape mismatch for {case_id}: {shapes}, skip")
                continue

            n_slices = mods_u8[0].shape[0]

            for z in range(n_slices):
                mask_z = lab[z]

                if slice_policy == "non_empty":
                    fg = int(np.sum(mask_z > 0))
                    if fg < min_fg:
                        stats_skipped += 1
                        continue

                # Stack 3 modalities as RGB [H, W, 3]
                rgb = np.stack([m[z] for m in mods_u8], axis=-1)  # H, W, 3

                # Resize
                rgb_r = np.stack([
                    _resize2d(rgb[:, :, c], (H, W), is_mask=False)
                    for c in range(n_modalities)
                ], axis=-1)
                mask_r = _resize2d(mask_z.astype(np.uint8), (H, W), is_mask=True)

                # Validate
                if np.any(mask_r >= num_classes):
                    raise ValueError(
                        f"Mask out of range for {case_id} z={z}: "
                        f"max={mask_r.max()}, num_classes={num_classes}"
                    )

                sid = f"{case_id}_z{z:03d}"
                img_out = proc_dir / "images" / f"{sid}.png"
                msk_out = proc_dir / "masks" / f"{sid}.png"

                if not args.overwrite and img_out.exists() and msk_out.exists():
                    pass  # skip write, but still record
                else:
                    Image.fromarray(rgb_r, mode="RGB").save(img_out)
                    Image.fromarray(mask_r, mode="L").save(msk_out)

                all_rows.append({
                    "image_path": str(img_out.as_posix()),
                    "mask_path": str(msk_out.as_posix()),
                    "id": sid,
                    "patient_id": patient_id,
                    "dataset": name,
                    "split": raw_split,
                    "spacing_yx": [float(spacing_yx[0]), float(spacing_yx[1])],
                    "source_label": str(case["label_path"].as_posix()),
                })
                stats_slices += 1

    print(f"\n[{name}] Total slices: {stats_slices}, skipped (empty): {stats_skipped}")
    print(f"[{name}] Split distribution:")
    from collections import Counter
    split_counts = Counter(r["split"] for r in all_rows)
    for s, c in sorted(split_counts.items()):
        print(f"  {s}: {c}")

    index_path = splits_dir / "index_all.jsonl"
    save_jsonl(index_path, all_rows)
    print(f"\nWrote master index: {index_path}")


if __name__ == "__main__":
    main()

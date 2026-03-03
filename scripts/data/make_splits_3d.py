"""
Generate volume-level 3D splits for Seg-MoE 3D pipeline.

Derives patient-level fold assignments from the existing 2D slice splits
(same group_key=patient_id), ensuring identical fold boundaries for
directly comparable 2D vs 3D experiments.

Output: data/splits/prostate_local_3d/splits_train5fold_testfixed.jsonl
Each row:
    {
      "id":          "NJMU_0000005734",         # patient ID (=volume ID)
      "patient_id":  "NJMU_0000005734",
      "split":       "train_fold0" | "val_fold0" | ... | "test",
      "image_paths": [".../_0000.nii.gz", ".../_0001.nii.gz", ".../_0002.nii.gz"],
      "mask_path":   ".../labelsTr/NJMU_0000005734.nii.gz"
    }

Usage:
    python scripts/data/make_splits_3d.py \\
        --dataset-config configs/3d/datasets/prostate_local_3d.yaml \\
        --source-splits  data/splits/prostate_local/splits_train5fold_testfixed.jsonl

    # For datasets where each patient has multiple series (e.g. ADC/T1/T2),
    # use patient_group_regex in dataset config to group series by real patient ID,
    # preventing the same patient from appearing in both train and val:
    python scripts/data/make_splits_3d.py \\
        --dataset-config configs/3d/datasets/liver_3d.yaml
    # (liver_3d.yaml sets patient_group_regex: '^(\\d+)_' to extract numeric prefix)
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


def _find_nifti_paths(raw_dir: Path, patient_id: str, modality_suffixes: List[str]) -> List[str]:
    """Find multi-modal NIfTI image paths for a patient."""
    images_dir = raw_dir / "imagesTr"
    paths = []
    for suf in modality_suffixes:
        p = images_dir / f"{patient_id}{suf}"
        if p.exists():
            paths.append(str(p))
    if not paths:
        # Single-file fallback: patient_id.nii.gz
        p = images_dir / f"{patient_id}.nii.gz"
        if p.exists():
            paths = [str(p)]
    return paths


def _find_mask_path(raw_dir: Path, patient_id: str) -> Optional[str]:
    labels_dir = raw_dir / "labelsTr"
    for ext in [".nii.gz", ".nii", ".mhd", ".nrrd"]:
        p = labels_dir / f"{patient_id}{ext}"
        if p.exists():
            return str(p)
    return None


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _save_jsonl(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {len(rows)} rows → {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate 3D volume-level splits")
    ap.add_argument("--dataset-config", required=True, help="configs/3d/datasets/prostate_local_3d.yaml")
    ap.add_argument("--source-splits",  default=None,
                    help="Path to existing 2D splits JSONL (optional, used to derive fold assignments). "
                         "If not provided, discovers all volumes and creates fresh 5-fold splits.")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-ratio", type=float, default=0.1,
                    help="Fraction of patients to hold out as fixed test set (only used when no source-splits)")
    args = ap.parse_args()

    import yaml
    with open(args.dataset_config, "r", encoding="utf-8") as f:
        dcfg = yaml.safe_load(f)

    raw_dir = Path(dcfg["paths"]["raw_dir"])
    splits_dir = Path(dcfg["paths"]["splits_dir"])
    modality_suffixes = dcfg["raw_structure"].get("modality_suffixes", ["_0000.nii.gz"])
    out_path = splits_dir / "splits_train5fold_testfixed.jsonl"

    # patient_group_regex: optional pattern to extract the "real" patient ID
    # from a case ID that includes a series suffix.
    # e.g. for liver dataset: case_id = "1075128_ADC" → group_id = "1075128"
    # Set in dataset config yaml as:   split.patient_group_regex: '^(\d+)_'
    patient_group_regex: Optional[str] = (
        dcfg.get("split", {}).get("patient_group_regex") or None
    )

    # ── Collect all patients with valid NIfTI files ──
    images_tr = raw_dir / "imagesTr"
    labels_tr = raw_dir / "labelsTr"

    if not images_tr.exists():
        raise FileNotFoundError(f"imagesTr not found at {images_tr}")

    # Find all unique patient IDs from label files
    label_files = sorted(labels_tr.glob("*.nii.gz")) + sorted(labels_tr.glob("*.nii"))
    patient_ids = []
    for lf in label_files:
        pid = lf.name.replace(".nii.gz", "").replace(".nii", "")
        img_paths = _find_nifti_paths(raw_dir, pid, modality_suffixes)
        if img_paths:
            patient_ids.append(pid)

    print(f"Found {len(patient_ids)} patients with images + labels.")

    # ── Build fold assignment ──
    if args.source_splits and Path(args.source_splits).exists():
        # Derive from 2D splits: extract patient→split mapping
        src_rows = _load_jsonl(Path(args.source_splits))
        pid_to_splits: Dict[str, set[str]] = defaultdict(set)
        for r in src_rows:
            pid = str(r.get("patient_id", r.get("id", "")))
            spl = str(r.get("split", ""))
            if pid and spl:
                pid_to_splits[pid].add(spl)

        rows: List[Dict[str, Any]] = []
        missing = []
        for pid in patient_ids:
            splits = pid_to_splits.get(pid)
            if not splits:
                missing.append(pid)
                splits = {"train_fold0"}   # fallback
            img_paths = _find_nifti_paths(raw_dir, pid, modality_suffixes)
            mask_path = _find_mask_path(raw_dir, pid)
            for spl in sorted(splits):
                row = {
                    "id":          pid,
                    "patient_id":  pid,
                    "split":       spl,
                    "image_paths": img_paths,
                    "mask_path":   mask_path or "",
                }
                rows.append(row)

        if missing:
            print(f"WARNING: {len(missing)} patients not in source splits, assigned to train_fold0: {missing[:5]}")

    else:
        # Fresh stratified k-fold split
        import random
        rng = random.Random(args.seed)

        # ── Build case_id → group_id mapping ──────────────────────────────
        # When patient_group_regex is set (e.g. '^(\d+)_' for liver),
        # multiple case IDs (series) map to the same group (real patient).
        # Fold assignment is done at the group level so that all series of the
        # same patient always land in the same fold, preventing data leakage.
        def get_group_id(pid: str) -> str:
            if patient_group_regex:
                m = re.match(patient_group_regex, pid)
                if m:
                    return m.group(1)
            return pid

        # Group case_ids by their group_id
        group_to_cases: Dict[str, List[str]] = defaultdict(list)
        for pid in patient_ids:
            group_to_cases[get_group_id(pid)].append(pid)

        unique_groups = list(group_to_cases.keys())
        rng.shuffle(unique_groups)

        if len(unique_groups) != len(patient_ids):
            print(f"Patient-level grouping active: {len(patient_ids)} cases → "
                  f"{len(unique_groups)} unique patients (avg {len(patient_ids)/len(unique_groups):.1f} series/patient)")

        n_test = max(1, int(len(unique_groups) * args.test_ratio))
        test_groups = set(unique_groups[:n_test])
        train_val_groups = unique_groups[n_test:]

        n_folds = args.n_folds
        fold_size = len(train_val_groups) // n_folds

        # Assign each group to a val fold index
        group_to_fold: Dict[str, int] = {}
        for i, grp in enumerate(train_val_groups):
            group_to_fold[grp] = min(i // fold_size, n_folds - 1)

        rows = []
        for pid in patient_ids:
            img_paths = _find_nifti_paths(raw_dir, pid, modality_suffixes)
            mask_path = _find_mask_path(raw_dir, pid)
            grp = get_group_id(pid)

            if grp in test_groups:
                spl = "test"
            else:
                fold_id = group_to_fold[grp]
                spl = f"val_fold{fold_id}"

            rows.append({
                "id":          pid,
                "patient_id":  pid,
                "group_id":    grp,          # real patient ID for reference
                "split":       spl,
                "image_paths": img_paths,
                "mask_path":   mask_path or "",
            })

        # Add mirror train splits
        all_rows_expanded = []
        pid_to_row = {r["id"]: r for r in rows}
        for fold_id in range(n_folds):
            for pid in patient_ids:
                r = pid_to_row[pid]
                if r["split"] == "test":
                    if fold_id == 0:
                        all_rows_expanded.append(dict(r))
                elif r["split"] == f"val_fold{fold_id}":
                    all_rows_expanded.append(dict(r))
                else:
                    new_r = dict(r)
                    new_r["split"] = f"train_fold{fold_id}"
                    all_rows_expanded.append(new_r)

        # Deduplicate: each patient appears once per fold (train) + once as val
        seen = set()
        rows = []
        for r in all_rows_expanded:
            key = (r["id"], r["split"])
            if key not in seen:
                seen.add(key)
                rows.append(r)

    _save_jsonl(rows, out_path)

    # Summary
    fold_counts: Dict[str, int] = defaultdict(int)
    for r in rows:
        fold_counts[r["split"]] += 1
    print("Split distribution:")
    for k in sorted(fold_counts):
        print(f"  {k}: {fold_counts[k]} volumes")


if __name__ == "__main__":
    main()

"""
Scan nnUNet raw dataset for shape/spacing mismatches using nnUNet's OWN reader.

This is necessary because nnUNet's SimpleITKIO may interpret NIfTI headers
differently from plain SimpleITK, leading to shape mismatches that only
surface during nnUNetv2_plan_and_preprocess.

Usage:
    python scripts/data/scan_nnunet_shape_mismatch.py ^
      --dataset-root D:/Seg-MoE/nnunet_data/nnUNet_raw/Dataset003_v2_LiverTumorSeg

    # Actually quarantine bad cases:
    python scripts/data/scan_nnunet_shape_mismatch.py ^
      --dataset-root D:/Seg-MoE/nnunet_data/nnUNet_raw/Dataset003_v2_LiverTumorSeg ^
      --apply
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Scan for shape mismatches using nnUNet reader")
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--apply", action="store_true", help="Quarantine bad cases")
    ap.add_argument("--quarantine-dir", default="quarantined_bad_cases",
                    help="Subfolder under dataset-root for quarantined files")
    args = ap.parse_args()

    root = Path(args.dataset_root)
    img_dir = root / "imagesTr"
    lab_dir = root / "labelsTr"
    quar = root / args.quarantine_dir

    # Use nnUNet's own reader
    try:
        from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
    except ImportError:
        print("ERROR: nnunetv2 not installed. pip install nnunetv2")
        sys.exit(1)

    reader = SimpleITKIO()

    # Discover all label files
    label_files = sorted(lab_dir.glob("*.nii.gz")) + sorted(lab_dir.glob("*.nii"))
    print(f"Labels found: {len(label_files)}")

    bad_cases = []
    checked = 0

    for lf in label_files:
        case_id = lf.name.replace(".nii.gz", "").replace(".nii", "")
        img_file = img_dir / f"{case_id}_0000.nii.gz"
        if not img_file.exists():
            img_file = img_dir / f"{case_id}_0000.nii"
        if not img_file.exists():
            print(f"[MISSING] {case_id}: no paired image")
            bad_cases.append((case_id, "missing_image"))
            continue

        checked += 1
        try:
            img_data, img_props = reader.read_images([str(img_file)])
            seg_data, seg_props = reader.read_seg(str(lf))

            img_shape = img_data.shape[1:]  # remove channel dim
            seg_shape = seg_data.shape[1:]

            if img_shape != seg_shape:
                print(f"[SHAPE] {case_id}: img={img_shape} seg={seg_shape}")
                bad_cases.append((case_id, f"shape_img={img_shape}_seg={seg_shape}"))
                continue

            # Check spacing
            import numpy as np
            if img_props.get("spacing") is not None and seg_props.get("spacing") is not None:
                sp_img = np.array(img_props["spacing"])
                sp_seg = np.array(seg_props["spacing"])
                if sp_img.shape != sp_seg.shape or not np.allclose(sp_img, sp_seg):
                    print(f"[SPACING] {case_id}: img={sp_img} seg={sp_seg}")
                    bad_cases.append((case_id, f"spacing"))

        except Exception as e:
            print(f"[ERROR] {case_id}: {e}")
            bad_cases.append((case_id, f"read_error: {e}"))

        if checked % 500 == 0:
            print(f"  ... checked {checked}/{len(label_files)}, bad so far: {len(bad_cases)}")

    print(f"\n=== Scan Summary ===")
    print(f"Total labels : {len(label_files)}")
    print(f"Checked pairs: {checked}")
    print(f"Bad cases     : {len(bad_cases)}")

    if not bad_cases:
        print("All cases OK!")
        return

    print("\nBad case list:")
    for cid, reason in bad_cases:
        print(f"  {cid}: {reason}")

    if not args.apply:
        print("\nDry-run. Re-run with --apply to quarantine these cases.")
        return

    # Quarantine
    q_img = quar / "imagesTr"
    q_lab = quar / "labelsTr"
    q_img.mkdir(parents=True, exist_ok=True)
    q_lab.mkdir(parents=True, exist_ok=True)

    moved = 0
    for cid, reason in bad_cases:
        # Move label
        for ext in [".nii.gz", ".nii"]:
            lp = lab_dir / f"{cid}{ext}"
            if lp.exists():
                shutil.move(str(lp), str(q_lab / lp.name))

        # Move all matching image channels (_0000, _0001, ...)
        for ip in sorted(img_dir.glob(f"{cid}_*.nii*")):
            shutil.move(str(ip), str(q_img / ip.name))

        moved += 1

    print(f"\nQuarantined {moved} cases → {quar}")

    # Update dataset.json numTraining
    dj_path = root / "dataset.json"
    if dj_path.exists():
        with open(dj_path, "r", encoding="utf-8") as f:
            dj = json.load(f)
        old_n = dj.get("numTraining", "?")
        remaining = len(list(lab_dir.glob("*.nii.gz"))) + len(list(lab_dir.glob("*.nii")))
        dj["numTraining"] = remaining
        with open(dj_path, "w", encoding="utf-8") as f:
            json.dump(dj, f, indent=2, ensure_ascii=False)
        print(f"Updated dataset.json: numTraining {old_n} → {remaining}")

    # Save quarantine log
    log_path = quar / "quarantine_log.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        for cid, reason in bad_cases:
            f.write(f"{cid}\t{reason}\n")
    print(f"Log saved → {log_path}")


if __name__ == "__main__":
    main()

"""
Fix nnUNet image/label geometry mismatches in nnUNet_raw dataset folders.

Default mode is dry-run (report only). Use --apply to modify files.

Handled mismatch types:
1) spacing/origin/direction mismatch with SAME size:
     - keep label voxel values unchanged
     - overwrite label geometry metadata to match image geometry

2) shape/dimension mismatch (for example 2D image vs 3D label):
     - considered invalid for nnUNet 3D
     - can be moved to quarantine with --quarantine-invalid

Optional:
- --resample-size-mismatch: force nearest-neighbor resample label to image grid
    (disabled by default because this may be semantically wrong on badly paired data)

Example:
        python scripts/data/fix_nnunet_spacing_mismatch.py \
            --dataset-root D:/Seg-MoE/nnunet_data/nnUNet_raw/Dataset003_v2_LiverTumorSeg

        python scripts/data/fix_nnunet_spacing_mismatch.py \
            --dataset-root D:/Seg-MoE/nnunet_data/nnUNet_raw/Dataset003_v2_LiverTumorSeg \
            --apply --quarantine-invalid
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable, Tuple

import SimpleITK as sitk


def _iter_label_files(labels_dir: Path) -> Iterable[Path]:
    yield from sorted(labels_dir.glob("*.nii.gz"))
    yield from sorted(labels_dir.glob("*.nii"))


def _case_id_from_label(label_path: Path) -> str:
    name = label_path.name
    if name.endswith(".nii.gz"):
        return name[: -len(".nii.gz")]
    if name.endswith(".nii"):
        return name[: -len(".nii")]
    return label_path.stem


def _geom_diff(
    img: sitk.Image,
    lab: sitk.Image,
    tol: float,
) -> Tuple[bool, bool, bool, bool]:
    spacing_diff = any(abs(a - b) > tol for a, b in zip(img.GetSpacing(), lab.GetSpacing()))
    origin_diff = any(abs(a - b) > tol for a, b in zip(img.GetOrigin(), lab.GetOrigin()))
    direction_diff = any(abs(a - b) > tol for a, b in zip(img.GetDirection(), lab.GetDirection()))
    size_diff = img.GetSize() != lab.GetSize()
    return spacing_diff, origin_diff, direction_diff, size_diff


def _backup_file(src: Path, backup_root: Path) -> None:
    backup_root.mkdir(parents=True, exist_ok=True)
    dst = backup_root / src.name
    if not dst.exists():
        shutil.copy2(src, dst)


def _move_to_quarantine(path: Path, quarantine_root: Path, subdir: str) -> Path:
    dst_dir = quarantine_root / subdir
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / path.name
    shutil.move(str(path), str(dst))
    return dst


def main() -> None:
    parser = argparse.ArgumentParser(description="Fix nnUNet spacing/origin/direction mismatches")
    parser.add_argument("--dataset-root", required=True, help="Path like .../nnUNet_raw/DatasetXXX_xxx")
    parser.add_argument("--apply", action="store_true", help="Actually write fixed labels")
    parser.add_argument("--tol", type=float, default=1e-6, help="Geometry tolerance")
    parser.add_argument(
        "--backup-dir",
        default="labelsTr_backup_before_geom_fix",
        help="Backup folder name under dataset root (used only with --apply)",
    )
    parser.add_argument(
        "--quarantine-invalid",
        action="store_true",
        help="Move invalid shape/dimension mismatch pairs to quarantine when --apply is set",
    )
    parser.add_argument(
        "--quarantine-dir",
        default="invalid_cases_after_geom_check",
        help="Quarantine folder name under dataset root",
    )
    parser.add_argument(
        "--resample-size-mismatch",
        action="store_true",
        help="Force nearest-neighbor resampling when size mismatches (use with caution)",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    images_dir = dataset_root / "imagesTr"
    labels_dir = dataset_root / "labelsTr"

    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(f"Expected imagesTr and labelsTr under: {dataset_root}")

    backup_root = dataset_root / args.backup_dir
    quarantine_root = dataset_root / args.quarantine_dir

    total = 0
    missing_image = 0
    mismatched = 0
    fixed = 0
    quarantined = 0
    invalid_shape = 0

    for lab_path in _iter_label_files(labels_dir):
        total += 1
        case_id = _case_id_from_label(lab_path)
        img_path = images_dir / f"{case_id}_0000.nii.gz"
        if not img_path.exists():
            missing_image += 1
            continue

        img = sitk.ReadImage(str(img_path))
        lab = sitk.ReadImage(str(lab_path))

        spacing_diff, origin_diff, direction_diff, size_diff = _geom_diff(img, lab, args.tol)
        dim_diff = img.GetDimension() != lab.GetDimension()
        if not (spacing_diff or origin_diff or direction_diff or size_diff):
            continue

        mismatched += 1
        print(
            f"[MISMATCH] {case_id} | "
            f"dim img/lab={img.GetDimension()}/{lab.GetDimension()} | "
            f"size img/lab={img.GetSize()}/{lab.GetSize()} | "
            f"spacing img/lab={img.GetSpacing()}/{lab.GetSpacing()}"
        )

        severe_shape_mismatch = dim_diff or size_diff
        if severe_shape_mismatch and not args.resample_size_mismatch:
            invalid_shape += 1

            if args.apply and args.quarantine_invalid:
                _backup_file(lab_path, backup_root)
                _backup_file(img_path, backup_root)
                moved_lab = _move_to_quarantine(lab_path, quarantine_root, "labelsTr")
                moved_img = _move_to_quarantine(img_path, quarantine_root, "imagesTr")
                quarantined += 1
                print(f"  -> quarantined: {moved_img} | {moved_lab}")

            continue

        if not args.apply:
            continue

        _backup_file(lab_path, backup_root)

        if not size_diff and not dim_diff:
            out = sitk.Image(lab)
            out.SetSpacing(img.GetSpacing())
            out.SetOrigin(img.GetOrigin())
            out.SetDirection(img.GetDirection())
        else:
            out = sitk.Resample(
                lab,
                img,
                sitk.Transform(),
                sitk.sitkNearestNeighbor,
                0,
                lab.GetPixelID(),
            )

        sitk.WriteImage(out, str(lab_path), useCompression=True)
        fixed += 1

    print("\n=== Summary ===")
    print(f"Total labels scanned: {total}")
    print(f"Missing paired image : {missing_image}")
    print(f"Geometry mismatches  : {mismatched}")
    print(f"Invalid shape/dim    : {invalid_shape}")
    if args.apply:
        print(f"Fixed labels         : {fixed}")
        print(f"Quarantined pairs    : {quarantined}")
        print(f"Backup folder        : {backup_root}")
        if args.quarantine_invalid:
            print(f"Quarantine folder    : {quarantine_root}")
    else:
        print("Dry-run only. Re-run with --apply to fix.")


if __name__ == "__main__":
    main()

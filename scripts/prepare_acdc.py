from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from seg_moe.utils.config import load_config
from seg_moe.utils.io import ensure_dir, save_jsonl


def _read_nii(path: Path) -> tuple[np.ndarray, tuple[float, float]]:
    import SimpleITK as sitk

    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)  # [z,y,x] or [y,x]
    spacing = img.GetSpacing()  # (x,y,z)
    sx, sy = float(spacing[0]), float(spacing[1])
    if arr.ndim == 2:
        arr = arr[None]
    return arr, (sy, sx)


def _to_uint8(vol: np.ndarray) -> np.ndarray:
    v = vol.astype(np.float32)
    v = v - float(np.min(v))
    mx = float(np.max(v))
    if mx > 0:
        v = v / mx
    return (v * 255.0).clip(0, 255).astype(np.uint8)


def _resize2d(arr: np.ndarray, size_hw: tuple[int, int], is_mask: bool) -> np.ndarray:
    H, W = size_hw
    pil = Image.fromarray(arr)
    pil = pil.resize((W, H), resample=Image.NEAREST if is_mask else Image.BILINEAR)
    return np.array(pil)


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare ACDC NIfTI into 2D slices")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    raw_dir = Path(cfg["paths"]["raw_dir"])
    proc_dir = Path(cfg["paths"]["processed_dir"])
    splits_dir = Path(cfg["paths"]["splits_dir"])

    ensure_dir(proc_dir / "images")
    ensure_dir(proc_dir / "masks")
    ensure_dir(splits_dir)

    H, W = int(cfg["input"]["image_size"][0]), int(cfg["input"]["image_size"][1])
    num_classes = int(cfg["task"]["num_classes"])

    train_root = raw_dir / str(cfg["raw_structure"]["training_dir"])
    if not train_root.exists():
        raise FileNotFoundError(f"Missing ACDC training dir: {train_root}")

    label_map = {int(a): int(b) for a, b in (cfg["task"].get("label_map") or {}).items()}

    rows = []
    uniq_vals = set()

    # Expected: training/patientXXX/*_frameXX.nii.gz and *_frameXX_gt.nii.gz
    img_files = sorted(train_root.rglob("*_frame*.nii*"))
    img_files = [p for p in img_files if "_gt" not in p.name]

    for img_p in img_files:
        gt_p = img_p.with_name(img_p.stem + "_gt" + img_p.suffix)
        if not gt_p.exists() and img_p.name.endswith(".nii.gz"):
            # handle Path.stem strips only .gz -> need special-case
            base = img_p.name.replace(".nii.gz", "")
            gt_p = img_p.with_name(base + "_gt.nii.gz")
        if not gt_p.exists():
            continue

        # patient_id = parent folder name (patientXXX)
        patient_id = img_p.parent.name

        img_zyx, spacing_yx = _read_nii(img_p)
        msk_zyx, _ = _read_nii(gt_p)

        img_u8 = _to_uint8(img_zyx)
        msk = msk_zyx.astype(np.int64)
        if label_map:
            mapped = msk.copy()
            for a, b in label_map.items():
                mapped[msk == a] = b
            msk = mapped

        for z in range(img_u8.shape[0]):
            m = msk[z]
            uniq_vals.update(np.unique(m).tolist())
            if int(np.sum(m > 0)) < 10:
                continue

            img2 = _resize2d(img_u8[z], (H, W), is_mask=False)
            m2 = _resize2d(m.astype(np.uint8), (H, W), is_mask=True)

            if np.any(m2 < 0) or np.any(m2 >= num_classes):
                raise ValueError(f"Mask out of range in {img_p.name} z={z}: min={m2.min()} max={m2.max()} num_classes={num_classes}")

            sid = f"{patient_id}_{img_p.stem}_z{z:03d}"
            img_out = proc_dir / "images" / f"{sid}.png"
            msk_out = proc_dir / "masks" / f"{sid}.png"

            Image.fromarray(img2, mode="L").save(img_out)
            Image.fromarray(m2, mode="L").save(msk_out)

            rows.append(
                {
                    "image_path": str(img_out.as_posix()),
                    "mask_path": str(msk_out.as_posix()),
                    "id": sid,
                    "patient_id": patient_id,
                    "dataset": cfg["name"],
                    "split": "all",
                    "spacing_yx": [float(spacing_yx[0]), float(spacing_yx[1])],
                    "source_image": str(img_p.as_posix()),
                    "source_label": str(gt_p.as_posix()),
                }
            )

    print(f"[{cfg['name']}] prepared slices: {len(rows)}")
    print(f"[{cfg['name']}] unique label values (sampled): {sorted(list(uniq_vals))[:50]}")

    index_path = splits_dir / "index_all.jsonl"
    save_jsonl(index_path, rows)
    print(f"Wrote master index: {index_path}")


if __name__ == "__main__":
    main()

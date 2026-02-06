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
    spacing = img.GetSpacing()  # (x,y,z) or (x,y)
    sx, sy = float(spacing[0]), float(spacing[1])
    if arr.ndim == 2:
        arr = arr[None]
    return arr, (sy, sx)


def _to_uint8(image_zyx: np.ndarray, mode: str, ct_window: tuple[float, float] | None) -> np.ndarray:
    img = image_zyx.astype(np.float32)
    if mode == "ct_window" and ct_window is not None:
        lo, hi = float(ct_window[0]), float(ct_window[1])
        img = np.clip(img, lo, hi)
        img = (img - lo) / max(1e-6, (hi - lo))
    else:
        # robust minmax per-volume
        img = img - float(np.min(img))
        mx = float(np.max(img))
        if mx > 0:
            img = img / mx
    return (img * 255.0).clip(0, 255).astype(np.uint8)


def _resize2d(arr: np.ndarray, size_hw: tuple[int, int], is_mask: bool) -> np.ndarray:
    H, W = size_hw
    pil = Image.fromarray(arr)
    pil = pil.resize((W, H), resample=Image.NEAREST if is_mask else Image.BILINEAR)
    return np.array(pil)


def _default_patient_id_from_stem(stem: str) -> str:
    # strip double extensions like .nii.gz handled by Path.stem only once
    # We still keep full stem as patient id by default.
    return stem


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare 3D NIfTI volumes into 2D PNG slices + index_all.jsonl")
    ap.add_argument("--config", required=True)
    ap.add_argument("--images-dir", default=None, help="Override raw_structure images dir")
    ap.add_argument("--labels-dir", default=None, help="Override raw_structure labels dir")
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

    ncfg = cfg.get("nifti", {}) or {}
    slice_policy = str(ncfg.get("slice_policy", "non_empty"))
    min_fg = int(ncfg.get("min_fg_pixels", 10))
    intensity_mode = str((ncfg.get("intensity", {}) or {}).get("mode", "minmax")).lower()
    ct_window = (ncfg.get("intensity", {}) or {}).get("ct_window")

    label_map = {int(a): int(b) for a, b in (cfg["task"].get("label_map") or {}).items()}

    rs = cfg.get("raw_structure", {}) or {}
    images_tr_dir = args.images_dir or rs.get("images_tr_dir") or rs.get("images_dir")
    labels_tr_dir = args.labels_dir or rs.get("labels_tr_dir") or rs.get("labels_dir")
    label_suffix = rs.get("label_suffix")

    if not images_tr_dir or not labels_tr_dir:
        raise ValueError("Need raw_structure.images_tr_dir and raw_structure.labels_tr_dir (or pass --images-dir/--labels-dir)")

    img_root = raw_dir / str(images_tr_dir)
    lab_root = raw_dir / str(labels_tr_dir)
    if not img_root.exists():
        raise FileNotFoundError(f"Missing images dir: {img_root}")
    if not lab_root.exists():
        raise FileNotFoundError(f"Missing labels dir: {lab_root}")

    img_files = sorted(list(img_root.rglob("*.nii")) + list(img_root.rglob("*.nii.gz")))
    # Skip macOS resource forks (._*) and other hidden files
    img_files = [p for p in img_files if not p.name.startswith("._")]
    if not img_files:
        raise FileNotFoundError(f"No NIfTI files found in: {img_root}")

    rows = []
    uniq_vals = set()

    def _find_label_path(img_path: Path) -> Path | None:
        # Strategy:
        # 1) same filename
        # 2) same stem (.nii or .nii.gz)
        # 3) configurable suffix (e.g., BTCV: *_seg)
        # 4) common medical dataset suffixes
        name = img_path.name
        direct = lab_root / name
        if direct.exists():
            return direct

        # exact stem match
        candidates = list(lab_root.rglob(img_path.stem + ".nii.gz")) + list(lab_root.rglob(img_path.stem + ".nii"))
        if candidates:
            return Path(candidates[0])

        # suffix-based match (before extension)
        suffixes: list[str] = []
        if label_suffix:
            suffixes.append(str(label_suffix))
        suffixes.extend(["_seg", "_mask", "_label", "_gt"])

        base = name
        ext = ""
        if base.lower().endswith(".nii.gz"):
            base = base[: -len(".nii.gz")]
            ext = ".nii.gz"
        elif base.lower().endswith(".nii"):
            base = base[: -len(".nii")]
            ext = ".nii"

        for suf in suffixes:
            cand = lab_root / f"{base}{suf}{ext}"
            if cand.exists():
                return cand

        # last resort: glob with base prefix
        for suf in suffixes:
            globbed = list(lab_root.rglob(f"{base}{suf}.nii.gz")) + list(lab_root.rglob(f"{base}{suf}.nii"))
            if globbed:
                return Path(globbed[0])
        return None

    for img_p in img_files:
        lab_p = _find_label_path(img_p)
        if lab_p is None or not lab_p.exists():
            continue
        lab_p = Path(lab_p)

        img_zyx, spacing_yx = _read_nii(img_p)
        lab_zyx, _ = _read_nii(lab_p)

        img_u8 = _to_uint8(img_zyx, intensity_mode, tuple(ct_window) if ct_window else None)
        lab = lab_zyx.astype(np.int64)
        if label_map:
            mapped = lab.copy()
            for a, b in label_map.items():
                mapped[lab == a] = b
            lab = mapped

        # patient id
        pid = _default_patient_id_from_stem(img_p.stem)

        for z in range(img_u8.shape[0]):
            m = lab[z]
            uniq_vals.update(np.unique(m).tolist())

            if slice_policy == "non_empty":
                fg = int(np.sum(m > 0))
                if fg < min_fg:
                    continue

            img2 = _resize2d(img_u8[z], (H, W), is_mask=False)
            m2 = _resize2d(m.astype(np.uint8), (H, W), is_mask=True)

            # hard check range
            if np.any(m2 < 0) or np.any(m2 >= num_classes):
                raise ValueError(f"Mask out of range in {img_p.name} z={z}: min={m2.min()} max={m2.max()} num_classes={num_classes}")

            sid = f"{pid}_z{z:03d}"
            img_out = proc_dir / "images" / f"{sid}.png"
            msk_out = proc_dir / "masks" / f"{sid}.png"

            Image.fromarray(img2, mode="L").save(img_out)
            Image.fromarray(m2, mode="L").save(msk_out)

            rows.append(
                {
                    "image_path": str(img_out.as_posix()),
                    "mask_path": str(msk_out.as_posix()),
                    "id": sid,
                    "patient_id": pid,
                    "dataset": cfg["name"],
                    "split": "all",
                    "spacing_yx": [float(spacing_yx[0]), float(spacing_yx[1])],
                    "source_image": str(img_p.as_posix()),
                    "source_label": str(lab_p.as_posix()),
                }
            )

    print(f"[{cfg['name']}] prepared slices: {len(rows)}")
    print(f"[{cfg['name']}] unique label values (sampled): {sorted(list(uniq_vals))[:50]}")

    index_path = splits_dir / "index_all.jsonl"
    save_jsonl(index_path, rows)
    print(f"Wrote master index: {index_path}")


if __name__ == "__main__":
    main()

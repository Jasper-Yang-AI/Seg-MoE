from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from seg_moe.utils.config import load_config
from seg_moe.utils.io import ensure_dir, load_jsonl


def _iter_mask_paths(index_rows: list[dict]) -> list[Path]:
    paths = []
    for r in index_rows:
        p = r.get("mask_path")
        if p:
            paths.append(Path(p))
    return paths


def _sample_indices(n: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if k >= n:
        return np.arange(n)
    return rng.choice(n, size=k, replace=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit prepared masks: unique values, range, and class pixel ratios.")
    ap.add_argument("--dataset-config", required=True)
    ap.add_argument("--splits", action="store_true", help="Audit per-split using data/splits/<dataset>/*.jsonl (if available)")
    ap.add_argument("--sample", type=int, default=20, help="Number of samples to audit (set 0 for full scan)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None, help="Output JSON path (default: data/processed/<dataset>/label_audit.json)")
    args = ap.parse_args()

    dcfg = load_config(args.dataset_config)
    num_classes = int(dcfg["task"]["num_classes"])

    splits_dir = Path(dcfg["paths"]["splits_dir"])
    index_all = splits_dir / "index_all.jsonl"
    if not index_all.exists():
        raise FileNotFoundError(f"Missing index: {index_all}. Run prepare_* first.")

    rows_all = load_jsonl(index_all)

    # If requested, try to use split file (more strict, catches train/val/test mismatch)
    split_rows = None
    if args.splits:
        stype = dcfg["split"]["type"]
        if stype == "holdout20_then_5fold":
            split_file = splits_dir / "splits_holdout20_5fold.jsonl"
        elif stype == "train_5fold_test_fixed":
            split_file = splits_dir / "splits_train5fold_testfixed.jsonl"
        else:
            split_file = splits_dir / "splits_5fold.jsonl"
        if split_file.exists():
            split_rows = load_jsonl(split_file)

    processed_dir = Path(dcfg["paths"]["processed_dir"])
    out_path = Path(args.out) if args.out else (processed_dir / "label_audit.json")
    ensure_dir(out_path.parent)

    label_map = {int(k): int(v) for k, v in (dcfg["task"].get("label_map") or {}).items()}

    def audit_rows(rows: list[dict], name: str) -> dict:
        mask_paths = [Path(r["mask_path"]) for r in rows if r.get("mask_path")]
        n = len(mask_paths)
        if n == 0:
            return {"name": name, "n": 0, "error": "no masks"}

        idx = np.arange(n) if args.sample == 0 else _sample_indices(n, args.sample, args.seed)

        uniq_global: set[int] = set()
        counts = np.zeros((num_classes,), dtype=np.int64)
        out_of_range = 0
        missing_files = 0

        for j in idx:
            p = mask_paths[int(j)]
            if not p.exists():
                missing_files += 1
                continue
            arr = np.array(Image.open(p).convert("L"), dtype=np.int64)
            u = np.unique(arr)
            uniq_global.update([int(x) for x in u.tolist()])

            if np.any(arr < 0) or np.any(arr >= num_classes):
                out_of_range += 1

            for c in range(num_classes):
                counts[c] += int((arr == c).sum())

        total = int(counts.sum())
        ratios = {str(c): (float(counts[c]) / float(total) if total > 0 else 0.0) for c in range(num_classes)}

        return {
            "name": name,
            "n": n,
            "audited": int(len(idx)),
            "missing_files": int(missing_files),
            "unique_values": sorted(list(uniq_global)),
            "out_of_range_samples": int(out_of_range),
            "pixel_counts": {str(c): int(counts[c]) for c in range(num_classes)},
            "pixel_ratios": ratios,
        }

    report: dict = {
        "dataset": dcfg["name"],
        "num_classes": num_classes,
        "label_map": label_map,
        "index": str(index_all.as_posix()),
        "note": "unique/pixel_ratio computed on sampled masks unless --sample 0",
    }

    report["all"] = audit_rows(rows_all, "all")

    if split_rows is not None:
        by_split = defaultdict(list)
        for r in split_rows:
            by_split[str(r.get("split"))].append(r)
        report["splits"] = {k: audit_rows(v, k) for k, v in sorted(by_split.items())}

    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote: {out_path}")

    # Hard checks (exit code != 0 if violated)
    bad = []
    uniq = report["all"].get("unique_values", [])
    if any((x < 0 or x >= num_classes) for x in uniq):
        bad.append(f"unique_values out of range: {uniq}")
    if report["all"].get("out_of_range_samples", 0) > 0:
        bad.append("some sampled masks contain out-of-range class ids")

    # Warn if a class never appears in sampled pixels
    ratios = report["all"].get("pixel_ratios", {})
    missing_classes = [c for c in range(num_classes) if float(ratios.get(str(c), 0.0)) == 0.0]
    if missing_classes:
        print(f"WARNING: classes with 0 pixel ratio in audited set: {missing_classes}")

    if bad:
        raise SystemExit("\n".join(["LABEL AUDIT FAILED:"] + bad))


if __name__ == "__main__":
    main()

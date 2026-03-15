"""
Audit whether an OOF manifest is truly out-of-fold and leakage-free.

Checks:
1) Basic integrity: required fields, unique sample_id, prob_path exists.
2) Fold consistency: split == val_fold{k}, sample_fold == predictor_fold == k.
3) Split alignment: sample_id exists in provided split file and exactly matches expected val fold split.
4) Coverage: compare manifest sample counts vs split val counts per fold.
5) Optional checkpoint hint check: model_ckpt_paths contains fold{k} in path string.

Usage:
python scripts/utils/audit_oof_manifest.py \
  --manifest runs/segmoe_2d_prostate/cache/oof/layer1/oof_manifest.jsonl \
  --splits data/splits/prostate_local/splits_train5fold_testfixed.jsonl \
  --out runs/segmoe_2d_prostate/results/oof_audit_layer1.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _extract_fold_from_split(split: str) -> int | None:
    m = re.fullmatch(r"val_fold(\d+)", str(split))
    if not m:
        return None
    return int(m.group(1))


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit OOF manifest leakage and fold alignment")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--splits", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--check-ckpt-fold", action="store_true", help="Check model_ckpt_paths include fold{k}")
    args = ap.parse_args()

    manifest_rows = list(_read_jsonl(args.manifest))
    split_rows = list(_read_jsonl(args.splits))

    split_map: Dict[str, set[str]] = defaultdict(set)
    for r in split_rows:
        split_map[str(r["id"])].add(str(r.get("split", "")))
    split_val_counts = Counter()
    for r in split_rows:
        s = str(r.get("split", ""))
        if s.startswith("val_fold"):
            split_val_counts[s] += 1

    errors: List[str] = []
    warnings: List[str] = []
    seen = set()

    manifest_counts = Counter()
    fold_counts = Counter()
    fold_ckpt_mismatch = defaultdict(int)

    for i, r in enumerate(manifest_rows):
        prefix = f"row#{i} sample_id={r.get('sample_id')}"

        sid = str(r.get("sample_id", ""))
        if not sid:
            errors.append(f"{prefix}: missing sample_id")
            continue
        if sid in seen:
            errors.append(f"{prefix}: duplicate sample_id")
        seen.add(sid)

        split = str(r.get("split", ""))
        sf = r.get("sample_fold")
        pf = r.get("predictor_fold")
        pp = r.get("prob_path")

        if sf is None or pf is None:
            errors.append(f"{prefix}: missing sample_fold/predictor_fold")
            continue

        try:
            sf = int(sf)
            pf = int(pf)
        except Exception:
            errors.append(f"{prefix}: non-int sample_fold/predictor_fold")
            continue

        expected_split = f"val_fold{sf}"
        if split != expected_split:
            errors.append(f"{prefix}: split={split} != expected {expected_split}")

        if pf != sf:
            errors.append(f"{prefix}: predictor_fold({pf}) != sample_fold({sf})")

        # split-file alignment
        split_sid_set = split_map.get(sid)
        if split_sid_set is None:
            errors.append(f"{prefix}: not found in splits file")
        else:
            if expected_split not in split_sid_set:
                # In k-fold split files, each sample usually appears multiple times.
                # OOF row is valid if its expected val_fold exists among the sample's split labels.
                preview = sorted(list(split_sid_set))[:8]
                errors.append(
                    f"{prefix}: splits file labels {preview}, missing expected {expected_split}"
                )

        if pp is None:
            errors.append(f"{prefix}: missing prob_path")
        else:
            p = Path(str(pp))
            full = p if p.is_absolute() else (args.manifest.parent / p)
            if not full.exists():
                errors.append(f"{prefix}: prob_path missing on disk: {full}")

        if args.check_ckpt_fold:
            ck = r.get("model_ckpt_paths") or {}
            for _, v in ck.items():
                if f"fold{sf}" not in str(v).replace("/", "\\"):
                    fold_ckpt_mismatch[sf] += 1

        manifest_counts[split] += 1
        fold_counts[sf] += 1

    # Coverage checks
    for split_name, n_val in sorted(split_val_counts.items()):
        n_oof = manifest_counts.get(split_name, 0)
        if n_oof != n_val:
            warnings.append(f"coverage mismatch {split_name}: manifest={n_oof}, splits={n_val}")

    unexpected_splits = [s for s in manifest_counts if not str(s).startswith("val_fold")]
    if unexpected_splits:
        errors.append(f"manifest contains non-val splits: {sorted(unexpected_splits)}")

    report = {
        "manifest": str(args.manifest),
        "splits": str(args.splits),
        "n_manifest_rows": len(manifest_rows),
        "n_unique_sample_ids": len(seen),
        "manifest_counts_by_split": dict(sorted(manifest_counts.items())),
        "split_val_counts": dict(sorted(split_val_counts.items())),
        "fold_counts": dict(sorted((int(k), int(v)) for k, v in fold_counts.items())),
        "errors": errors,
        "warnings": warnings,
        "is_strict_oof": len(errors) == 0,
    }

    if args.check_ckpt_fold:
        report["ckpt_fold_mismatch_counts"] = dict(sorted((int(k), int(v)) for k, v in fold_ckpt_mismatch.items()))
        if any(v > 0 for v in fold_ckpt_mismatch.values()):
            warnings.append("Some model_ckpt_paths do not contain expected fold tag")

    out_path = args.out
    if out_path is None:
        out_path = args.manifest.parent / "oof_audit_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Saved audit report: {out_path}")
    print(f"is_strict_oof={report['is_strict_oof']} | errors={len(errors)} | warnings={len(warnings)}")
    if warnings:
        print("Top warnings:")
        for w in warnings[:10]:
            print(f"- {w}")
    if errors:
        print("Top errors:")
        for e in errors[:10]:
            print(f"- {e}")


if __name__ == "__main__":
    main()

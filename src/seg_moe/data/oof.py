from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from seg_moe.utils.io import load_jsonl


_VAL_FOLD_RE = re.compile(r"^val_fold(\d+)$")


@dataclass(frozen=True)
class OOFRecord:
    sample_id: str
    sample_fold: int
    predictor_fold: int
    prob_path: Path
    num_classes: int

    raw: Dict[str, Any]


def parse_val_fold(split: str) -> int | None:
    """Return the validation fold index encoded in split, else None."""

    match = _VAL_FOLD_RE.fullmatch(str(split).strip())
    return int(match.group(1)) if match else None


def resolve_prediction_cache_paths(
    exp_cfg: Mapping[str, Any],
    layer: str,
    *,
    predictor_fold: int,
    split: str,
) -> tuple[Path, Path]:
    """Resolve cache dir and manifest path for a prediction split.

    Validation folds keep using the shared OOF cache configured in the experiment
    because those artifacts are used for layer-wise training without leakage.

    Non-validation splits such as ``test`` are written to a dedicated inference
    cache rooted at:

      runs/${exp_name}/cache/inference/{layer}/fold_{predictor_fold}/{split}
    """

    exp_name = str(exp_cfg.get("exp_name", ""))
    layering = dict(exp_cfg.get("layering", {}) or {})

    def _resolve(path_like: str | Path) -> Path:
        return Path(str(path_like).replace("${exp_name}", exp_name))

    cache_root = _resolve(layering.get("cache_root", "runs/${exp_name}/cache"))
    split = str(split).strip()
    if not split:
        raise ValueError("split must be a non-empty string")

    if layer == "layer1":
        default_cache_dir = cache_root / "oof" / "layer1"
        cache_key = "oof_cache_dir"
        manifest_key = "oof_manifest_path"
        default_manifest = default_cache_dir / "oof_manifest.jsonl"
    elif layer == "layer2":
        default_cache_dir = cache_root / "oof" / "layer2"
        cache_key = "l2_oof_cache_dir"
        manifest_key = "l2_oof_manifest_path"
        default_manifest = default_cache_dir / "oof_manifest_layer2.jsonl"
    else:
        raise ValueError(f"Unsupported layer={layer!r}; expected 'layer1' or 'layer2'")

    if parse_val_fold(split) is not None:
        cache_dir = _resolve(layering.get(cache_key, default_cache_dir))
        manifest_path = _resolve(layering.get(manifest_key, default_manifest))
        return cache_dir, manifest_path

    infer_dir = cache_root / "inference" / layer / f"fold_{int(predictor_fold)}" / split
    return infer_dir, infer_dir / "manifest.jsonl"


def load_oof_manifest(manifest_path: str | Path, *, repo_root: Optional[str | Path] = None) -> Dict[str, OOFRecord]:
    """Load OOF manifest and return mapping sample_id -> OOFRecord.

    Path resolution:
    - If prob_path in manifest is absolute, use it.
    - Else resolve relative to manifest directory.
    - If repo_root provided, allow resolving relative to repo_root as fallback.

    Raises:
        FileNotFoundError: if manifest does not exist
        ValueError: if duplicate sample_id entries
    """

    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing OOF manifest: {manifest_path}. "
            "Run scripts/inference/generate_layer1_oof.py first (or disable use_oof_for_layer2)."
        )

    rows = load_jsonl(manifest_path)
    mapping: Dict[str, OOFRecord] = {}

    for r in rows:
        sid = str(r.get("sample_id"))
        if not sid or sid == "None":
            raise ValueError(f"Invalid manifest row missing sample_id: {r}")
        if sid in mapping:
            raise ValueError(f"Duplicate sample_id in manifest: {sid}")

        prob_path_raw = r.get("prob_path")
        if prob_path_raw is None:
            raise ValueError(f"Manifest row missing prob_path for sample_id={sid}")

        prob_path = Path(str(prob_path_raw))
        if not prob_path.is_absolute():
            cand = (manifest_path.parent / prob_path).resolve()
            if cand.exists():
                prob_path = cand
            elif repo_root is not None:
                cand2 = (Path(repo_root).resolve() / prob_path).resolve()
                prob_path = cand2
            else:
                prob_path = cand

        rec = OOFRecord(
            sample_id=sid,
            sample_fold=int(r.get("sample_fold")),
            predictor_fold=int(r.get("predictor_fold")),
            prob_path=prob_path,
            num_classes=int(r.get("num_classes")),
            raw=dict(r),
        )
        mapping[sid] = rec

    return mapping


def get_oof_prob_path(oof_map: Mapping[str, OOFRecord], sample_id: str) -> Path:
    """Return prob_path for sample_id or raise an actionable error."""

    if sample_id not in oof_map:
        raise KeyError(
            f"Missing OOF record for sample_id={sample_id}. "
            "You likely need to regenerate OOF cache for this dataset/experiment."
        )
    return oof_map[sample_id].prob_path

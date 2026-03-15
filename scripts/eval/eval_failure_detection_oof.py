"""
Failure detection from OOF ensemble uncertainty (2D).

This script evaluates whether uncertainty signals can identify low-Dice failures:
1) Correlation: high entropy/disagreement vs low Dice.
2) Classification: can risk scores detect worst-q Dice failures?
3) Triage: does top-k% highest-risk capture low-Dice cases?

Inputs:
- OOF manifest with per-sample npz paths (logits or probs, shape [K,M,H,W]).
- Splits/index jsonl with sample id -> mask path (+ patient id).

Outputs:
- per_sample_scores.csv
- per_patient_scores.csv
- summary.json

Usage:
python scripts/eval/eval_failure_detection_oof.py \
  --manifest runs/segmoe_2d_prostate/cache/oof/layer1/oof_manifest.jsonl \
  --splits data/splits/prostate_local/splits_train5fold_testfixed.jsonl \
  --outdir runs/segmoe_2d_prostate/results/failure_detection_oof
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import rankdata, spearmanr


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _load_probs(npz_path: Path) -> np.ndarray:
    d = np.load(npz_path)
    if "probs" in d:
        return d["probs"].astype(np.float32)
    if "logits" in d:
        logits = d["logits"].astype(np.float32)
        logits = logits - logits.max(axis=1, keepdims=True)
        ex = np.exp(logits)
        return ex / (ex.sum(axis=1, keepdims=True) + 1e-8)
    raise KeyError(f"Missing 'probs' or 'logits' in {npz_path}")


def _dice_per_class(pred: np.ndarray, target: np.ndarray, c: int) -> float:
    p = pred == c
    t = target == c
    inter = float(np.logical_and(p, t).sum())
    denom = float(p.sum() + t.sum())
    if denom == 0:
        return 1.0
    return (2.0 * inter) / (denom + 1e-8)


def _mean_fg_dice(pred: np.ndarray, target: np.ndarray, num_classes: int) -> float:
    vals = [_dice_per_class(pred, target, c) for c in range(1, num_classes)]
    return float(np.mean(vals)) if vals else 0.0


def _pairwise_tv_disagreement(probs_km: np.ndarray) -> np.ndarray:
    # probs_km: [K,M,H,W]
    k = probs_km.shape[0]
    if k < 2:
        return np.zeros(probs_km.shape[2:], dtype=np.float32)
    pairs = []
    for i in range(k):
        for j in range(i + 1, k):
            # Total variation distance per pixel in [0,1]
            tv = 0.5 * np.abs(probs_km[i] - probs_km[j]).sum(axis=0)
            pairs.append(tv)
    return np.mean(np.stack(pairs, axis=0), axis=0).astype(np.float32)


def _auc_from_scores(y_true: np.ndarray, score: np.ndarray) -> float:
    y_true = y_true.astype(np.int32)
    n_pos = int(y_true.sum())
    n_neg = int((1 - y_true).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(score)
    rank_pos = float(ranks[y_true == 1].sum())
    auc = (rank_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _evaluate_detection(df: pd.DataFrame, score_col: str, failure_q: float, risk_topk: float) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if df.empty:
        return out

    dice = df["dice_mean"].to_numpy(dtype=np.float64)
    score = df[score_col].to_numpy(dtype=np.float64)

    rho, pval = spearmanr(score, dice)

    # Define failure by worst-q Dice (low Dice is failure)
    thr = float(np.quantile(dice, failure_q))
    y_fail = (dice <= thr).astype(np.int32)

    auc = _auc_from_scores(y_fail, score)

    n = len(df)
    k = max(1, int(np.ceil(risk_topk * n)))
    top_idx = np.argsort(-score)[:k]

    precision = float(y_fail[top_idx].mean())
    recall = float(y_fail[top_idx].sum() / max(1, int(y_fail.sum())))
    base_rate = float(y_fail.mean())
    lift = float(precision / (base_rate + 1e-12))

    out["n"] = float(n)
    out["dice_failure_threshold"] = thr
    out["spearman_rho_score_vs_dice"] = float(rho)
    out["spearman_pvalue"] = float(pval)
    out["auroc_failure"] = auc
    out["risk_topk_fraction"] = float(risk_topk)
    out["failure_rate"] = base_rate
    out["topk_precision"] = precision
    out["topk_recall"] = recall
    out["topk_lift"] = lift
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Failure detection from OOF uncertainty")
    ap.add_argument("--manifest", type=Path, required=True, help="OOF manifest jsonl")
    ap.add_argument("--splits", type=Path, required=True, help="splits/index jsonl with id, mask_path, patient_id")
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--failure-quantile", type=float, default=0.10, help="Worst-q Dice as failures")
    ap.add_argument("--risk-topk", type=float, default=0.10, help="Top-k% risk triage size")
    ap.add_argument("--sample-fold", type=int, default=None, help="Only evaluate manifest rows with this sample_fold")
    ap.add_argument("--max-samples", type=int, default=0, help="Optional cap on number of processed samples (0=all)")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    idx_map: Dict[str, dict] = {}
    for r in _read_jsonl(args.splits):
        idx_map[str(r["id"])] = r

    rows: List[dict] = []
    manifest_dir = args.manifest.parent

    processed = 0
    for rec in _read_jsonl(args.manifest):
        if args.sample_fold is not None and int(rec.get("sample_fold", -1)) != int(args.sample_fold):
            continue

        sid = str(rec["sample_id"])
        meta = idx_map.get(sid)
        if meta is None:
            continue

        npz_path = manifest_dir / rec["prob_path"]
        if not npz_path.exists():
            continue

        probs_km = _load_probs(npz_path)  # [K,M,H,W]
        _, m, _, _ = probs_km.shape

        mean_probs = probs_km.mean(axis=0)  # [M,H,W]
        pred = mean_probs.argmax(axis=0).astype(np.int64)

        mask = np.array(Image.open(meta["mask_path"]).convert("L"), dtype=np.int64)
        dice_mean = _mean_fg_dice(pred, mask, m)

        # Entropy from mean probabilities, normalized to [0,1]
        entropy_map = -(mean_probs * np.log(mean_probs + 1e-8)).sum(axis=0) / (np.log(m) + 1e-8)
        entropy_mean = float(entropy_map.mean())

        # Expert std disagreement (same spirit as training feature)
        disagreement_map = probs_km.std(axis=0).mean(axis=0)  # mean over class -> [H,W]
        disagreement_mean = float(disagreement_map.mean())

        # Pairwise expert disagreement baseline (TV distance)
        pairwise_tv_map = _pairwise_tv_disagreement(probs_km)
        pairwise_tv_mean = float(pairwise_tv_map.mean())

        rows.append(
            {
                "sample_id": sid,
                "patient_id": meta.get("patient_id", sid.rsplit("_z", 1)[0]),
                "split": rec.get("split", meta.get("split")),
                "sample_fold": rec.get("sample_fold"),
                "predictor_fold": rec.get("predictor_fold"),
                "dice_mean": float(dice_mean),
                "entropy_mean": entropy_mean,
                "disagreement_mean": disagreement_mean,
                "pairwise_tv_mean": pairwise_tv_mean,
            }
        )
        processed += 1
        if args.max_samples > 0 and processed >= args.max_samples:
            break

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No matched samples found between manifest and splits/index")

    sample_csv = args.outdir / "per_sample_scores.csv"
    df.to_csv(sample_csv, index=False)

    # Patient-level aggregation for clinically meaningful triage
    g = (
        df.groupby("patient_id", as_index=False)
        .agg(
            dice_mean=("dice_mean", "mean"),
            dice_min=("dice_mean", "min"),
            entropy_mean=("entropy_mean", "mean"),
            disagreement_mean=("disagreement_mean", "mean"),
            pairwise_tv_mean=("pairwise_tv_mean", "mean"),
            n_slices=("sample_id", "count"),
        )
    )
    patient_csv = args.outdir / "per_patient_scores.csv"
    g.to_csv(patient_csv, index=False)

    summary = {
        "sample_level": {
            "entropy": _evaluate_detection(df, "entropy_mean", args.failure_quantile, args.risk_topk),
            "disagreement_std": _evaluate_detection(df, "disagreement_mean", args.failure_quantile, args.risk_topk),
            "pairwise_tv": _evaluate_detection(df, "pairwise_tv_mean", args.failure_quantile, args.risk_topk),
        },
        "patient_level": {
            "entropy": _evaluate_detection(g, "entropy_mean", args.failure_quantile, args.risk_topk),
            "disagreement_std": _evaluate_detection(g, "disagreement_mean", args.failure_quantile, args.risk_topk),
            "pairwise_tv": _evaluate_detection(g, "pairwise_tv_mean", args.failure_quantile, args.risk_topk),
        },
        "config": {
            "manifest": str(args.manifest),
            "splits": str(args.splits),
            "failure_quantile": args.failure_quantile,
            "risk_topk": args.risk_topk,
        },
    }

    out_json = args.outdir / "summary.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved: {sample_csv}")
    print(f"Saved: {patient_csv}")
    print(f"Saved: {out_json}")

    # Console quick view
    print("\n[Patient-level quick summary]")
    for key in ("entropy", "disagreement_std", "pairwise_tv"):
        s = summary["patient_level"][key]
        print(
            f"{key:>16s} | rho={s.get('spearman_rho_score_vs_dice', float('nan')): .3f} | "
            f"AUROC={s.get('auroc_failure', float('nan')): .3f} | "
            f"TopK-Prec={s.get('topk_precision', float('nan')): .3f} | "
            f"TopK-Recall={s.get('topk_recall', float('nan')): .3f}"
        )


if __name__ == "__main__":
    main()

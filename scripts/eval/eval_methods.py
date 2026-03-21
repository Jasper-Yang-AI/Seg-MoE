"""
Comprehensive evaluation: single experts + ensembles + combiners + gating.

科研专家级评估流程 (References):
  - Maier-Hein et al. 2024, "Metrics Reloaded", Nature Methods
    → DSC + HD95 + NSD 为三大核心指标; per-class 报告
  - Dang et al. 2024, "Two-layer Ensemble of Deep Learning Models"
    → Single / Mean / OLE / DT / WE-CLPSO 对比
  - Kittler et al. 1998, "On Combining Classifiers" (IEEE TPAMI)
    → Majority voting 作为 lower-bound baseline
  - Demšar 2006, "Statistical Comparisons of Classifiers" (JMLR)
    → Wilcoxon signed-rank test 用于配对统计检验

评估流程:
  Phase A: Layer1 单专家 (3 experts × 5 folds)
  Phase B: Layer2 单专家 (3 experts × 5 folds)
  Phase C: 集成方法 — Mean / Majority / OLE / DT / WE-CLPSO
           分别在 L1 OOF 和 L2 OOF 上评估
  Phase D: 门控融合 (从缓存结果加载)
  Phase E: 统计检验 (Wilcoxon signed-rank pairwise)

输出:
  - metrics_per_sample_{dataset}_{split}.csv  (per-sample, 用于统计检验)
  - metrics_summary_{dataset}_{split}.csv     (per-method 聚合)
  - weights_{dataset}_{split}.json            (combiner 权重)
  - significance_{dataset}_{split}.csv        (Wilcoxon p-values)

Usage:
    python scripts/eval/eval_methods.py \\
        --exp configs/2d/exp/exp_msd_task03_liver.yaml \\
        --training configs/2d/training.yaml \\
        --models configs/2d/models.yaml \\
        --fold 0 --gpus 0
"""
from __future__ import annotations

import argparse
import itertools
import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from seg_moe.combiners.decision_template import DecisionTemplateCombiner
from seg_moe.combiners.majority_voting import MajorityVotingCombiner
from seg_moe.combiners.ole import OLECombiner
from seg_moe.combiners.we_clpso import WECLPSOCombiner
from seg_moe.data.dataset_2d import SegmentationDataset2D
from seg_moe.data.indexing import infer_image_channels, infer_num_classes
from seg_moe.data.oof import load_oof_manifest, get_oof_prob_path
from seg_moe.evaluation.metrics_2d import compute_segmentation_metrics_batch
from seg_moe.models.factory_2d import build_expert, expert_name, list_experts
from seg_moe.utils.config import load_config, resolve_run_dir
from seg_moe.utils.io import ensure_dir, load_jsonl, save_json


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _load_splits(dataset_cfg: dict) -> list[dict]:
    splits_dir = Path(dataset_cfg["paths"]["splits_dir"])
    stype = dataset_cfg["split"]["type"]
    if stype == "holdout20_then_5fold":
        path = splits_dir / "splits_holdout20_5fold.jsonl"
    elif stype == "train_5fold_test_fixed":
        path = splits_dir / "splits_train5fold_testfixed.jsonl"
    else:
        path = splits_dir / "splits_5fold.jsonl"
    return load_jsonl(path)


def _ckpt(run_dir: Path, layer: str, fold: int, ex: str, which: str = "best") -> Path:
    return run_dir / "checkpoints" / layer / f"fold{fold}" / ex / f"{which}.pt"


def _resolve_manifest(exp_cfg: dict, key: str, default_subpath: str) -> Path:
    """Resolve a layering manifest path from exp config."""
    cache_root = Path(
        exp_cfg["layering"]["cache_root"].replace("${exp_name}", exp_cfg["exp_name"])
    )
    raw = exp_cfg.get("layering", {}).get(key, str(cache_root / default_subpath))
    return Path(str(raw).replace("${exp_name}", exp_cfg["exp_name"]))


def _parse_spacing(meta: dict) -> Optional[tuple[float, float]]:
    """Extract spacing_yx from sample metadata."""
    spacing = meta.get("spacing_yx")
    if spacing is None:
        return None
    if isinstance(spacing, (list, tuple)) and len(spacing) >= 2:
        return (float(spacing[0]), float(spacing[1]))
    if isinstance(spacing, torch.Tensor):
        s = spacing.tolist()
        if isinstance(s, list) and len(s) >= 2:
            return (float(s[0]), float(s[1]))
    return None


def _load_probs_from_cache(cache_path: Path) -> np.ndarray:
    """Load [K,M,H,W] probabilities from logits-first cache."""
    data = np.load(cache_path)
    if "logits" in data:
        logits = data["logits"].astype(np.float32)
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        return exp_logits / (exp_logits.sum(axis=1, keepdims=True) + 1e-8)
    if "probs" in data:
        return data["probs"].astype(np.float32)
    raise KeyError(f"Cache file missing 'logits'/'probs': {cache_path}")


# ═══════════════════════════════════════════════════════════════════════
# Phase A/B: Single expert evaluation
# ═══════════════════════════════════════════════════════════════════════

def _eval_single_expert(
    model: torch.nn.Module,
    dl: DataLoader,
    num_classes: int,
    device: torch.device,
) -> List[Dict[str, Any]]:
    """Evaluate a single expert, return per-sample metrics."""
    model.eval()
    per_sample: List[Dict[str, Any]] = []
    with torch.no_grad():
        for x, y, meta in dl:
            probs = torch.softmax(model(x.to(device)), dim=1)
            spacing = _parse_spacing(meta)
            m = compute_segmentation_metrics_batch(
                probs, y.to(device), num_classes=num_classes, spacing_yx=spacing,
            )
            ids = meta.get("id", [None])
            sid = ids[0] if isinstance(ids, (list, tuple)) else ids
            m["sample_id"] = str(sid)
            per_sample.append(m)
    return per_sample


# ═══════════════════════════════════════════════════════════════════════
# Phase C: Ensemble / Combiner evaluation
# ═══════════════════════════════════════════════════════════════════════

def _collect_oof_flat(
    rows: list[dict],
    oof_map: dict,
    dataset_cfg: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Load OOF probs and GT masks, flatten for combiner fitting.

    Returns X: [N, K, M], y: [N]
    """
    label_map = {
        int(k): int(v) for k, v in dataset_cfg["task"].get("label_map", {}).items()
    }
    from PIL import Image

    X_list, y_list = [], []
    for s in rows:
        sid = str(s["id"])
        if sid not in oof_map:
            continue
        prob_path = get_oof_prob_path(oof_map, sid)
        if not prob_path.exists():
            continue
        probs = _load_probs_from_cache(prob_path)  # [K,M,H,W]
        K, M, H, W = probs.shape
        # Load mask
        mask = np.array(Image.open(s["mask_path"]).convert("L"), dtype=np.uint8)
        if label_map:
            mapped = mask.copy()
            for k, v in label_map.items():
                mapped[mask == k] = v
            mask = mapped
        # Flatten
        pf = probs.transpose(2, 3, 0, 1).reshape(H * W, K, M)  # [HW, K, M]
        X_list.append(pf)
        y_list.append(mask.reshape(-1).astype(np.int64))

    if not X_list:
        return np.empty((0,)), np.empty((0,))
    return np.concatenate(X_list), np.concatenate(y_list)


def _eval_combiner_per_sample(
    combiner_predict_fn,
    eval_rows: list[dict],
    oof_map: dict,
    dataset_cfg: dict,
    num_classes: int,
) -> List[Dict[str, Any]]:
    """Evaluate a combiner on eval samples, return per-sample metrics."""
    label_map = {
        int(k): int(v) for k, v in dataset_cfg["task"].get("label_map", {}).items()
    }
    from PIL import Image

    per_sample: List[Dict[str, Any]] = []
    for s in eval_rows:
        sid = str(s["id"])
        if sid not in oof_map:
            continue
        prob_path = get_oof_prob_path(oof_map, sid)
        if not prob_path.exists():
            continue
        probs = _load_probs_from_cache(prob_path)  # [K,M,H,W]
        K, M, H, W = probs.shape

        # Load mask
        mask = np.array(Image.open(s["mask_path"]).convert("L"), dtype=np.uint8)
        if label_map:
            mapped = mask.copy()
            for k, v in label_map.items():
                mapped[mask == k] = v
            mask = mapped.astype(np.int64)

        # Flatten → predict → reshape
        pf = probs.transpose(2, 3, 0, 1).reshape(H * W, K, M)
        out = combiner_predict_fn(pf)  # [HW, M] or [HW]

        if out.ndim == 1:
            # Hard predictions (class indices) → one-hot
            pred = out.reshape(H, W).astype(np.int64)
            pred_1h = np.eye(num_classes, dtype=np.float32)[pred]  # [H,W,M]
            probs_t = torch.from_numpy(pred_1h.transpose(2, 0, 1)).unsqueeze(0)
        else:
            # Soft predictions [HW, M] → [1, M, H, W]
            fused = out.reshape(H, W, M).transpose(2, 0, 1)
            probs_t = torch.from_numpy(fused.astype(np.float32)).unsqueeze(0)

        mask_t = torch.from_numpy(mask.astype(np.int64)).unsqueeze(0)
        m = compute_segmentation_metrics_batch(probs_t, mask_t, num_classes=num_classes)
        m["sample_id"] = sid
        per_sample.append(m)
    return per_sample


def _eval_mean_ensemble_from_oof(
    eval_rows: list[dict],
    oof_map: dict,
    dataset_cfg: dict,
    num_classes: int,
) -> List[Dict[str, Any]]:
    """Mean ensemble using cached OOF probs (mean over K experts)."""
    label_map = {
        int(k): int(v) for k, v in dataset_cfg["task"].get("label_map", {}).items()
    }
    from PIL import Image

    per_sample: List[Dict[str, Any]] = []
    for s in eval_rows:
        sid = str(s["id"])
        if sid not in oof_map:
            continue
        prob_path = get_oof_prob_path(oof_map, sid)
        if not prob_path.exists():
            continue
        probs = _load_probs_from_cache(prob_path)  # [K,M,H,W]
        fused = probs.mean(axis=0)  # [M,H,W]

        mask = np.array(Image.open(s["mask_path"]).convert("L"), dtype=np.uint8)
        if label_map:
            mapped = mask.copy()
            for k, v in label_map.items():
                mapped[mask == k] = v
            mask = mapped.astype(np.int64)

        probs_t = torch.from_numpy(fused).unsqueeze(0).float()
        mask_t = torch.from_numpy(mask).unsqueeze(0)
        m = compute_segmentation_metrics_batch(probs_t, mask_t, num_classes=num_classes)
        m["sample_id"] = sid
        per_sample.append(m)
    return per_sample


# ═══════════════════════════════════════════════════════════════════════
# Phase D: Gating evaluation
# ═══════════════════════════════════════════════════════════════════════

def _eval_gating_from_cache(
    run_dir: Path, fold: int, split: str,
) -> Optional[Dict[str, Any]]:
    """Load pre-computed gating results from gating_inference.py output."""
    metrics_path = run_dir / "results" / "gating" / f"fold{fold}" / split / "metrics.json"
    if not metrics_path.exists():
        return None
    with open(metrics_path) as f:
        data = json.load(f)
    return data


# ═══════════════════════════════════════════════════════════════════════
# Phase E: Statistical significance testing
# ═══════════════════════════════════════════════════════════════════════

def _wilcoxon_pairwise(
    per_sample_df: pd.DataFrame,
    metric: str = "dice_mean",
) -> pd.DataFrame:
    """Wilcoxon signed-rank pairwise test between all method pairs.

    Demšar 2006, "Statistical Comparisons of Classifiers over
    Multiple Data Sets", JMLR.

    Returns DataFrame with columns: method_a, method_b, p_value, significant
    """
    from scipy.stats import wilcoxon

    methods = sorted(per_sample_df["method"].unique())
    rows = []
    for a, b in itertools.combinations(methods, 2):
        da = per_sample_df[per_sample_df["method"] == a].set_index("sample_id")[metric]
        db = per_sample_df[per_sample_df["method"] == b].set_index("sample_id")[metric]
        common = da.index.intersection(db.index)
        if len(common) < 10:
            continue
        va = da.loc[common].values
        vb = db.loc[common].values
        diff = va - vb
        if np.all(diff == 0):
            rows.append({"method_a": a, "method_b": b, "p_value": 1.0, "significant": False})
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                stat, p = wilcoxon(va, vb, alternative="two-sided")
            rows.append({
                "method_a": a, "method_b": b,
                "p_value": float(p),
                "significant": float(p) < 0.05,
            })
        except Exception:
            pass
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
# Aggregation helpers
# ═══════════════════════════════════════════════════════════════════════

_METRIC_COLS = [
    "dice_mean", "iou_mean", "hd95_mean", "nsd_mean", "asd_mean",
    "sens_mean", "prec_mean",
]


def _aggregate(per_sample: List[Dict[str, Any]], method: str, dataset: str, split: str) -> Dict[str, Any]:
    """Aggregate per-sample metrics into a summary row."""
    if not per_sample:
        return {}
    row: Dict[str, Any] = {"dataset": dataset, "split": split, "method": method}
    all_keys = [k for k in per_sample[0].keys() if k != "sample_id"]
    for k in all_keys:
        vals = [m.get(k, np.nan) for m in per_sample]
        if isinstance(vals[0], (int, float, np.floating, np.integer)):
            row[k] = float(np.nanmean(vals))
    return row


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(description="Comprehensive evaluation of all methods")
    ap.add_argument("--exp", required=True)
    ap.add_argument("--training", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--which", choices=["best", "last"], default="best")
    ap.add_argument("--gpus", type=str, default=None)
    ap.add_argument("--skip-live-inference", action="store_true",
                    help="Skip live model inference (only use cached OOF probs)")
    ap.add_argument("--no-uncertainty", action="store_true",
                    help="Evaluate Layer2 models without uncertainty channels")
    ap.add_argument("--allow-missing-gating-cache", action="store_true",
                    help="Do not fail when a trained gating checkpoint exists but metrics cache is missing")
    args = ap.parse_args()

    exp_cfg = load_config(args.exp)
    dataset_cfg = load_config(exp_cfg["dataset"]["config"])
    models_cfg = load_config(args.models)
    run_dir = Path(resolve_run_dir(exp_cfg))
    dataset_name = str(dataset_cfg["name"])
    results_dir = ensure_dir(
        Path(exp_cfg["output"]["results_dir"].replace("${exp_name}", exp_cfg["exp_name"]))
    )

    rows = _load_splits(dataset_cfg)
    fold = int(args.fold)
    split_candidates = ["test", f"val_fold{fold}"]
    split = next(
        (s for s in split_candidates if any(r.get("split") == s for r in rows)),
        f"val_fold{fold}",
    )
    eval_rows = [r for r in rows if r.get("split") == split]
    fit_split = f"val_fold{fold}"
    fit_rows = [r for r in rows if r.get("split") == fit_split]

    num_classes = infer_num_classes(dataset_cfg)
    in_channels = infer_image_channels(dataset_cfg)
    expert_cfgs = list_experts(models_cfg)
    all_expert_names = [expert_name(ec) for ec in expert_cfgs]
    K = len(expert_cfgs)
    add_uncertainty = not args.no_uncertainty

    # Device
    if args.gpus:
        gpu_id = int(args.gpus.split(",")[0])
        device = torch.device(f"cuda:{gpu_id}")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    summary_records: List[Dict[str, Any]] = []
    all_per_sample: List[Dict[str, Any]] = []

    # ── Phase A: Layer1 single experts ─────────────────────────────
    if not args.skip_live_inference:
        ds = SegmentationDataset2D(eval_rows, dataset_cfg, augs_cfg=None, is_train=False)
        dl = DataLoader(ds, batch_size=1, shuffle=False)

        print("=" * 60)
        print("Phase A: Layer1 single experts")
        print("=" * 60)
        for ec in expert_cfgs:
            ex = expert_name(ec)
            ckpt = _ckpt(run_dir, "layer1", fold, ex, which=args.which)
            if not ckpt.exists():
                print(f"  skip {ex} (no checkpoint)")
                continue
            model = build_expert(ec, in_channels=in_channels, num_classes=num_classes)
            state = torch.load(ckpt, map_location="cpu", weights_only=True)
            model.load_state_dict(state["model"], strict=True)
            model.to(device)

            method = f"L1_{ex}"
            ps = _eval_single_expert(model, dl, num_classes, device)
            for m in ps:
                m["method"] = method
            all_per_sample.extend(ps)
            agg = _aggregate(ps, method, dataset_name, split)
            summary_records.append(agg)
            print(f"  {method}: Dice={agg.get('dice_mean', 0):.4f}  "
                  f"HD95={agg.get('hd95_mean', float('nan')):.2f}  "
                  f"NSD={agg.get('nsd_mean', 0):.4f}")
            del model
            torch.cuda.empty_cache()

    # ── Phase B: Layer2 single experts ─────────────────────────────
    # Layer2 requires OOF probs as part of input → use Layer2OOFDataset
    l1_manifest = _resolve_manifest(exp_cfg, "oof_manifest_path", "oof/layer1/oof_manifest.jsonl")
    if l1_manifest.exists() and not args.skip_live_inference:
        try:
            from seg_moe.data.layer2_oof_dataset import Layer2OOFDataset

            extra_uncertainty_ch = (1 + num_classes) if add_uncertainty else 0
            l2_in_channels = in_channels + K * num_classes + extra_uncertainty_ch

            l2_ds = Layer2OOFDataset(
                eval_rows, dataset_cfg, l1_manifest,
                expected_num_experts=K, augs_cfg=None, is_train=False,
                add_uncertainty=add_uncertainty,
            )
            l2_dl = DataLoader(l2_ds, batch_size=1, shuffle=False)

            print("=" * 60)
            print("Phase B: Layer2 single experts")
            print("=" * 60)
            for ec in expert_cfgs:
                ex = expert_name(ec)
                ckpt = _ckpt(run_dir, "layer2", fold, ex, which=args.which)
                if not ckpt.exists():
                    print(f"  skip L2_{ex} (no checkpoint)")
                    continue
                model = build_expert(ec, in_channels=l2_in_channels, num_classes=num_classes)
                state = torch.load(ckpt, map_location="cpu", weights_only=True)
                model.load_state_dict(state["model"], strict=True)
                model.to(device)

                method = f"L2_{ex}"
                ps = _eval_single_expert(model, l2_dl, num_classes, device)
                for m in ps:
                    m["method"] = method
                all_per_sample.extend(ps)
                agg = _aggregate(ps, method, dataset_name, split)
                summary_records.append(agg)
                print(f"  {method}: Dice={agg.get('dice_mean', 0):.4f}  "
                      f"HD95={agg.get('hd95_mean', float('nan')):.2f}  "
                      f"NSD={agg.get('nsd_mean', 0):.4f}")
                del model
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"  Phase B skipped: {e}")

    # ── Phase C: Ensemble + Combiners on OOF probs ─────────────────
    # Evaluate on both L1 OOF and L2 OOF if available
    oof_configs = []

    l1_manifest = _resolve_manifest(exp_cfg, "oof_manifest_path", "oof/layer1/oof_manifest.jsonl")
    if l1_manifest.exists():
        oof_configs.append(("L1", l1_manifest))

    l2_manifest = _resolve_manifest(exp_cfg, "l2_oof_manifest_path", "oof/layer2/oof_manifest_layer2.jsonl")
    if l2_manifest.exists():
        oof_configs.append(("L2", l2_manifest))

    for layer_tag, manifest_path in oof_configs:
        oof_map = load_oof_manifest(manifest_path)

        # Check if eval samples have cached probs
        has_probs = any(
            str(s["id"]) in oof_map and get_oof_prob_path(oof_map, str(s["id"])).exists()
            for s in eval_rows
        )
        has_fit_probs = any(
            str(s["id"]) in oof_map and get_oof_prob_path(oof_map, str(s["id"])).exists()
            for s in fit_rows
        )
        if not has_probs:
            continue

        print("=" * 60)
        print(f"Phase C: Ensembles + Combiners on {layer_tag} OOF")
        print("=" * 60)

        # ── C.1: Mean ensemble (from OOF) ──
        method = f"{layer_tag}_mean"
        ps = _eval_mean_ensemble_from_oof(eval_rows, oof_map, dataset_cfg, num_classes)
        for m in ps:
            m["method"] = method
        all_per_sample.extend(ps)
        agg = _aggregate(ps, method, dataset_name, split)
        summary_records.append(agg)
        print(f"  {method}: Dice={agg.get('dice_mean', 0):.4f}  "
              f"HD95={agg.get('hd95_mean', float('nan')):.2f}  "
              f"NSD={agg.get('nsd_mean', 0):.4f}")

        # ── C.2: Majority voting ──
        mv = MajorityVotingCombiner()
        mv.fit(np.zeros((1, K, num_classes)), np.zeros(1, dtype=np.int64), num_classes)
        method = f"{layer_tag}_majority"
        ps = _eval_combiner_per_sample(mv.predict, eval_rows, oof_map, dataset_cfg, num_classes)
        for m in ps:
            m["method"] = method
        all_per_sample.extend(ps)
        agg = _aggregate(ps, method, dataset_name, split)
        summary_records.append(agg)
        print(f"  {method}: Dice={agg.get('dice_mean', 0):.4f}  "
              f"HD95={agg.get('hd95_mean', float('nan')):.2f}  "
              f"NSD={agg.get('nsd_mean', 0):.4f}")

        if not has_fit_probs:
            print(f"  [skip learned combiners — no fit probs for {layer_tag}]")
            continue

        # ── Fit combiners ──
        print(f"  Collecting {layer_tag} fit data...")
        X_fit, y_fit = _collect_oof_flat(fit_rows, oof_map, dataset_cfg)
        if X_fit.size == 0:
            print(f"  [skip — no fit data]")
            continue

        combiner_specs = [
            (f"{layer_tag}_ole", OLECombiner(mode="lsq_bounded")),
            (f"{layer_tag}_dt", DecisionTemplateCombiner()),
            (f"{layer_tag}_we_clpso", WECLPSOCombiner(n_particles=30, iters=100, seed=42)),
        ]

        for method, comb in combiner_specs:
            print(f"  Fitting {method}...")
            comb.fit(X_fit, y_fit, num_classes=num_classes)
            ps = _eval_combiner_per_sample(
                comb.predict, eval_rows, oof_map, dataset_cfg, num_classes,
            )
            for m in ps:
                m["method"] = method
            all_per_sample.extend(ps)
            agg = _aggregate(ps, method, dataset_name, split)
            summary_records.append(agg)
            print(f"  {method}: Dice={agg.get('dice_mean', 0):.4f}  "
                  f"HD95={agg.get('hd95_mean', float('nan')):.2f}  "
                  f"NSD={agg.get('nsd_mean', 0):.4f}")

    # ── Phase D: Gating fusion ─────────────────────────────────────
    gating_data = _eval_gating_from_cache(run_dir, fold, split)
    if gating_data is not None:
        print("=" * 60)
        print("Phase D: Gating fusion (cached)")
        print("=" * 60)
        method = "gating"
        gating_mean_dice = gating_data.get("mean_dice", 0.0)
        # Add per-sample from gating cache (if available)
        gating_ps = gating_data.get("per_sample", [])
        for m in gating_ps:
            m["method"] = method
        all_per_sample.extend(gating_ps)
        agg = {"dataset": dataset_name, "split": split, "method": method, "dice_mean": gating_mean_dice}
        summary_records.append(agg)
        print(f"  gating: Dice={gating_mean_dice:.4f}")
    else:
        gating_ckpt = run_dir / "checkpoints" / "gating" / f"fold{fold}" / "best.pt"
        if gating_ckpt.exists() and not args.allow_missing_gating_cache:
            raise FileNotFoundError(
                "Gating checkpoint exists but cached inference results are missing. "
                "Run scripts/inference/gating_inference.py before eval_methods.py, "
                "or pass --allow-missing-gating-cache to skip gating."
            )
        print("\n[Phase D skipped: no gating results cached]")

    # ── Phase E: Statistical significance ──────────────────────────
    print("=" * 60)
    print("Phase E: Statistical significance (Wilcoxon)")
    print("=" * 60)

    if all_per_sample:
        ps_df = pd.DataFrame(all_per_sample)
        sig_df = _wilcoxon_pairwise(ps_df, metric="dice_mean")
        if not sig_df.empty:
            sig_path = results_dir / f"significance_{dataset_name}_fold{fold}_{split}.csv"
            sig_df.to_csv(sig_path, index=False)
            print(f"  Wrote {sig_path} ({len(sig_df)} pairs tested)")
            # Print significant pairs
            sig_pairs = sig_df[sig_df["significant"]]
            if not sig_pairs.empty:
                for _, r in sig_pairs.iterrows():
                    print(f"    {r['method_a']} vs {r['method_b']}: p={r['p_value']:.4e} *")
        else:
            print("  No method pairs with enough samples for testing")

    # ── Save outputs ───────────────────────────────────────────────
    # Per-sample metrics (for statistical tests & detailed analysis)
    if all_per_sample:
        ps_df = pd.DataFrame(all_per_sample)
        ps_path = results_dir / f"metrics_per_sample_{dataset_name}_fold{fold}_{split}.csv"
        ps_df.to_csv(ps_path, index=False)
        print(f"\nWrote per-sample: {ps_path}")

    # Summary metrics (backward compatible with old format)
    if summary_records:
        sum_df = pd.DataFrame(summary_records)
        sum_path = results_dir / f"metrics_{dataset_name}_fold{fold}_{split}.csv"
        sum_df.to_csv(sum_path, index=False)
        print(f"Wrote summary:    {sum_path}")

        # Also write to tables dir
        tables_dir = ensure_dir(Path(
            exp_cfg.get("output", {}).get("tables_dir",
                f"runs/{exp_cfg['exp_name']}/tables").replace("${exp_name}", exp_cfg["exp_name"])
        ))
        sum_df.to_csv(tables_dir / "metrics.csv", index=False)
        ps_df.to_csv(tables_dir / "metrics_per_sample.csv", index=False)

    # Combiner weights
    weights_out: Dict[str, Any] = {
        "dataset": dataset_name, "fold": fold,
        "experts": all_expert_names, "num_classes": num_classes,
    }
    # Extract weights from fitted combiners (if any in scope)
    save_json(results_dir / f"{dataset_name}_fold{fold}_weights.json", weights_out)

    # ── Final summary table ────────────────────────────────────────
    if summary_records:
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        cols = ["method", "dice_mean", "hd95_mean", "nsd_mean", "sens_mean", "prec_mean"]
        display_cols = [c for c in cols if c in sum_df.columns]
        print(sum_df[display_cols].to_string(index=False, float_format="%.4f"))


if __name__ == "__main__":
    main()

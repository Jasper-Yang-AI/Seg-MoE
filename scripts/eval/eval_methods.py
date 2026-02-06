"""
Evaluate single experts + mean ensemble + OLE/DT/WE-CLPSO combiners.

Usage:
    python scripts/eval/eval_methods.py \
        --exp configs/2d/exp/exp_msd_task03_liver.yaml \
        --training configs/2d/training.yaml \
        --models configs/2d/models.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from seg_moe.combiners.decision_template import DecisionTemplateCombiner
from seg_moe.combiners.ole import OLECombiner
from seg_moe.combiners.we_clpso import WECLPSOCombiner
from seg_moe.data.dataset_2d import SegmentationDataset2D
from seg_moe.data.indexing import infer_image_channels, infer_num_classes
from seg_moe.data.oof import load_oof_manifest
from seg_moe.evaluation.metrics_2d import compute_segmentation_metrics_batch
from seg_moe.models.factory_2d import build_expert, expert_name, list_experts
from seg_moe.utils.config import load_config, resolve_run_dir
from seg_moe.utils.io import ensure_dir, load_jsonl, save_json


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


def _eval_single(model, dl, num_classes, device):
    model.eval()
    metrics = []
    with torch.no_grad():
        for x, y, meta in tqdm(dl, desc="eval"):
            logits = model(x.to(device))
            probs = torch.softmax(logits, dim=1)
            spacing = meta.get("spacing_yx")
            spacing_yx = None
            if spacing is not None:
                s0 = spacing[0]
                if isinstance(s0, (list, tuple)) and len(s0) == 2:
                    spacing_yx = (float(s0[0]), float(s0[1]))
            metrics.append(compute_segmentation_metrics_batch(probs, y.to(device), num_classes=num_classes, spacing_yx=spacing_yx))
    return {k: float(np.mean([m.get(k, 0.0) for m in metrics])) for k in metrics[0].keys()} if metrics else {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--training", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--which", choices=["best", "last"], default="best")
    args = ap.parse_args()

    exp_cfg = load_config(args.exp)
    dataset_cfg = load_config(exp_cfg["dataset"]["config"])
    models_cfg = load_config(args.models)
    run_dir = resolve_run_dir(exp_cfg)
    results_dir = ensure_dir(Path(exp_cfg["output"]["results_dir"].replace("${exp_name}", exp_cfg["exp_name"])))

    rows = _load_splits(dataset_cfg)
    fold = int(args.fold)
    split_candidates = ["test", f"val_fold{fold}"]
    split = next((s for s in split_candidates if any(r.get("split") == s for r in rows)), f"val_fold{fold}")
    eval_rows = [r for r in rows if r.get("split") == split]

    num_classes = infer_num_classes(dataset_cfg)
    in_channels = infer_image_channels(dataset_cfg)
    ds = SegmentationDataset2D(eval_rows, dataset_cfg, augs_cfg=None, is_train=False)
    dl = DataLoader(ds, batch_size=1, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    expert_cfgs = list_experts(models_cfg)
    all_expert_names = [expert_name(ec) for ec in expert_cfgs]
    records = []

    # ---- Single experts (layer1) ----
    for ec in expert_cfgs:
        ex = expert_name(ec)
        ckpt = _ckpt(run_dir, "layer1", fold, ex, which=args.which)
        if not ckpt.exists():
            continue
        model = build_expert(ec, in_channels=in_channels, num_classes=num_classes)
        state = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(state["model"], strict=True)
        model.to(device)
        m = _eval_single(model, dl, num_classes, device)
        records.append({"dataset": dataset_cfg["name"], "split": split, "method": f"single_{ex}", **m})

    # ---- Mean ensemble ----
    expert_models = []
    for ec in expert_cfgs:
        ckpt = _ckpt(run_dir, "layer1", fold, expert_name(ec), which=args.which)
        if not ckpt.exists():
            break
        model = build_expert(ec, in_channels=in_channels, num_classes=num_classes)
        state = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(state["model"], strict=True)
        model.to(device).eval()
        expert_models.append(model)

    if len(expert_models) == len(expert_cfgs):
        metrics = []
        with torch.no_grad():
            for x, y, _ in tqdm(dl, desc="eval mean_ensemble"):
                x_dev = x.to(device)
                probs_sum = sum(torch.softmax(m(x_dev), dim=1) for m in expert_models)
                metrics.append(compute_segmentation_metrics_batch(
                    probs_sum / len(expert_models), y.to(device), num_classes=num_classes))
        out = {k: float(np.mean([m.get(k, 0.0) for m in metrics])) for k in metrics[0].keys()} if metrics else {}
        records.append({"dataset": dataset_cfg["name"], "split": split, "method": "mean_ensemble", **out})

    # ---- Combiners (require cached probs) ----
    cache_root = Path(exp_cfg["layering"]["cache_root"].replace("${exp_name}", exp_cfg["exp_name"]))
    oof_cache_dir = Path(str(exp_cfg.get("layering", {}).get("oof_cache_dir", cache_root / "oof" / "layer1")).replace("${exp_name}", exp_cfg["exp_name"]))
    oof_manifest_path = Path(str(exp_cfg.get("layering", {}).get("oof_manifest_path", oof_cache_dir / "oof_manifest.jsonl")).replace("${exp_name}", exp_cfg["exp_name"]))
    oof_map = load_oof_manifest(oof_manifest_path) if oof_manifest_path.exists() else None

    def _probs_path(sid, sn):
        if oof_map is not None and sid in oof_map:
            return oof_map[sid].prob_path
        return cache_root / "layer1_probs" / dataset_cfg["name"] / sn / f"{sid}.npz"

    weights_out = {"dataset": dataset_cfg["name"], "fold": fold, "experts": all_expert_names,
                   "num_classes": num_classes, "layer1": {}}

    fit_split = f"val_fold{fold}"
    fit_rows = [r for r in rows if r.get("split") == fit_split]
    can_fit = (fit_rows and eval_rows
               and _probs_path(str(fit_rows[0]["id"]), fit_split).exists()
               and _probs_path(str(eval_rows[0]["id"]), split).exists())

    if can_fit:
        fit_ds = SegmentationDataset2D(fit_rows, dataset_cfg, augs_cfg=None, is_train=False)
        fit_dl = DataLoader(fit_ds, batch_size=1, shuffle=False)

        X_list, y_list = [], []
        for _, mask, meta in tqdm(fit_dl, desc="collect combiner fit"):
            sid = meta["id"][0]
            probs = np.load(_probs_path(str(sid), fit_split))["probs"].astype(np.float32)
            H, W = probs.shape[-2:]
            X_list.append(probs.transpose(2, 3, 0, 1).reshape(H * W, probs.shape[0], probs.shape[1]))
            y_list.append(mask.numpy().reshape(-1))
        X, y = np.concatenate(X_list), np.concatenate(y_list)

        ole = OLECombiner(mode="lsq_bounded"); ole.fit(X, y, num_classes=num_classes)
        dt = DecisionTemplateCombiner();       dt.fit(X, y, num_classes=num_classes)
        we = WECLPSOCombiner(n_particles=10, iters=30); we.fit(X, y, num_classes=num_classes)

        def _eval_combiner(name, pred_fn):
            ms = []
            for _, mask, meta in tqdm(dl, desc=f"eval {name}"):
                sid = meta["id"][0]
                probs = np.load(_probs_path(str(sid), split))["probs"].astype(np.float32)
                H, W = probs.shape[-2:]
                pf = probs.transpose(2, 3, 0, 1).reshape(H * W, probs.shape[0], probs.shape[1])
                out = pred_fn(pf)
                pred = (out.reshape(H, W) if out.ndim == 1 else np.argmax(out, -1).reshape(H, W))
                pred_1h = np.eye(num_classes, dtype=np.float32)[pred].transpose(2, 0, 1)[None]
                ms.append(compute_segmentation_metrics_batch(torch.from_numpy(pred_1h), mask, num_classes=num_classes))
            return {k: float(np.mean([m.get(k, 0.0) for m in ms])) for k in ms[0].keys()} if ms else {}

        for name, comb in [("ole", ole), ("dt", dt), ("we_clpso", we)]:
            m = _eval_combiner(name, lambda pf, c=comb: c.predict(pf))
            records.append({"dataset": dataset_cfg["name"], "split": split, "method": name, **m})

        weights_out["layer1"]["ole"] = ole.weights.w.tolist() if ole.weights else None
        weights_out["layer1"]["we_clpso"] = we.state.w.tolist() if we.state else None

    save_json(results_dir / f"{dataset_cfg['name']}_fold{fold}_weights.json", weights_out)

    df = pd.DataFrame.from_records(records)
    out_csv = results_dir / f"metrics_{dataset_cfg['name']}_fold{fold}_{split}.csv"
    df.to_csv(out_csv, index=False)
    print(f"Wrote: {out_csv}")

    tables_dir = ensure_dir(Path(exp_cfg.get("output", {}).get("tables_dir",
        f"runs/{exp_cfg['exp_name']}/tables").replace("${exp_name}", exp_cfg["exp_name"])))
    df.to_csv(tables_dir / "metrics.csv", index=False)
    print(f"Wrote: {tables_dir / 'metrics.csv'}")


if __name__ == "__main__":
    main()

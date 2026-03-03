"""
Evaluate 3D Seg-MoE pipeline on the held-out test set.

Compares four methods:
  1. single_best   – single best expert (highest val dice from L1 ckpts)
  2. mean_ensemble – average of all K Layer2 expert probs
  3. gating        – PatchConvGate3D-weighted fusion
  4. individual    – per-expert breakdown (optional with --all-experts)

Outputs a results CSV + JSON to runs/<exp_name>/results/.

Usage:
    python scripts/eval/eval_3d.py \\
        --exp    configs/3d/exp/exp_prostate_local_3d.yaml \\
        --models configs/3d/models_3d.yaml \\
        --gating-config configs/3d/gating_3d.yaml \\
        --which best
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from seg_moe.data.layer2_oof_dataset_3d import Layer2OOFDataset3D
from seg_moe.evaluation.metrics_3d import compute_segmentation_metrics_3d
from seg_moe.gating.patch_gating_3d import PatchConvGate3D, PatchGatingConfig3D, fuse_volume_sliding_window
from seg_moe.models.factory_3d import (
    build_expert_3d,
    expert_name_3d,
    list_experts_3d,
    transfer_layer1_to_layer2_3d,
)
from seg_moe.utils.config import load_config, resolve_run_dir
from seg_moe.utils.io import ensure_dir, load_jsonl


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_splits(dataset_cfg: dict) -> list[dict]:
    path = Path(dataset_cfg["paths"]["splits_dir"]) / "splits_train5fold_testfixed.jsonl"
    return load_jsonl(path)


def _ckpt(run_dir: Path, layer: str, fold: int, ex_name: str, which: str) -> Path:
    return run_dir / "checkpoints" / layer / f"fold{fold}" / ex_name / f"{which}.pt"


def _sliding_window(model, x, roi_size, sw_batch, overlap, device):
    try:
        from monai.inferers import sliding_window_inference
        return sliding_window_inference(
            x, roi_size, sw_batch, model,
            overlap=overlap, mode="gaussian", device=device
        )
    except ImportError:
        return model(x)


def _load_l2_model(ec, run_dir, fold, in_ch, num_classes, in_ch_l1, which, device):
    name  = expert_name_3d(ec)
    l2_ck = _ckpt(run_dir, "layer2", fold, name, which)
    l1_ck = _ckpt(run_dir, "layer1", fold, name, which)
    if l2_ck.exists():
        m = build_expert_3d(ec, in_channels=in_ch, num_classes=num_classes)
        st = torch.load(l2_ck, map_location="cpu", weights_only=True)
        m.load_state_dict({k.removeprefix("module."): v for k, v in st["model"].items()},
                          strict=False)
    elif l1_ck.exists():
        m_l1 = build_expert_3d(ec, in_channels=in_ch_l1, num_classes=num_classes)
        st = torch.load(l1_ck, map_location="cpu", weights_only=True)
        m_l1.load_state_dict({k.removeprefix("module."): v for k, v in st["model"].items()},
                             strict=False)
        m = build_expert_3d(ec, in_channels=in_ch, num_classes=num_classes)
        transfer_layer1_to_layer2_3d(
            m_l1, m,
            base_in_channels=in_ch_l1,
            extra_in_channels=in_ch - in_ch_l1,
        )
    else:
        raise FileNotFoundError(f"No ckpt for {name} fold{fold}")
    return m.eval().to(device)


def _load_gating_model(run_dir, fold, gate_cfg, expert_cfgs, num_classes, device):
    K = len(expert_cfgs)
    M = num_classes
    ps = tuple(int(x) for x in gate_cfg["gating"]["patch_size"])
    g = gate_cfg["gating"]
    model_cfg = PatchGatingConfig3D(
        num_experts=K,
        num_classes=M,
        patch_size=ps,
        stride=tuple(int(x) for x in g.get("stride", [16, 16, 8])),
        hidden_dim=int(g.get("hidden_dim", 64)),
        dropout=float(g.get("dropout", 0.1)),
        per_class=bool(g.get("per_class", False)),
        use_residual_head=bool(g.get("use_residual_head", True)),
        temperature_start=float(g.get("temperature_start", 2.0)),
        temperature_end=float(g.get("temperature_end", 0.5)),
        load_balance_weight=float(g.get("load_balance_weight", 0.01)),
        spatial_smooth_weight=float(g.get("spatial_smooth_weight", 0.0)),
        blend_mode=str(g.get("blend_mode", "gaussian")),
    )
    model = PatchConvGate3D(model_cfg).to(device)
    ckpt_path = run_dir / "checkpoints" / "gating" / f"fold{fold}" / "best.pt"
    if ckpt_path.exists():
        st = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict({k.removeprefix("module."): v for k, v in st["model"].items()},
                              strict=False)
        print(f"  Loaded gating from {ckpt_path}")
    else:
        print(f"  WARNING: Gating checkpoint not found at {ckpt_path}. Using random weights.")
    return model.eval()


def _aggregate(per_vol: list[dict]) -> dict:
    if not per_vol:
        return {}
    keys = [k for k in per_vol[0] if k not in ("sample_id",)]
    agg: dict = {}
    for k in keys:
        vals = [v[k] for v in per_vol if isinstance(v.get(k), (int, float)) and not np.isnan(v[k])]
        agg[k + "_mean"] = float(np.mean(vals)) if vals else float("nan")
        agg[k + "_std"]  = float(np.std(vals))  if vals else float("nan")
    return agg


def _rows_to_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _metrics_to_record(metrics: dict, sample_id: str, class_names: list[str], num_classes: int) -> dict:
    rec = {"sample_id": sample_id}
    for ci in range(1, num_classes):
        cname = class_names[ci - 1] if (ci - 1) < len(class_names) else f"cls_{ci}"
        for met in ("dice", "iou", "hd95", "nsd", "vs"):
            k = f"{met}_c{ci}"
            if k in metrics:
                rec[f"{cname}_{met}"] = metrics[k]
    for mk in ("dice_mean", "iou_mean", "hd95_mean", "nsd_mean", "vs_mean"):
        if mk in metrics:
            rec[mk] = metrics[mk]
    return rec


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate 3D Seg-MoE on test set")
    ap.add_argument("--exp",           required=True)
    ap.add_argument("--models",        required=True)
    ap.add_argument("--gating-config", default="configs/3d/gating_3d.yaml")
    ap.add_argument("--which",         choices=["best", "last"], default="best")
    ap.add_argument("--fold",          type=int, default=0,
                    help="Which training fold's checkpoints to use (default: 0)")
    ap.add_argument("--all-experts",   action="store_true",
                    help="Also evaluate each expert individually")
    ap.add_argument("--no-gating",     action="store_true",
                    help="Skip gating evaluation (e.g. gating not yet trained)")
    args = ap.parse_args()

    exp_cfg     = load_config(args.exp)
    dataset_cfg = load_config(exp_cfg["dataset"]["config"])
    models_cfg  = load_config(args.models)
    gate_cfg    = load_config(args.gating_config)
    run_dir     = Path(resolve_run_dir(exp_cfg))
    results_dir = run_dir / "results"
    ensure_dir(results_dir)

    exp_name       = exp_cfg["exp_name"]
    layering       = exp_cfg["layering"]
    l1_manifest_p  = Path(str(layering["oof_manifest_path"]).replace("${exp_name}", exp_name))

    rows        = _load_splits(dataset_cfg)
    test_rows   = [r for r in rows if r.get("split") == "test"]
    if not test_rows:
        print("WARNING: no 'test' rows in splits — using all data for smoke eval.")
        test_rows = rows[:5]
    print(f"\n[eval_3d] Test volumes: {len(test_rows)}")

    num_classes = int(dataset_cfg["task"]["num_classes"])
    in_ch_l1    = int(dataset_cfg["input"].get("image_channels", 1))
    expert_cfgs = list_experts_3d(models_cfg)
    K           = len(expert_cfgs)
    M           = num_classes
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sw_cfg   = dataset_cfg.get("input", {})
    roi_size = tuple(int(s) for s in sw_cfg.get("roi_size", [128, 128, 64]))
    sw_batch = 2

    # ----- Datasets -----
    test_ds_l1 = None  # lazy: only loaded for L1 single_best
    test_ds_l2 = Layer2OOFDataset3D(
        samples=test_rows,
        dataset_cfg=dataset_cfg,
        oof_manifest_path=str(l1_manifest_p),
        is_train=False,
    )
    in_ch_l2 = test_ds_l2.in_channels

    fold = args.fold

    # ----- Load Layer2 models -----
    print(f"\n[eval_3d] Loading Layer2 models (fold={fold}) ...")
    l2_models: List[torch.nn.Module] = []
    for ec in expert_cfgs:
        m = _load_l2_model(ec, run_dir, fold, in_ch_l2, num_classes, in_ch_l1, args.which, device)
        l2_models.append(m)
        print(f"  {expert_name_3d(ec)} loaded")

    # ----- Optionally load gating -----
    gate_model: Optional[PatchConvGate3D] = None
    if not args.no_gating:
        gate_model = _load_gating_model(run_dir, fold, gate_cfg, expert_cfgs, num_classes, device)
        ps = tuple(int(x) for x in gate_cfg["gating"]["patch_size"])
        st = tuple(int(x) for x in gate_cfg["gating"]["stride"])

    # ----- Evaluation loop -----
    class_names = dataset_cfg["task"].get("class_names", [f"cls_{c}" for c in range(1, M)])

    methods = ["mean_ensemble"]
    if not args.no_gating:
        methods.append("gating")
    if args.all_experts:
        for ec in expert_cfgs:
            methods.append(f"expert_{expert_name_3d(ec)}")

    per_vol_records: dict[str, list[dict]] = {m: [] for m in methods}
    per_vol_records["mean_ensemble"] = []

    for si, sample in enumerate(tqdm(test_rows, desc="testing")):
        sid = str(sample["id"])
        img_t, mask_t, _ = test_ds_l2[si]
        img_t = img_t.unsqueeze(0).to(device)      # [1, C2, D, H, W]
        mask_np = mask_t.numpy().astype(np.int64)  # [D, H, W]

        # Collect L2 per-expert probs
        expert_probs: List[np.ndarray] = []
        for ei, m in enumerate(l2_models):
            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=device.type == "cuda", dtype=torch.bfloat16):
                    raw = _sliding_window(m, img_t, roi_size, sw_batch, 0.5, device)
            probs = torch.softmax(raw.squeeze(0), dim=0).float().cpu().numpy()
            expert_probs.append(probs)

        # ---- Mean ensemble ----
        mean_probs = np.mean(np.stack(expert_probs, axis=0), axis=0)  # [M, D, H, W]
        pred_mean  = mean_probs.argmax(axis=0).astype(np.int64)
        m_metrics = compute_segmentation_metrics_3d(pred_mean, mask_np, num_classes)
        per_vol_records["mean_ensemble"].append(
            _metrics_to_record(m_metrics, sid, class_names, num_classes)
        )

        # ---- Gating ----
        if gate_model is not None:
            logits_stack = np.stack(
                [np.log(p.clip(1e-7) / p.clip(1e-7).sum(axis=0, keepdims=True).clip(1e-7))
                 for p in expert_probs], axis=0
            )  # [K, M, D, H, W]
            fused_logits = fuse_volume_sliding_window(
                gate_model,
                torch.from_numpy(logits_stack),
                patch_size=ps,
                stride=st,
                num_classes=M,
                num_experts=K,
                device=device,
            )  # [M, D, H, W]
            pred_gate = fused_logits.argmax(dim=0).numpy().astype(np.int64)
            g_metrics = compute_segmentation_metrics_3d(pred_gate, mask_np, num_classes)
            per_vol_records["gating"].append(
                _metrics_to_record(g_metrics, sid, class_names, num_classes)
            )

        # ---- Per-expert ----
        if args.all_experts:
            for ei, ec in enumerate(expert_cfgs):
                mkey = f"expert_{expert_name_3d(ec)}"
                pred_e = expert_probs[ei].argmax(axis=0).astype(np.int64)
                e_mets = compute_segmentation_metrics_3d(pred_e, mask_np, num_classes)
                per_vol_records[mkey].append(
                    _metrics_to_record(e_mets, sid, class_names, num_classes)
                )

    # ---- Save results ----
    summary: dict[str, Any] = {}
    for method, records in per_vol_records.items():
        if not records:
            continue
        csv_path = results_dir / f"{method}_per_volume.csv"
        _rows_to_csv(records, csv_path)
        agg = _aggregate(records)
        summary[method] = agg
        print(f"\n--- {method} ---")
        for k, v in sorted(agg.items()):
            if k.endswith("_mean"):
                print(f"  {k}: {v:.4f}")

    summary_path = results_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[eval_3d] Results saved to {results_dir}")


if __name__ == "__main__":
    main()

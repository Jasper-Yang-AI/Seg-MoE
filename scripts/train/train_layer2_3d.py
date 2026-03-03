"""
3D Layer2 expert training script.

流程 (与 train_layer2.py 完全对应, 但适配 3D 体积分割):
  Layer1 OOF → concat input [vol + L1_probs + uncertainty] → train Layer2 experts

Input channels:
    C_total = image_channels + K * num_classes + (1 + num_classes)
            = 3             + 3 * 4            + (1 + 4)
            = 3 + 12 + 5 = 20  (prostate default)

Usage:
    python scripts/train/train_layer2_3d.py \\
        --exp      configs/3d/exp/exp_prostate_local_3d.yaml \\
        --training configs/3d/training_layer2_3d.yaml \\
        --models   configs/3d/models_3d.yaml \\
        --augs     configs/3d/augs_3d.yaml \\
        --fold 0   --gpus 0
"""
from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from seg_moe.data.layer2_oof_dataset_3d import Layer2OOFDataset3D
from seg_moe.evaluation.metrics_3d import compute_dice_batch_3d
from seg_moe.models.factory_3d import (
    build_expert_3d, expert_name_3d, list_experts_3d, transfer_layer1_to_layer2_3d,
)
from seg_moe.training.losses import ce_plus_dice
from seg_moe.utils.config import load_config, resolve_run_dir
from seg_moe.utils.io import ensure_dir, load_jsonl
from seg_moe.utils.seed import seed_everything

# Reuse training loop from layer1 script
from scripts.train.train_layer1_3d import (
    _parse_gpu_ids, _wrap_dp, _unwrap,
    _load_splits, _sliding_window_inference, train_model_3d,
)

_IS_WINDOWS = platform.system() == "Windows"
_DEFAULT_WORKERS = 2 if _IS_WINDOWS else 4


def main() -> None:
    ap = argparse.ArgumentParser(description="Train 3D Layer2 experts with OOF probs")
    ap.add_argument("--exp",        required=True)
    ap.add_argument("--training",   required=True)
    ap.add_argument("--models",     required=True)
    ap.add_argument("--augs",       required=True)
    ap.add_argument("--fold",       type=int, default=0)
    ap.add_argument("--gpus",       type=str, default=None)
    ap.add_argument("--which",      choices=["best", "last"], default="best")
    ap.add_argument("--resume",     default="none")
    ap.add_argument("--skip-if-done", action="store_true")
    ap.add_argument("--no-pretrain",  action="store_true",
                    help="Skip Layer1 → Layer2 weight transfer")
    ap.add_argument("--no-uncertainty", action="store_true",
                    help="Disable entropy + disagreement channels")
    ap.add_argument("--epochs",     type=int, default=None)
    ap.add_argument("--smoke",      action="store_true")
    ap.add_argument("--num-workers", type=int, default=None)
    args = ap.parse_args()

    exp_cfg      = load_config(args.exp)
    training_cfg = load_config(args.training)
    models_cfg   = load_config(args.models)
    augs_cfg     = load_config(args.augs)
    dataset_cfg  = load_config(exp_cfg["dataset"]["config"])

    seed_everything(int(exp_cfg.get("seed", 42)))
    if args.epochs is not None:
        training_cfg["epochs"] = args.epochs
    if args.num_workers is not None:
        training_cfg.setdefault("dataloader", {})["num_workers"] = args.num_workers
    if args.smoke:
        training_cfg["epochs"] = 2
        training_cfg["batch_size"] = 1

    run_dir = Path(resolve_run_dir(exp_cfg))
    ensure_dir(run_dir)
    device, gpu_ids = _parse_gpu_ids(args.gpus)

    num_classes = int(dataset_cfg["task"]["num_classes"])
    base_in     = int(dataset_cfg["input"].get("image_channels", 1))
    fold        = int(args.fold)
    expert_cfgs = list_experts_3d(models_cfg)
    K           = len(expert_cfgs)

    # OOF manifest path
    cache_root = Path(
        exp_cfg["layering"]["cache_root"].replace("${exp_name}", exp_cfg["exp_name"])
    )
    oof_manifest_path = Path(
        str(exp_cfg["layering"]["oof_manifest_path"]).replace("${exp_name}", exp_cfg["exp_name"])
    )
    if not oof_manifest_path.exists():
        raise FileNotFoundError(
            f"Layer1 OOF manifest not found: {oof_manifest_path}\n"
            "Run scripts/inference/generate_layer1_oof_3d.py first."
        )

    add_unc = not args.no_uncertainty
    extra_unc_ch = (1 + num_classes) if add_unc else 0
    in_channels  = base_in + K * num_classes + extra_unc_ch
    unc_str = f" + uncertainty({extra_unc_ch}ch)" if add_unc else ""
    print(f"3D Layer2 | fold={fold} | in_ch={in_channels} (base={base_in} + {K}×{num_classes}{unc_str})")

    rows   = _load_splits(dataset_cfg)
    nw     = int(training_cfg.get("dataloader", {}).get("num_workers", _DEFAULT_WORKERS))
    pm     = bool(training_cfg.get("dataloader", {}).get("pin_memory", True))
    bs     = int(training_cfg.get("batch_size", 2))
    epochs = int(training_cfg.get("epochs", 150))

    if args.smoke:
        # Smoke: reuse dummy loader
        from torch.utils.data import TensorDataset
        roi = training_cfg.get("sliding_window", {}).get("roi_size", [128, 128, 64])
        D, H, W = roi
        imgs  = torch.randn(4, in_channels, D, H, W)
        masks = torch.randint(0, num_classes, (4, D, H, W))
        train_loader = DataLoader(
            TensorDataset(imgs, masks),
            batch_size=1,
            collate_fn=lambda b: (torch.stack([x[0] for x in b]),
                                  torch.stack([x[1] for x in b]), {})
        )
        val_loader = train_loader
    else:
        train_ds = Layer2OOFDataset3D(
            [r for r in rows if r.get("split") == f"train_fold{fold}"],
            dataset_cfg, oof_manifest_path,
            expected_num_experts=K, augs_cfg=augs_cfg, is_train=True,
            add_uncertainty=add_unc)
        val_ds = Layer2OOFDataset3D(
            [r for r in rows if r.get("split") == f"val_fold{fold}"],
            dataset_cfg, oof_manifest_path,
            expected_num_experts=K, augs_cfg=augs_cfg, is_train=False,
            add_uncertainty=add_unc)

        print(f"  Train: {len(train_ds)}  Val: {len(val_ds)}")
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                                  num_workers=nw, pin_memory=pm, drop_last=True)
        val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False,
                                  num_workers=nw, pin_memory=pm)

    expert_overrides = training_cfg.get("expert_overrides", {}) or {}

    for ec in expert_cfgs:
        name = expert_name_3d(ec)
        tag  = f"layer2/fold{fold}/{name}"
        ckpt_dir  = run_dir / "checkpoints" / "layer2" / f"fold{fold}" / name
        best_ckpt = ckpt_dir / "best.pt"
        last_ckpt = ckpt_dir / "last.pt"

        if args.skip_if_done and best_ckpt.exists():
            print(f"  Skip: {best_ckpt}")
            continue

        resume_from: str | None = None
        if args.resume.lower() == "last" and last_ckpt.exists():
            resume_from = str(last_ckpt)
        elif args.resume.lower() == "best" and best_ckpt.exists():
            resume_from = str(best_ckpt)
        elif args.resume.lower() not in ("none", ""):
            resume_from = args.resume

        # Build Layer2 model (extra input channels)
        model = build_expert_3d(ec, in_channels=in_channels, num_classes=num_classes)

        # ── B1: Transfer Layer1 weights ──
        if not args.no_pretrain and resume_from is None:
            l1_ckpt = run_dir / "checkpoints" / "layer1" / f"fold{fold}" / name / f"{args.which}.pt"
            if l1_ckpt.exists():
                l1_model = build_expert_3d(ec, in_channels=base_in, num_classes=num_classes)
                state = torch.load(l1_ckpt, map_location="cpu", weights_only=True)
                l1_model.load_state_dict(
                    {k.removeprefix("module."): v for k, v in state["model"].items()}, strict=False
                )
                transfer_layer1_to_layer2_3d(
                    l1_model, model,
                    base_in_channels=base_in,
                    extra_in_channels=in_channels - base_in,
                )
                del l1_model
                print(f"  Transferred Layer1→Layer2 weights for {name}")
            else:
                print(f"  [Layer2] Layer1 ckpt not found: {l1_ckpt}, training from scratch")

        model = _wrap_dp(model, gpu_ids)

        # ── B4: Per-expert overrides ──
        tcfg = {**training_cfg, "epochs": epochs}
        if name in expert_overrides:
            ovr = expert_overrides[name]
            for k, v in ovr.items():
                if isinstance(v, dict) and isinstance(tcfg.get(k), dict):
                    tcfg[k] = {**tcfg[k], **v}
                else:
                    tcfg[k] = v
            print(f"  Applied expert overrides for {name}: {list(ovr.keys())}")

        print(f"\n  Training 3D Layer2: {tag}")
        result = train_model_3d(model, train_loader, val_loader, num_classes,
                                tcfg, run_dir, tag, device, resume_from=resume_from)
        print(f"  Done {name}: best_dice={result['best_metric']:.4f}")


if __name__ == "__main__":
    main()

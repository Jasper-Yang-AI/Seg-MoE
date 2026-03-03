#!/usr/bin/env python
"""
Import SwinUNETR 3D official-training weights into Seg-MoE format.

Usage:
    python scripts/monai/import_swinunetr_weights_3d.py \\
        --source runs/swinunetr_official_3d_prostate/fold0/best_model.pt \\
        --exp  configs/3d/exp/exp_prostate_local_3d.yaml \\
        --models configs/3d/models_3d.yaml --fold 0

    # 5-fold batch (PowerShell)
    foreach ($fold in 0..4) {
        python scripts/monai/import_swinunetr_weights_3d.py `
            --source runs/swinunetr_official_3d_prostate_local_3d/fold$fold/best_model.pt `
            --exp  configs/3d/exp/exp_prostate_local_3d.yaml `
            --models configs/3d/models_3d.yaml --fold $fold `
            --expert-name swinunetr-3d
    }
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import torch

from seg_moe.models.factory_3d import build_expert_3d, expert_name_3d, list_experts_3d
from seg_moe.utils.config import load_config, resolve_run_dir


def _resolve_swin_expert(experts, requested_name):
    swin_types = {"swin_unetr", "swinunetr"}
    candidates = [e for e in experts if e.get("type", "").lower() in swin_types]
    if requested_name:
        for e in candidates:
            if expert_name_3d(e) == requested_name:
                return e, requested_name
        raise ValueError(f"Expert '{requested_name}' not found among SwinUNETR 3D: {[expert_name_3d(e) for e in candidates]}")
    if len(candidates) == 1:
        return candidates[0], expert_name_3d(candidates[0])
    if not candidates:
        raise ValueError("No SwinUNETR expert found in experts_3d")
    raise ValueError(f"Multiple SwinUNETR: specify --expert-name. Candidates: {[expert_name_3d(e) for e in candidates]}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Import SwinUNETR 3D official weights into Seg-MoE")
    ap.add_argument("--source", required=True)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--layer", default="layer1", choices=["layer1", "layer2"])
    ap.add_argument("--expert-name", default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dataset-config", default=None)
    args = ap.parse_args()

    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")

    exp_cfg = load_config(args.exp)
    models_cfg = load_config(args.models)
    dataset_cfg = load_config(args.dataset_config or exp_cfg["dataset"]["config"])
    run_dir = Path(resolve_run_dir(exp_cfg))

    experts = list_experts_3d(models_cfg)
    swin_ec, swin_name = _resolve_swin_expert(experts, args.expert_name)

    num_classes = int(dataset_cfg["task"]["num_classes"])
    in_channels = int(dataset_cfg["input"].get("image_channels", 3))

    print()
    print("=" * 60)
    print("Importing SwinUNETR 3D weights into Seg-MoE")
    print("=" * 60)
    print(f"  Source  : {source}")
    print(f"  Expert  : {swin_name}")
    print(f"  Fold    : {args.fold}")

    ckpt = torch.load(source, map_location="cpu", weights_only=True)
    source_state = ckpt.get("model", ckpt.get("state_dict", ckpt))
    epoch = ckpt.get("epoch", -1)
    best_metric = ckpt.get("best_metric", -1.0)
    print(f"  Epoch   : {epoch}")
    if best_metric >= 0:
        print(f"  Dice    : {best_metric:.4f}")

    model = build_expert_3d(swin_ec, in_channels=in_channels, num_classes=num_classes)
    info = model.load_state_dict(source_state, strict=False)
    if info.missing_keys:
        print(f"\n  Missing keys ({len(info.missing_keys)}):")
        for k in info.missing_keys[:10]:
            print(f"    - {k}")
        raise RuntimeError(f"Weight import failed: {len(info.missing_keys)} missing keys.")
    if info.unexpected_keys:
        print(f"  Unexpected keys ({len(info.unexpected_keys)}) — ignored")
    model.load_state_dict(source_state, strict=True)
    print("  All weights loaded (strict match)")

    out_dir = run_dir / "checkpoints" / args.layer / f"fold{args.fold}" / swin_name
    best_path = out_dir / "best.pt"
    if best_path.exists() and not args.overwrite:
        existing = torch.load(best_path, map_location="cpu", weights_only=False)
        if str(existing.get("source_path", "")) and Path(existing.get("source_path", "")).resolve() != source.resolve():
            raise RuntimeError(f"Checkpoint conflict at {best_path}. Use --overwrite.")
    out_dir.mkdir(parents=True, exist_ok=True)

    seg_moe_ckpt = {
        "model": model.state_dict(), "epoch": epoch, "global_step": -1,
        "best_metric": best_metric, "source": "monai_swinunetr_official_3d",
        "source_path": str(source),
    }
    torch.save(seg_moe_ckpt, best_path)
    torch.save(seg_moe_ckpt, out_dir / "last.pt")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Saved : {best_path}")
    print(f"  Params: {n_params:,}")
    print("=" * 60)
    print("Import complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

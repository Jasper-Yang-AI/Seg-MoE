#!/usr/bin/env python
"""
Import SegResNet 3D official-training weights into Seg-MoE format.

注意: 官方训练使用 SegResNetDS(dsdepth=2), Seg-MoE 使用 SegResNet(dsdepth=1).
本脚本自动处理 key 映射差异, 只保留主输出头权重.

Usage:
    python scripts/monai/import_segresnet_weights_3d.py \\
        --source runs/segresnet_official_3d_prostate/fold0/best_model.pt \\
        --exp  configs/3d/exp/exp_prostate_local_3d.yaml \\
        --models configs/3d/models_3d.yaml --fold 0

    # 5-fold batch (PowerShell)
    foreach ($fold in 0..4) {
        python scripts/monai/import_segresnet_weights_3d.py `
            --source runs/segresnet_official_3d_prostate_local_3d/fold$fold/best_model.pt `
            --exp  configs/3d/exp/exp_prostate_local_3d.yaml `
            --models configs/3d/models_3d.yaml --fold $fold `
            --expert-name segresnet-3d
    }
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import torch

from seg_moe.models.factory_3d import build_expert_3d, expert_name_3d, list_experts_3d
from seg_moe.utils.config import load_config, resolve_run_dir


def _resolve_segresnet_expert(experts, requested_name):
    types = {"segresnet", "seg_resnet", "monai_segresnet", "monai_segresnet_ds"}
    candidates = [e for e in experts if e.get("type", "").lower() in types]
    if requested_name:
        for e in candidates:
            if expert_name_3d(e) == requested_name:
                return e, requested_name
        raise ValueError(f"Expert '{requested_name}' not found. Candidates: {[expert_name_3d(e) for e in candidates]}")
    if len(candidates) == 1:
        return candidates[0], expert_name_3d(candidates[0])
    if not candidates:
        raise ValueError("No SegResNet expert found in experts_3d")
    raise ValueError(f"Multiple SegResNet: specify --expert-name. Candidates: {[expert_name_3d(e) for e in candidates]}")


def _ds_to_non_ds_mapping(source_sd: dict, target_sd: dict) -> dict:
    """Map SegResNetDS state_dict keys to SegResNet (non-DS) keys.

    SegResNetDS adds 'conv_final.2.' for deep-supervision heads.
    SegResNet has a single 'conv_final.' output head.
    Shared encoder/decoder keys are identical.
    """
    mapped = {}
    for key, val in source_sd.items():
        # Skip DS auxiliary heads (conv_final.2.X, conv_final.3.X, etc.)
        if "conv_final." in key:
            parts = key.split("conv_final.")
            sub = parts[-1]
            # conv_final.0.X is the main output head → map to conv_final.X in target
            if sub.startswith("0."):
                new_key = key.replace("conv_final.0.", "conv_final.")
                if new_key in target_sd:
                    mapped[new_key] = val
                elif key in target_sd:
                    mapped[key] = val
            # For SegResNetDS that has conv_final as single head (dsdepth=1 fallback)
            elif not sub[0].isdigit():
                if key in target_sd:
                    mapped[key] = val
        else:
            if key in target_sd:
                mapped[key] = val
    return mapped


def main() -> None:
    ap = argparse.ArgumentParser(description="Import SegResNet 3D official weights into Seg-MoE")
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
    seg_ec, seg_name = _resolve_segresnet_expert(experts, args.expert_name)

    num_classes = int(dataset_cfg["task"]["num_classes"])
    in_channels = int(dataset_cfg["input"].get("image_channels", 3))

    print()
    print("=" * 60)
    print("Importing SegResNet 3D weights into Seg-MoE")
    print("=" * 60)
    print(f"  Source  : {source}")
    print(f"  Expert  : {seg_name}")
    print(f"  Fold    : {args.fold}")

    ckpt = torch.load(source, map_location="cpu", weights_only=True)
    source_state = ckpt.get("model", ckpt.get("state_dict", ckpt))
    epoch = ckpt.get("epoch", -1)
    best_metric = ckpt.get("best_metric", -1.0)
    print(f"  Epoch   : {epoch}")
    if best_metric >= 0:
        print(f"  Dice    : {best_metric:.4f}")

    # Build Seg-MoE model (SegResNet, non-DS for inference)
    model = build_expert_3d(seg_ec, in_channels=in_channels, num_classes=num_classes)
    target_sd = model.state_dict()

    # Try direct load first
    try:
        model.load_state_dict(source_state, strict=True)
        print("  Direct strict load succeeded")
    except RuntimeError:
        # Fall back to DS→non-DS mapping
        print("  Direct load failed (likely DS→non-DS). Attempting key mapping...")
        mapped = _ds_to_non_ds_mapping(source_state, target_sd)
        missing = [k for k in target_sd if k not in mapped]
        unexpected = [k for k in mapped if k not in target_sd]

        if missing:
            print(f"  Missing keys ({len(missing)}):")
            for k in missing[:10]:
                print(f"    - {k}")
            if len(missing) > len(target_sd) * 0.1:
                raise RuntimeError(f"Too many missing keys ({len(missing)}/{len(target_sd)}). Architecture mismatch.")
            # For remaining missing keys, keep random init
            for k in missing:
                mapped[k] = target_sd[k]

        model.load_state_dict(mapped, strict=True)
        print(f"  DS→non-DS mapping: loaded {len(mapped) - len(missing)} transferred, {len(missing)} kept random")

    # Save
    out_dir = run_dir / "checkpoints" / args.layer / f"fold{args.fold}" / seg_name
    best_path = out_dir / "best.pt"
    if best_path.exists() and not args.overwrite:
        existing = torch.load(best_path, map_location="cpu", weights_only=False)
        if str(existing.get("source_path", "")) and Path(existing.get("source_path", "")).resolve() != source.resolve():
            raise RuntimeError(f"Checkpoint conflict at {best_path}. Use --overwrite.")
    out_dir.mkdir(parents=True, exist_ok=True)

    seg_moe_ckpt = {
        "model": model.state_dict(), "epoch": epoch, "global_step": -1,
        "best_metric": best_metric, "source": "monai_segresnet_official_3d",
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

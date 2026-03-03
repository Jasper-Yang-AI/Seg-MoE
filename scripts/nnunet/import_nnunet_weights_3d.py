#!/usr/bin/env python
"""
Import nnUNet v2 3D official trained weights into Seg-MoE 3D checkpoint format.

与 2D 版本逻辑完全一致,  但:
  - 使用 factory_3d (list_experts_3d, build_expert_3d)
  - 默认 config=3d_fullres
  - Conv3d 架构

Usage:
    # nnUNet 官方训练 (3d_fullres)
    nnUNetv2_train 2 3d_fullres 0 --npz

    # 导入权重
    python scripts/nnunet/import_nnunet_weights_3d.py \\
        --nnunet-base nnunet_data \\
        --dataset-id 2 \\
        --config 3d_fullres \\
        --folds 0 1 2 3 4 \\
        --exp configs/3d/exp/exp_prostate_local_3d.yaml \\
        --models configs/3d/models_3d.yaml \\
        --expert-name nnunet-3d \\
        --update-models-yaml configs/3d/models_3d.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml

from seg_moe.models.factory_3d import build_expert_3d, expert_name_3d, list_experts_3d
from seg_moe.utils.config import load_config, resolve_run_dir


# ---------------------------------------------------------------------------
# nnUNet plans parsing (reused from 2D import script)
# ---------------------------------------------------------------------------

def _find_dataset_dir(nnunet_results: Path, dataset_id: int) -> Optional[Path]:
    pattern = f"Dataset{dataset_id:03d}"
    for d in nnunet_results.iterdir():
        if d.name.startswith(pattern) and d.is_dir():
            return d
    return None


def _find_trainer_dir(dataset_dir: Path, config: str) -> Optional[Path]:
    for d in dataset_dir.iterdir():
        if d.is_dir() and d.name.endswith(f"__{config}"):
            return d
    default = dataset_dir / f"nnUNetTrainer__nnUNetPlans__{config}"
    return default if default.exists() else None


def _load_plans(nnunet_preprocessed: Path, dataset_id: int) -> Dict[str, Any]:
    pattern = f"Dataset{dataset_id:03d}"
    for d in nnunet_preprocessed.iterdir():
        if d.name.startswith(pattern) and d.is_dir():
            plans_path = d / "nnUNetPlans.json"
            if plans_path.exists():
                with open(plans_path, "r") as f:
                    return json.load(f)
            raise FileNotFoundError(f"Plans not found: {plans_path}")
    raise FileNotFoundError(f"No preprocessed directory for {pattern}")


def _extract_architecture_config(plans: Dict[str, Any], config: str) -> Dict[str, Any]:
    """Extract architecture config from nnUNet plans (supports new & legacy formats)."""
    if config not in plans.get("configurations", {}):
        available = list(plans.get("configurations", {}).keys())
        raise ValueError(f"Config '{config}' not in plans. Available: {available}")

    cfg = plans["configurations"][config]

    # New format: architecture.arch_kwargs
    arch = cfg.get("architecture", None)
    if arch and "arch_kwargs" in arch:
        kw = arch["arch_kwargs"]
        n_stages = kw.get("n_stages", len(kw.get("kernel_sizes", [])))
        return {
            "n_stages": n_stages,
            "features_per_stage": list(kw["features_per_stage"]),
            "conv_kernel_sizes": kw.get("kernel_sizes", [[3, 3, 3]] * n_stages),
            "pool_op_kernel_sizes": kw.get("strides", [[1, 1, 1]] + [[2, 2, 2]] * (n_stages - 1)),
            "n_conv_per_stage_encoder": kw.get("n_conv_per_stage", [2] * n_stages),
            "n_conv_per_stage_decoder": kw.get("n_conv_per_stage_decoder", [2] * (n_stages - 1)),
            "UNet_class_name": arch.get("network_class_name", "PlainConvUNet").split(".")[-1],
            "patch_size": cfg.get("patch_size", [128, 128, 64]),
        }

    # Legacy format
    base_features = cfg.get("UNet_base_num_features", 32)
    max_features = cfg.get("unet_max_num_features", 512)
    conv_kernel_sizes = cfg.get("conv_kernel_sizes", [])
    n_stages = len(conv_kernel_sizes) if conv_kernel_sizes else cfg.get("n_stages", 5)
    features_per_stage = [min(base_features * (2 ** i), max_features) for i in range(n_stages)]
    pool_op_kernel_sizes = cfg.get("pool_op_kernel_sizes", [[1, 1, 1]] + [[2, 2, 2]] * (n_stages - 1))

    return {
        "n_stages": n_stages,
        "features_per_stage": features_per_stage,
        "conv_kernel_sizes": conv_kernel_sizes if conv_kernel_sizes else [[3, 3, 3]] * n_stages,
        "pool_op_kernel_sizes": pool_op_kernel_sizes,
        "n_conv_per_stage_encoder": cfg.get("n_conv_per_stage_encoder", [2] * n_stages),
        "n_conv_per_stage_decoder": cfg.get("n_conv_per_stage_decoder", [2] * (n_stages - 1)),
        "UNet_class_name": cfg.get("UNet_class_name", "PlainConvUNet"),
        "patch_size": cfg.get("patch_size", [128, 128, 64]),
    }


def _load_nnunet_checkpoint(nnunet_results, dataset_id, config, fold, which="best"):
    dataset_dir = _find_dataset_dir(nnunet_results, dataset_id)
    if dataset_dir is None:
        raise FileNotFoundError(f"No results for Dataset{dataset_id:03d}")
    trainer_dir = _find_trainer_dir(dataset_dir, config)
    if trainer_dir is None:
        raise FileNotFoundError(f"No trainer dir for '{config}' in {dataset_dir}")
    fold_dir = trainer_dir / f"fold_{fold}"
    if not fold_dir.exists():
        raise FileNotFoundError(f"Fold dir not found: {fold_dir}")

    ckpt_path = fold_dir / f"checkpoint_{which}.pth"
    if not ckpt_path.exists():
        ckpt_path = fold_dir / f"checkpoint_best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    network_weights = ckpt.get("network_weights")
    if network_weights is None:
        raise ValueError(f"'network_weights' not in checkpoint. Keys: {list(ckpt.keys())}")

    info = {
        "current_epoch": ckpt.get("current_epoch", -1),
        "trainer_name": ckpt.get("trainer_name", "unknown"),
        "source_path": str(ckpt_path),
    }
    return network_weights, info


# ---------------------------------------------------------------------------
# Build nnUNet wrapper and load weights
# ---------------------------------------------------------------------------

def _build_and_load(arch_config, network_weights, in_channels, num_classes):
    """Build NnUNet wrapper (3D) and load official weights."""
    try:
        from seg_moe.models.wrappers.nnunet_wrapper import NnUNetWrapper
    except ImportError:
        raise ImportError("NnUNetWrapper not found. Ensure seg_moe.models.wrappers.nnunet_wrapper exists.")

    wrapper = NnUNetWrapper(
        in_channels=in_channels,
        num_classes=num_classes,
        patch_size=tuple(arch_config["patch_size"]),
        n_stages=arch_config["n_stages"],
        features_per_stage=arch_config["features_per_stage"],
        conv_op="Conv3d",
        deep_supervision=True,
        n_conv_per_stage_encoder=arch_config.get("n_conv_per_stage_encoder"),
        n_conv_per_stage_decoder=arch_config.get("n_conv_per_stage_decoder"),
        kernel_sizes=arch_config.get("conv_kernel_sizes"),
        strides=arch_config.get("pool_op_kernel_sizes"),
    )

    prefixed = {f"model.{k}": v for k, v in network_weights.items()}
    info = wrapper.load_state_dict(prefixed, strict=False)
    if info.missing_keys:
        print(f"  Missing keys ({len(info.missing_keys)}):")
        for k in info.missing_keys[:10]:
            print(f"    - {k}")
        raise RuntimeError(f"Weight import failed: {len(info.missing_keys)} missing keys.")
    wrapper.load_state_dict(prefixed, strict=True)
    print("  All weights loaded (strict match)")
    return wrapper


def _resolve_expert_name(models_cfg, requested):
    experts = list_experts_3d(models_cfg)
    nnunet_experts = [e for e in experts if e.get("type", "").lower() in ("nnunet", "nnunet_v2")]
    if requested:
        for e in nnunet_experts:
            if expert_name_3d(e) == requested:
                return requested
        raise ValueError(f"Expert '{requested}' not found. Candidates: {[expert_name_3d(e) for e in nnunet_experts]}")
    if len(nnunet_experts) == 1:
        return expert_name_3d(nnunet_experts[0])
    if not nnunet_experts:
        raise ValueError("No nnUNet expert in experts_3d")
    raise ValueError(f"Multiple nnUNet experts, specify --expert-name: {[expert_name_3d(e) for e in nnunet_experts]}")


def _update_models_yaml_3d(yaml_path: Path, arch_config: Dict[str, Any]) -> None:
    """Auto-update models_3d.yaml with actual nnUNet 3D plans architecture."""
    if not yaml_path.exists():
        print(f"  Warning: {yaml_path} not found")
        return
    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    experts = cfg.get("experts_3d", [])
    idx = None
    for i, e in enumerate(experts):
        if e.get("type", "").lower() in ("nnunet", "nnunet_v2"):
            idx = i
            break
    if idx is None:
        print("  No nnunet expert in experts_3d, skip")
        return
    experts[idx]["params"] = {
        "conv_op": "Conv3d",
        "n_stages": arch_config["n_stages"],
        "features_per_stage": arch_config["features_per_stage"],
        "patch_size": arch_config["patch_size"],
        "deep_supervision": True,
        "n_conv_per_stage_encoder": arch_config.get("n_conv_per_stage_encoder"),
        "n_conv_per_stage_decoder": arch_config.get("n_conv_per_stage_decoder"),
        "conv_kernel_sizes": arch_config.get("conv_kernel_sizes"),
        "pool_op_kernel_sizes": arch_config.get("pool_op_kernel_sizes"),
    }
    header_lines = []
    for line in yaml_path.read_text("utf-8").splitlines():
        if line.startswith("#") or line.strip() == "":
            header_lines.append(line)
        else:
            break
    header = "\n".join(header_lines) + "\n\n" if header_lines else ""
    yaml_path.write_text(
        header + yaml.dump(cfg, default_flow_style=False, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    print(f"  Updated {yaml_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Import nnUNet v2 3D weights into Seg-MoE")
    ap.add_argument("--nnunet-base", default="nnunet_data")
    ap.add_argument("--dataset-id", type=int, required=True)
    ap.add_argument("--config", default="3d_fullres", help="nnUNet config (3d_fullres / 3d_lowres)")
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--which", default="best", choices=["best", "final"])
    ap.add_argument("--exp", required=True)
    ap.add_argument("--models", default="configs/3d/models_3d.yaml")
    ap.add_argument("--expert-name", default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--update-models-yaml", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    base_dir = Path(args.nnunet_base).resolve()
    nnunet_results = base_dir / "nnUNet_results"
    nnunet_preprocessed = base_dir / "nnUNet_preprocessed"

    exp_cfg = load_config(args.exp)
    models_cfg = load_config(args.models)
    dataset_cfg = load_config(exp_cfg["dataset"]["config"])
    run_dir = Path(resolve_run_dir(exp_cfg))
    expert = _resolve_expert_name(models_cfg, args.expert_name)

    num_classes = int(dataset_cfg["task"]["num_classes"])
    in_channels = int(dataset_cfg["input"].get("image_channels", 3))
    print(f"Dataset: {dataset_cfg['name']} | in_ch={in_channels}, classes={num_classes}")

    plans = _load_plans(nnunet_preprocessed, args.dataset_id)
    arch = _extract_architecture_config(plans, args.config)

    print(f"\nnnUNet 3D architecture:")
    print(f"  n_stages:             {arch['n_stages']}")
    print(f"  features_per_stage:   {arch['features_per_stage']}")
    print(f"  patch_size:           {arch['patch_size']}")

    success = 0
    for fold in args.folds:
        print(f"\n{'='*60}\nImporting fold {fold}\n{'='*60}")
        try:
            weights, info = _load_nnunet_checkpoint(
                nnunet_results, args.dataset_id, args.config, fold, args.which)
        except FileNotFoundError as e:
            print(f"  Skip fold {fold}: {e}")
            continue

        print(f"  nnUNet epoch: {info.get('current_epoch', '?')}  keys: {len(weights)}")

        wrapper = _build_and_load(arch, weights, in_channels, num_classes)

        out_dir = run_dir / "checkpoints" / "layer1" / f"fold{fold}" / expert
        if args.dry_run:
            print(f"  [dry-run] Would save to {out_dir}")
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        best_path = out_dir / "best.pt"
        if best_path.exists() and not args.overwrite:
            existing = torch.load(best_path, map_location="cpu", weights_only=False)
            if str(existing.get("nnunet_source", "")) and Path(existing["nnunet_source"]).resolve() != Path(info["source_path"]).resolve():
                raise RuntimeError(f"Checkpoint conflict at {best_path}. Use --overwrite.")

        save_dict = {
            "model": wrapper.state_dict(),
            "epoch": info.get("current_epoch", 1000),
            "global_step": -1,
            "best_metric": -1.0,
            "nnunet_source": info["source_path"],
            "nnunet_plans": {
                "n_stages": arch["n_stages"],
                "features_per_stage": arch["features_per_stage"],
                "patch_size": arch["patch_size"],
                "deep_supervision": True,
            },
        }
        torch.save(save_dict, best_path)
        torch.save(save_dict, out_dir / "last.pt")
        print(f"  Saved: {best_path}")
        success += 1

    if args.update_models_yaml:
        _update_models_yaml_3d(Path(args.update_models_yaml), arch)

    print(f"\n{'='*60}")
    print(f"Import complete! {success}/{len(args.folds)} folds")
    print("=" * 60)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Import nnUNet v2 official trained weights into Seg-MoE checkpoint format.

完全兼容官方 nnUNet v2 的 plans 格式和 checkpoint 格式:
  - 从 nnUNetPlans.json 的 architecture.arch_kwargs 提取网络参数
  - 从 checkpoint 的 network_weights 提取权重
  - 自动处理新/旧 plans 格式
  - 自动更新 models.yaml 中的 nnUNet expert 配置

Usage:
    python scripts/nnunet/import_nnunet_weights.py \
        --nnunet-base nnunet_data \
        --dataset-id 3 \
        --config 2d \
        --folds 0 1 2 3 4 \
        --exp configs/2d/exp/exp_msd_task03_liver.yaml

    # 仅导入 fold 0, dry-run 查看配置
    python scripts/nnunet/import_nnunet_weights.py \
        --nnunet-base nnunet_data \
        --dataset-id 3 \
        --config 2d \
        --folds 0 \
        --exp configs/2d/exp/exp_msd_task03_liver.yaml \
        --dry-run

    # 自动更新 models.yaml
    python scripts/nnunet/import_nnunet_weights.py \
        --nnunet-base nnunet_data \
        --dataset-id 3 \
        --config 2d \
        --folds 0 1 2 3 4 \
        --exp configs/2d/exp/exp_msd_task03_liver.yaml \
        --update-models-yaml configs/2d/models.yaml
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml


def _find_dataset_dir(nnunet_results: Path, dataset_id: int) -> Optional[Path]:
    """Find the nnUNet results directory for a given dataset ID."""
    pattern = f"Dataset{dataset_id:03d}"
    for d in nnunet_results.iterdir():
        if d.name.startswith(pattern) and d.is_dir():
            return d
    return None


def _find_trainer_dir(dataset_dir: Path, config: str) -> Optional[Path]:
    """Find the trainer directory (e.g., nnUNetTrainer__nnUNetPlans__2d)."""
    for d in dataset_dir.iterdir():
        if d.is_dir() and d.name.endswith(f"__{config}"):
            return d
    # Fallback: exact match
    default = dataset_dir / f"nnUNetTrainer__nnUNetPlans__{config}"
    if default.exists():
        return default
    return None


def _load_plans(nnunet_preprocessed: Path, dataset_id: int) -> Dict[str, Any]:
    """Load nnUNet plans JSON."""
    dataset_dir = None
    pattern = f"Dataset{dataset_id:03d}"
    for d in nnunet_preprocessed.iterdir():
        if d.name.startswith(pattern) and d.is_dir():
            dataset_dir = d
            break

    if dataset_dir is None:
        raise FileNotFoundError(f"No preprocessed directory found for {pattern} in {nnunet_preprocessed}")

    plans_path = dataset_dir / "nnUNetPlans.json"
    if not plans_path.exists():
        raise FileNotFoundError(f"Plans file not found: {plans_path}")

    with open(plans_path, "r") as f:
        plans = json.load(f)
    return plans


def _extract_architecture_config(
    plans: Dict[str, Any],
    config: str = "2d",
) -> Dict[str, Any]:
    """Extract architecture configuration from nnUNet plans.

    兼容新版 (architecture.arch_kwargs) 和旧版 (UNet_base_num_features) plans 格式.

    Returns a dict with keys: n_stages, features_per_stage, conv_kernel_sizes,
        pool_op_kernel_sizes, n_conv_per_stage_encoder, n_conv_per_stage_decoder,
        UNet_class_name, patch_size.
    """
    if config not in plans.get("configurations", {}):
        available = list(plans.get("configurations", {}).keys())
        raise ValueError(f"Configuration '{config}' not found in plans. Available: {available}")

    cfg = plans["configurations"][config]

    # ---- 新版 plans: architecture.arch_kwargs 直接存储所有参数 ----
    arch_section = cfg.get("architecture", None)
    if arch_section and "arch_kwargs" in arch_section:
        kw = arch_section["arch_kwargs"]
        n_stages = kw.get("n_stages", len(kw.get("kernel_sizes", [])))

        arch_config = {
            "n_stages": n_stages,
            "features_per_stage": list(kw["features_per_stage"]),
            "conv_kernel_sizes": kw.get("kernel_sizes", [[3, 3]] * n_stages),
            "pool_op_kernel_sizes": kw.get("strides", [[1, 1]] + [[2, 2]] * (n_stages - 1)),
            "n_conv_per_stage_encoder": kw.get("n_conv_per_stage", [2] * n_stages),
            "n_conv_per_stage_decoder": kw.get("n_conv_per_stage_decoder", [2] * (n_stages - 1)),
            "UNet_class_name": arch_section.get("network_class_name", "PlainConvUNet").split(".")[-1],
            "patch_size": cfg.get("patch_size", [256, 256]),
        }
        print(f"  [plans] New format: architecture.arch_kwargs found")
        return arch_config

    # ---- 旧版 plans: 从 UNet_base_num_features + unet_max_num_features 计算 ----
    print(f"  [plans] Legacy format: computing features from UNet_base_num_features")
    base_features = cfg.get("UNet_base_num_features", 32)
    max_features = cfg.get("unet_max_num_features", 512)
    conv_kernel_sizes = cfg.get("conv_kernel_sizes", [])
    n_stages = len(conv_kernel_sizes) if conv_kernel_sizes else cfg.get("n_stages", 6)

    features_per_stage = [min(base_features * (2 ** i), max_features) for i in range(n_stages)]
    pool_op_kernel_sizes = cfg.get("pool_op_kernel_sizes", [[1, 1]] + [[2, 2]] * (n_stages - 1))

    arch_config = {
        "n_stages": n_stages,
        "features_per_stage": features_per_stage,
        "conv_kernel_sizes": conv_kernel_sizes or [[3, 3]] * n_stages,
        "pool_op_kernel_sizes": pool_op_kernel_sizes,
        "n_conv_per_stage_encoder": cfg.get("n_conv_per_stage_encoder", [2] * n_stages),
        "n_conv_per_stage_decoder": cfg.get("n_conv_per_stage_decoder", [2] * (n_stages - 1)),
        "UNet_class_name": cfg.get("UNet_class_name", "PlainConvUNet"),
        "patch_size": cfg.get("patch_size", [256, 256]),
    }

    return arch_config


def _load_nnunet_checkpoint(
    nnunet_results: Path,
    dataset_id: int,
    config: str,
    fold: int,
    which: str = "best",
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Load nnUNet checkpoint and return (network_weights, checkpoint_info)."""
    dataset_dir = _find_dataset_dir(nnunet_results, dataset_id)
    if dataset_dir is None:
        raise FileNotFoundError(f"No results directory for Dataset{dataset_id:03d} in {nnunet_results}")

    trainer_dir = _find_trainer_dir(dataset_dir, config)
    if trainer_dir is None:
        raise FileNotFoundError(f"No trainer directory found for config '{config}' in {dataset_dir}")

    fold_dir = trainer_dir / f"fold_{fold}"
    if not fold_dir.exists():
        raise FileNotFoundError(f"Fold directory not found: {fold_dir}")

    # nnUNet v2 checkpoint naming
    if which == "best":
        ckpt_path = fold_dir / "checkpoint_best.pth"
    elif which == "final":
        ckpt_path = fold_dir / "checkpoint_final.pth"
    else:
        ckpt_path = fold_dir / which

    if not ckpt_path.exists():
        # Try alternative naming
        alt_path = fold_dir / f"checkpoint_{which}.pth"
        if alt_path.exists():
            ckpt_path = alt_path
        else:
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading nnUNet checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    network_weights = ckpt.get("network_weights", None)
    if network_weights is None:
        raise ValueError(f"'network_weights' key not found in checkpoint. Keys: {list(ckpt.keys())}")

    info = {
        "current_epoch": ckpt.get("current_epoch", -1),
        "trainer_name": ckpt.get("trainer_name", "unknown"),
        "fold": fold,
        "source_path": str(ckpt_path),
    }

    # Try to get init_args for verification
    init_args = ckpt.get("init_args", {})
    if init_args:
        info["init_args"] = {
            k: v for k, v in init_args.items()
            if k in ("plans", "configuration", "fold", "dataset_json")
        }

    return network_weights, info


def _build_wrapper_and_load(
    arch_config: Dict[str, Any],
    network_weights: Dict[str, torch.Tensor],
    in_channels: int,
    num_classes: int,
) -> torch.nn.Module:
    """Build NnUNetWrapper with matching architecture and load nnUNet weights."""
    from seg_moe.models.wrappers.nnunet_wrapper import NnUNetWrapper

    is_2d = len(arch_config["patch_size"]) == 2

    wrapper = NnUNetWrapper(
        in_channels=in_channels,
        num_classes=num_classes,
        patch_size=tuple(arch_config["patch_size"]),
        n_stages=arch_config["n_stages"],
        features_per_stage=arch_config["features_per_stage"],
        conv_op="Conv2d" if is_2d else "Conv3d",
        deep_supervision=True,  # nnUNet 官方训练使用深度监督
        n_conv_per_stage_encoder=arch_config.get("n_conv_per_stage_encoder"),
        n_conv_per_stage_decoder=arch_config.get("n_conv_per_stage_decoder"),
        kernel_sizes=arch_config.get("conv_kernel_sizes"),
        strides=arch_config.get("pool_op_kernel_sizes"),
    )

    # nnUNet state dict keys are direct PlainConvUNet keys (e.g., "encoder.stages.0...")
    # NnUNetWrapper stores PlainConvUNet as self.model, so keys need "model." prefix
    prefixed_weights = {}
    for k, v in network_weights.items():
        prefixed_weights[f"model.{k}"] = v

    # Load with strict=False first to see mismatches, then strict=True
    info = wrapper.load_state_dict(prefixed_weights, strict=False)
    if info.missing_keys:
        print(f"  Missing keys ({len(info.missing_keys)}):")
        for k in info.missing_keys[:10]:
            print(f"    - {k}")
        if len(info.missing_keys) > 10:
            print(f"    ... and {len(info.missing_keys) - 10} more")
    if info.unexpected_keys:
        print(f"  Unexpected keys ({len(info.unexpected_keys)}):")
        for k in info.unexpected_keys[:10]:
            print(f"    - {k}")
        if len(info.unexpected_keys) > 10:
            print(f"    ... and {len(info.unexpected_keys) - 10} more")

    if not info.missing_keys and not info.unexpected_keys:
        print("  All weights loaded successfully (strict match)")

    return wrapper


def _print_yaml_snippet(arch_config: Dict[str, Any], output_dir: str) -> None:
    """Print the models.yaml configuration snippet for the imported nnUNet."""
    snippet = {
        "name": "nnunet-2d",
        "type": "nnunet",
        "params": {
            "conv_op": "Conv2d",
            "n_stages": arch_config["n_stages"],
            "features_per_stage": arch_config["features_per_stage"],
            "patch_size": arch_config["patch_size"],
            "deep_supervision": True,
            "n_conv_per_stage_encoder": arch_config.get("n_conv_per_stage_encoder"),
            "n_conv_per_stage_decoder": arch_config.get("n_conv_per_stage_decoder"),
            "conv_kernel_sizes": arch_config.get("conv_kernel_sizes"),
            "pool_op_kernel_sizes": arch_config.get("pool_op_kernel_sizes"),
        },
    }

    print("\n" + "=" * 60)
    print("models.yaml nnUNet expert 配置片段:")
    print("=" * 60)
    print(yaml.dump({"experts_v2": [snippet]}, default_flow_style=False, allow_unicode=True, sort_keys=False))


def _update_models_yaml(yaml_path: Path, arch_config: Dict[str, Any]) -> None:
    """Auto-update models.yaml with the actual nnUNet plans architecture.

    Reads the YAML, finds the nnunet expert, and updates its params to match
    the official nnUNet plans. Preserves comments where possible.
    """
    if not yaml_path.exists():
        print(f"  Warning: {yaml_path} not found, skip auto-update")
        return

    content = yaml_path.read_text(encoding="utf-8")

    # Parse YAML to find and update nnunet expert params
    cfg = yaml.safe_load(content)
    experts = cfg.get("experts_v2", [])

    nnunet_idx = None
    for i, e in enumerate(experts):
        if e.get("type") == "nnunet":
            nnunet_idx = i
            break

    if nnunet_idx is None:
        print(f"  Warning: No nnunet expert found in {yaml_path}, skip auto-update")
        return

    # Update the params
    experts[nnunet_idx]["params"] = {
        "conv_op": "Conv2d" if len(arch_config["patch_size"]) == 2 else "Conv3d",
        "n_stages": arch_config["n_stages"],
        "features_per_stage": arch_config["features_per_stage"],
        "patch_size": arch_config["patch_size"],
        "deep_supervision": True,
        "n_conv_per_stage_encoder": arch_config.get("n_conv_per_stage_encoder"),
        "n_conv_per_stage_decoder": arch_config.get("n_conv_per_stage_decoder"),
        "conv_kernel_sizes": arch_config.get("conv_kernel_sizes"),
        "pool_op_kernel_sizes": arch_config.get("pool_op_kernel_sizes"),
    }

    # Re-serialize — we lose comments, so rebuild with clear structure
    # Read original header comments (lines starting with #)
    header_lines = []
    for line in content.splitlines():
        if line.startswith("#") or line.strip() == "":
            header_lines.append(line)
        else:
            break
    header = "\n".join(header_lines) + "\n\n" if header_lines else ""

    # Write clean YAML
    output = header + yaml.dump(cfg, default_flow_style=False, allow_unicode=True, sort_keys=False)
    yaml_path.write_text(output, encoding="utf-8")
    print(f"  ✅ Updated {yaml_path} with nnUNet plans architecture")
    print(f"     features_per_stage: {arch_config['features_per_stage']}")
    print(f"     patch_size: {arch_config['patch_size']}")
    print(f"     n_stages: {arch_config['n_stages']}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Import nnUNet v2 weights into Seg-MoE")
    ap.add_argument("--nnunet-base", default="nnunet_data", help="nnUNet data root directory")
    ap.add_argument("--dataset-id", type=int, required=True, help="nnUNet dataset ID (e.g., 3)")
    ap.add_argument("--config", default="2d", help="nnUNet configuration (2d / 3d_fullres / 3d_lowres)")
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4], help="Folds to import")
    ap.add_argument("--which", default="best", choices=["best", "final"], help="Which checkpoint to use")
    ap.add_argument("--exp", required=True, help="Seg-MoE experiment config YAML")
    ap.add_argument("--expert-name", default="nnunet-2d", help="Expert name in Seg-MoE")
    ap.add_argument("--update-models-yaml", default=None, help="Path to models.yaml to auto-update")
    ap.add_argument("--dry-run", action="store_true", help="Only display config, don't save")
    args = ap.parse_args()

    base_dir = Path(args.nnunet_base).resolve()
    nnunet_results = base_dir / "nnUNet_results"
    nnunet_preprocessed = base_dir / "nnUNet_preprocessed"

    # Load Seg-MoE experiment config
    from seg_moe.utils.config import load_config, resolve_run_dir
    exp_cfg = load_config(args.exp)
    dataset_cfg = load_config(exp_cfg["dataset"]["config"])
    run_dir = Path(resolve_run_dir(exp_cfg))

    # Infer channels and classes from dataset config
    from seg_moe.data.indexing import infer_image_channels, infer_num_classes
    in_channels = infer_image_channels(dataset_cfg)
    num_classes = infer_num_classes(dataset_cfg)
    print(f"Dataset: {dataset_cfg['name']} | in_channels={in_channels}, num_classes={num_classes}")

    # Load nnUNet plans
    print(f"\nLoading nnUNet plans from: {nnunet_preprocessed}")
    plans = _load_plans(nnunet_preprocessed, args.dataset_id)
    arch_config = _extract_architecture_config(plans, args.config)

    print(f"\nArchitecture from nnUNet plans:")
    print(f"  UNet class:                  {arch_config['UNet_class_name']}")
    print(f"  n_stages:                    {arch_config['n_stages']}")
    print(f"  features_per_stage:          {arch_config['features_per_stage']}")
    print(f"  conv_kernel_sizes:           {arch_config['conv_kernel_sizes']}")
    print(f"  pool_op_kernel_sizes:        {arch_config['pool_op_kernel_sizes']}")
    print(f"  n_conv_per_stage_encoder:    {arch_config['n_conv_per_stage_encoder']}")
    print(f"  n_conv_per_stage_decoder:    {arch_config['n_conv_per_stage_decoder']}")
    print(f"  patch_size:                  {arch_config['patch_size']}")

    if arch_config["UNet_class_name"] != "PlainConvUNet":
        print(f"\n⚠ Warning: nnUNet planned {arch_config['UNet_class_name']}, "
              f"but Seg-MoE wrapper only supports PlainConvUNet.")
        print("  Will attempt to load anyway...")

    # Import each fold
    success_count = 0
    for fold in args.folds:
        print(f"\n{'=' * 60}")
        print(f"Importing fold {fold}")
        print("=" * 60)

        try:
            network_weights, ckpt_info = _load_nnunet_checkpoint(
                nnunet_results, args.dataset_id, args.config, fold, args.which,
            )
        except FileNotFoundError as e:
            print(f"  ⚠ Skip fold {fold}: {e}")
            continue

        print(f"  nnUNet epoch: {ckpt_info.get('current_epoch', '?')}")
        print(f"  Trainer: {ckpt_info.get('trainer_name', '?')}")
        print(f"  Weights keys: {len(network_weights)}")

        # Build wrapper and load weights
        wrapper = _build_wrapper_and_load(
            arch_config, network_weights,
            in_channels=in_channels,
            num_classes=num_classes,
        )

        # Save in Seg-MoE format
        output_dir = run_dir / "checkpoints" / "layer1" / f"fold{fold}" / args.expert_name
        if args.dry_run:
            print(f"  [dry-run] Would save to: {output_dir}")
            continue

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "best.pt"

        save_dict = {
            "model": wrapper.state_dict(),
            "epoch": ckpt_info.get("current_epoch", 1000),
            "global_step": -1,
            "best_metric": -1.0,  # Unknown from nnUNet
            "metrics": {},
            "nnunet_source": ckpt_info.get("source_path", ""),
            "nnunet_plans": {
                "n_stages": arch_config["n_stages"],
                "features_per_stage": arch_config["features_per_stage"],
                "conv_kernel_sizes": arch_config["conv_kernel_sizes"],
                "pool_op_kernel_sizes": arch_config["pool_op_kernel_sizes"],
                "n_conv_per_stage_encoder": arch_config["n_conv_per_stage_encoder"],
                "n_conv_per_stage_decoder": arch_config["n_conv_per_stage_decoder"],
                "patch_size": arch_config["patch_size"],
                "deep_supervision": True,
            },
        }

        torch.save(save_dict, output_path)
        print(f"  ✅ Saved: {output_path}")

        # Also save as last.pt for resume compatibility
        torch.save(save_dict, output_dir / "last.pt")
        success_count += 1

    # Print config snippet
    output_dir_example = f"runs/{exp_cfg['exp_name']}/checkpoints/layer1/fold0/{args.expert_name}"
    _print_yaml_snippet(arch_config, output_dir_example)

    # Auto-update models.yaml if requested
    if args.update_models_yaml:
        print(f"\nAuto-updating models.yaml: {args.update_models_yaml}")
        _update_models_yaml(Path(args.update_models_yaml), arch_config)

    print(f"\n{'=' * 60}")
    print(f"导入完成! 成功: {success_count}/{len(args.folds)} folds")
    print("=" * 60)
    print()
    if not args.update_models_yaml:
        print("提示: 使用 --update-models-yaml configs/2d/models.yaml 可自动更新配置文件")
        print()
    print("后续步骤:")
    print(f"  # 1. 训练 SwinUNETR + SegResNet (nnUNet --skip-if-done 自动跳过)")
    print(f"  python scripts/train/train_2d_experts.py \\")
    print(f"    --exp {args.exp} \\")
    print(f"    --training configs/2d/training_dual_5090.yaml \\")
    print(f"    --models configs/2d/models.yaml \\")
    print(f"    --augs configs/2d/augs.yaml \\")
    print(f"    --fold 0 --layer layer1 --gpus 0,1 --skip-if-done")
    print()
    print("  # 2. 生成 OOF 预测 → 训练 Layer2 → 评估")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Import SegResNet official-training weights into Seg-MoE format.

与 import_swinunetr_weights.py 类似:
  - 训练使用 SegResNetDS (dsdepth=2, 深度监督)
  - Seg-MoE 推理使用 SegResNetDS (dsdepth=1, 单输出)
  - 两者 state_dict 中, dsdepth=1 的 keys 是 dsdepth=2 的子集
  - 导入时仅加载推理所需的权重, 忽略深度监督辅助头参数

Usage:
    # 单折导入
    python scripts/monai/import_segresnet_weights.py \\
        --source runs/segresnet_official_msd_task03_liver/fold0/best_model.pt \\
        --exp  configs/2d/exp/exp_msd_task03_liver.yaml \\
        --models configs/2d/models.yaml --fold 0

    # 5 折批量导入 (PowerShell)
    foreach ($fold in 0..4) {
        python scripts/monai/import_segresnet_weights.py `
            --source runs/segresnet_official_msd_task03_liver/fold$fold/best_model.pt `
            --exp  configs/2d/exp/exp_msd_task03_liver.yaml `
            --models configs/2d/models.yaml --fold $fold
    }

导入后 train_2d_experts.py --skip-if-done 会自动跳过 SegResNet
(因为 best.pt 已存在).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from seg_moe.data.indexing import infer_num_classes
from seg_moe.models.factory_2d import build_expert, expert_name, list_experts
from seg_moe.utils.config import load_config, resolve_run_dir


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Import SegResNet official weights into Seg-MoE"
    )
    ap.add_argument("--source", required=True, help="Official training checkpoint (best_model.pt)")
    ap.add_argument("--exp", required=True, help="Experiment config YAML")
    ap.add_argument("--models", required=True, help="Models config YAML")
    ap.add_argument("--fold", type=int, required=True, help="Fold index (0-4)")
    ap.add_argument("--layer", default="layer1", choices=["layer1", "layer2"])
    ap.add_argument("--dataset-config", default=None)
    args = ap.parse_args()

    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(f"Source checkpoint not found: {source}")

    # ── Load configs ──
    exp_cfg = load_config(args.exp)
    models_cfg = load_config(args.models)
    dataset_cfg = load_config(args.dataset_config or exp_cfg["dataset"]["config"])
    run_dir = resolve_run_dir(exp_cfg)

    # ── Find SegResNet expert config ──
    seg_cfg = None
    seg_name = None
    for ec in list_experts(models_cfg):
        if ec.get("type", "").lower() in ("monai_segresnet", "monai_segresnet_ds"):
            seg_cfg = ec
            seg_name = expert_name(ec)
            break
    if seg_cfg is None:
        raise ValueError("No SegResNet expert found in models.yaml (experts_v2)")

    num_classes = infer_num_classes(dataset_cfg)
    in_channels = 3  # ImageNet-normalised

    # ── Banner ──
    print()
    print("=" * 60)
    print("Importing SegResNet weights into Seg-MoE")
    print("=" * 60)
    print(f"  Source  : {source}")
    print(f"  Expert  : {seg_name}")
    print(f"  Fold    : {args.fold}")
    print(f"  Layer   : {args.layer}")

    # ── Load source checkpoint ──
    ckpt = torch.load(source, map_location="cpu", weights_only=True)
    source_state = ckpt.get("model", ckpt.get("state_dict", ckpt))
    epoch = ckpt.get("epoch", -1)
    best_metric = ckpt.get("best_metric", -1.0)
    train_dsdepth = ckpt.get("config", {}).get("dsdepth", 2)

    print(f"  Epoch   : {epoch}")
    if best_metric >= 0:
        print(f"  Dice    : {best_metric:.4f}")
    print(f"  Train DS: dsdepth={train_dsdepth}")

    # ── Build Seg-MoE model (dsdepth=1 for inference) ──
    model = build_expert(seg_cfg, in_channels=in_channels, num_classes=num_classes)

    # ── Load weights ──
    # Training model (dsdepth=2) has extra deep supervision head keys
    # that the inference model (dsdepth=1) doesn't have.
    # Use strict=False to allow these extra keys, then verify no missing keys.
    info = model.load_state_dict(source_state, strict=False)

    if info.missing_keys:
        print(f"\n  ❌ Missing keys ({len(info.missing_keys)}):")
        for k in info.missing_keys[:10]:
            print(f"    - {k}")
        raise RuntimeError(
            f"Weight import failed: {len(info.missing_keys)} missing keys. "
            "Architecture mismatch between source checkpoint and models.yaml. "
            "Ensure the same SegResNet params (init_filters, blocks_down, etc.) "
            "were used for training."
        )

    if info.unexpected_keys:
        # Expected: deep supervision head keys from training (dsdepth>1)
        # Extra DS keys in dsdepth=2 training ckpt: up_layers.N.head.{weight,bias}
        ds_keys = [k for k in info.unexpected_keys if "ds_head" in k or "up_layers" in k]
        other_keys = [k for k in info.unexpected_keys if k not in ds_keys]
        if ds_keys:
            print(f"\n  ℹ️  Deep supervision keys ({len(ds_keys)}) — expected, ignored:")
            for k in ds_keys[:3]:
                print(f"    - {k}")
            if len(ds_keys) > 3:
                print(f"    ... and {len(ds_keys) - 3} more")
        if other_keys:
            print(f"\n  ⚠️  Unexpected keys ({len(other_keys)}) — ignored:")
            for k in other_keys[:5]:
                print(f"    - {k}")

    # Verify by re-extracting only the keys the inference model needs
    inference_state = {k: v for k, v in source_state.items() if k in model.state_dict()}
    model.load_state_dict(inference_state, strict=True)
    print("  ✅ All inference weights loaded (strict match on shared keys)")

    # ── Save in Seg-MoE format ──
    out_dir = Path(run_dir) / "checkpoints" / args.layer / f"fold{args.fold}" / seg_name
    out_dir.mkdir(parents=True, exist_ok=True)

    seg_moe_ckpt = {
        "model": model.state_dict(),
        "epoch": epoch,
        "global_step": -1,
        "best_metric": best_metric,
        "metrics": {},
        "source": "monai_segresnet_official",
        "source_path": str(source),
    }

    best_path = out_dir / "best.pt"
    last_path = out_dir / "last.pt"
    torch.save(seg_moe_ckpt, best_path)
    torch.save(seg_moe_ckpt, last_path)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Saved : {best_path}")
    print(f"  Saved : {last_path}")
    print(f"  Params: {n_params:,}")

    print("\n" + "=" * 60)
    print("✅ Import complete!")
    print("=" * 60)
    print()
    print("train_2d_experts.py --skip-if-done will auto-skip this expert")
    print("since best.pt now exists.")


if __name__ == "__main__":
    main()

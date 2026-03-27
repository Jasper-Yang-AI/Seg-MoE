#!/usr/bin/env python
"""
Import SwinUNETR official-training weights into Seg-MoE format.

与 import_nnunet_weights.py 类似, 但更简单:
  - SwinUNETR 直接使用 MONAI 的 SwinUNETR class, 无需 wrapper
  - state_dict keys 与 Seg-MoE build_expert() 构建的模型**完全相同**
  - 不需要 key 前缀变换

Usage:
    # 单折导入
    python scripts/monai/import_swinunetr_weights.py \\
        --source runs/swinunetr_official_msd03/fold0/best_model.pt \\
        --exp  configs/2d/exp/exp_msd_task03_liver.yaml \\
        --models configs/2d/models.yaml --fold 0

    # 5 折批量导入 (PowerShell)
    foreach ($fold in 0..4) {
        python scripts/monai/import_swinunetr_weights.py `
            --source runs/swinunetr_official_msd03/fold$fold/best_model.pt `
            --exp  configs/2d/exp/exp_msd_task03_liver.yaml `
            --models configs/2d/models.yaml --fold $fold
    }

导入后 train_2d_experts.py --skip-if-done 会自动跳过 SwinUNETR
(因为 best.pt 已存在).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import torch

from seg_moe.data.indexing import infer_num_classes
from seg_moe.models.factory_2d import build_expert, expert_name, list_experts
from seg_moe.utils.checkpoint import load_trusted_torch_checkpoint
from seg_moe.utils.config import load_config, resolve_run_dir


def _validate_unique_expert_names(experts: List[Dict[str, Any]]) -> None:
    names = [expert_name(e) for e in experts]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(
            f"Duplicate expert names in models.yaml: {duplicates}. "
            "Each expert name must be unique to avoid checkpoint path conflicts."
        )


def _resolve_swin_expert(
    experts: List[Dict[str, Any]],
    requested_name: str | None,
) -> tuple[Dict[str, Any], str]:
    swin_types = {"monai_swin_unetr", "swinunetr"}
    candidates = [e for e in experts if e.get("type", "").lower() in swin_types]

    if requested_name:
        for e in candidates:
            if expert_name(e) == requested_name:
                return e, requested_name
        raise ValueError(
            f"Requested expert '{requested_name}' not found among SwinUNETR experts: "
            f"{[expert_name(e) for e in candidates]}"
        )

    if len(candidates) == 1:
        e = candidates[0]
        return e, expert_name(e)

    if len(candidates) == 0:
        raise ValueError("No SwinUNETR expert found in models.yaml (experts_v2)")

    raise ValueError(
        "Multiple SwinUNETR experts found. Please specify --expert-name explicitly. "
        f"Candidates: {[expert_name(e) for e in candidates]}"
    )


def _assert_no_checkpoint_conflict(out_dir: Path, source: Path, overwrite: bool) -> None:
    if overwrite:
        return
    best_path = out_dir / "best.pt"
    if not best_path.exists():
        return

    existing = torch.load(best_path, map_location="cpu", weights_only=False)
    existing_source = str(existing.get("source_path", ""))
    if existing_source and Path(existing_source).resolve() != source.resolve():
        raise RuntimeError(
            "Checkpoint target conflict detected. Existing checkpoint source differs:\n"
            f"  target: {best_path}\n"
            f"  existing source: {existing_source}\n"
            f"  new source     : {source}\n"
            "Use --expert-name to choose another expert, or --overwrite to replace."
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Import SwinUNETR official weights into Seg-MoE"
    )
    ap.add_argument("--source", required=True, help="Official training checkpoint (best_model.pt)")
    ap.add_argument("--exp", required=True, help="Experiment config YAML")
    ap.add_argument("--models", required=True, help="Models config YAML")
    ap.add_argument("--fold", type=int, required=True, help="Fold index (0-4)")
    ap.add_argument("--layer", default="layer1", choices=["layer1", "layer2"])
    ap.add_argument("--dataset-config", default=None)
    ap.add_argument("--expert-name", default=None, help="Expert name in models.yaml (recommended)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing checkpoint if target exists")
    args = ap.parse_args()

    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(f"Source checkpoint not found: {source}")

    # ── Load configs ──
    exp_cfg = load_config(args.exp)
    models_cfg = load_config(args.models)
    dataset_cfg = load_config(args.dataset_config or exp_cfg["dataset"]["config"])
    run_dir = resolve_run_dir(exp_cfg)

    # ── Find SwinUNETR expert config ──
    experts = list_experts(models_cfg)
    _validate_unique_expert_names(experts)
    swin_cfg, swin_name = _resolve_swin_expert(experts, args.expert_name)

    num_classes = infer_num_classes(dataset_cfg)
    in_channels = 3  # ImageNet-normalised

    # ── Banner ──
    print()
    print("=" * 60)
    print("Importing SwinUNETR weights into Seg-MoE")
    print("=" * 60)
    print(f"  Source  : {source}")
    print(f"  Expert  : {swin_name}")
    print(f"  Fold    : {args.fold}")
    print(f"  Layer   : {args.layer}")

    # ── Load source checkpoint ──
    ckpt = load_trusted_torch_checkpoint(source, map_location="cpu")
    source_state = ckpt.get("model", ckpt.get("state_dict", ckpt))
    epoch = ckpt.get("epoch", -1)
    best_metric = ckpt.get("best_metric", -1.0)

    print(f"  Epoch   : {epoch}")
    if best_metric >= 0:
        print(f"  Dice    : {best_metric:.4f}")

    # ── Build Seg-MoE model & validate compatibility ──
    model = build_expert(swin_cfg, in_channels=in_channels, num_classes=num_classes)

    info = model.load_state_dict(source_state, strict=False)
    if info.missing_keys:
        print(f"\n  ❌ Missing keys ({len(info.missing_keys)}):")
        for k in info.missing_keys[:10]:
            print(f"    - {k}")
        raise RuntimeError(
            f"Weight import failed: {len(info.missing_keys)} missing keys. "
            "Architecture mismatch between source checkpoint and models.yaml. "
            "Ensure the same SwinUNETR params were used for training."
        )
    if info.unexpected_keys:
        print(f"\n  ⚠️  Unexpected keys ({len(info.unexpected_keys)}) — ignored:")
        for k in info.unexpected_keys[:5]:
            print(f"    - {k}")

    # ── Strict reload to guarantee correctness ──
    model.load_state_dict(source_state, strict=True)
    print("  ✅ All weights loaded (strict match)")

    # ── Save in Seg-MoE format ──
    out_dir = Path(run_dir) / "checkpoints" / args.layer / f"fold{args.fold}" / swin_name
    _assert_no_checkpoint_conflict(out_dir, source, overwrite=args.overwrite)
    out_dir.mkdir(parents=True, exist_ok=True)

    seg_moe_ckpt = {
        "model": model.state_dict(),
        "epoch": epoch,
        "global_step": -1,
        "best_metric": best_metric,
        "metrics": {},
        "source": "monai_swinunetr_official",
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

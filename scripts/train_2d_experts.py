from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from seg_moe.data.dataset_2d import SegmentationDataset2D
from seg_moe.data.indexing import infer_image_channels, infer_num_classes
from seg_moe.models.factory_2d import build_smp_model, expert_name, list_experts
from seg_moe.training.engine import train_model
from seg_moe.utils.config import apply_debug_overrides, load_config, merge_configs, resolve_run_dir
from seg_moe.utils.io import ensure_dir, load_jsonl
from seg_moe.utils.seed import seed_everything

# SOTA models support
try:
    from seg_moe.models.factory_sota import build_sota_model, expert_name_sota, list_sota_experts
    SOTA_AVAILABLE = True
except ImportError:
    SOTA_AVAILABLE = False


def _load_splits(dataset_cfg: dict) -> list[dict]:
    splits_dir = Path(dataset_cfg["paths"]["splits_dir"])
    # choose split file by split.type
    stype = dataset_cfg["split"]["type"]
    if stype == "holdout20_then_5fold":
        path = splits_dir / "splits_holdout20_5fold.jsonl"
    elif stype == "train_5fold_test_fixed":
        path = splits_dir / "splits_train5fold_testfixed.jsonl"
    else:
        path = splits_dir / "splits_5fold.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Splits not found: {path}. Run prepare_* and scripts/make_splits.py first.")
    return load_jsonl(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--training", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--augs", required=True)
    ap.add_argument("--debug", default=None)
    ap.add_argument("--layer", choices=["layer1", "layer2"], default="layer1")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--dataset-config", default=None, help="Override dataset config YAML (instead of exp.dataset.config)")
    ap.add_argument(
        "--resume",
        default="none",
        help="Resume mode: none|last|best|/path/to/ckpt.pt (per-expert).",
    )
    ap.add_argument("--skip-if-done", action="store_true", help="Skip training if best checkpoint already exists")
    args = ap.parse_args()

    exp_cfg = load_config(args.exp)
    dataset_cfg = load_config(args.dataset_config or exp_cfg["dataset"]["config"])
    training_cfg = load_config(args.training)
    models_cfg = load_config(args.models)
    augs_cfg = load_config(args.augs)
    debug_cfg = load_config(args.debug) if args.debug else None

    merged = merge_configs(exp_cfg, {"training": training_cfg, "models": models_cfg, "augs": augs_cfg, "dataset_cfg": dataset_cfg})
    merged = apply_debug_overrides(merged, debug_cfg)

    seed = int(merged.get("seed", 42))
    det = merged.get("deterministic", {})
    seed_everything(seed, deterministic=bool(det.get("torch_deterministic", True)), cudnn_benchmark=bool(det.get("cudnn_benchmark", False)))

    run_dir = resolve_run_dir(exp_cfg)
    ensure_dir(run_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = infer_num_classes(dataset_cfg)
    in_channels = infer_image_channels(dataset_cfg)

    rows = _load_splits(dataset_cfg)
    fold = int(args.fold)

    train_split = f"train_fold{fold}"
    val_split = f"val_fold{fold}"

    train_rows = [r for r in rows if r.get("split") == train_split]
    val_rows = [r for r in rows if r.get("split") == val_split]

    limits = (merged.get("datalimits", {}) or {})
    train_limit = limits.get("limit_train_samples")
    val_limit = limits.get("limit_val_samples")

    train_ds = SegmentationDataset2D(train_rows, dataset_cfg, augs_cfg=augs_cfg, is_train=True, limit=train_limit)
    val_ds = SegmentationDataset2D(val_rows, dataset_cfg, augs_cfg=augs_cfg, is_train=False, limit=val_limit)

    bs = int((merged.get("training", {}) or {}).get("batch_size", training_cfg.get("batch_size", 8)))
    epochs = int((merged.get("training", {}) or {}).get("epochs", training_cfg.get("epochs", 300)))
    training_cfg = {**training_cfg, "batch_size": bs, "epochs": epochs}

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=int(training_cfg.get("dataloader", {}).get("num_workers", 4)), pin_memory=bool(training_cfg.get("dataloader", {}).get("pin_memory", True)))
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=int(training_cfg.get("dataloader", {}).get("num_workers", 4)), pin_memory=bool(training_cfg.get("dataloader", {}).get("pin_memory", True)))

    # Detect if using SOTA models or classic SMP models
    use_sota = SOTA_AVAILABLE and "sota_experts" in models_cfg
    
    if use_sota:
        print("=" * 60)
        print("Using SOTA model configuration (Swin-UNetR/nnUNet/VM-UNet)")
        print("=" * 60)
        
        sota_experts_cfg = list_sota_experts(models_cfg)
        for arch, config in sota_experts_cfg:
            if not config.get("enabled", True):
                print(f"Skipping disabled expert: {arch}")
                continue
            
            name = config.get("name", expert_name_sota(arch))
            tag = f"{args.layer}/fold{fold}/{name}"
            ckpt_dir = Path(run_dir) / "checkpoints" / args.layer / f"fold{fold}" / name
            best_ckpt = ckpt_dir / "best.pt"
            last_ckpt = ckpt_dir / "last.pt"
            
            if args.skip_if_done and best_ckpt.exists():
                print(f"Skip (exists): {best_ckpt}")
                continue

            resume_from = None
            if isinstance(args.resume, str) and args.resume.lower() in {"last", "best"}:
                resume_from = str(last_ckpt if args.resume.lower() == "last" else best_ckpt)
            elif isinstance(args.resume, str) and args.resume.lower() not in {"none", ""}:
                resume_from = args.resume

            # Build SOTA model
            print(f"\nBuilding SOTA model: {arch}")
            model = build_sota_model(
                arch=arch,
                in_channels=in_channels,
                classes=num_classes,
                config=config.get("config", {}),
                pretrained=config.get("pretrained", True),
            )
            
            # Apply architecture-specific training adjustments
            arch_training_cfg = training_cfg.copy()
            sota_training = models_cfg.get("sota_training", {})
            if arch in sota_training:
                lr_mult = sota_training[arch].get("lr_multiplier", 1.0)
                arch_training_cfg["lr"] = training_cfg["lr"] * lr_mult
                print(f"  LR adjusted: {training_cfg['lr']:.2e} -> {arch_training_cfg['lr']:.2e} (×{lr_mult})")
            
            print(f"Training SOTA expert: {tag} on {device}")
            train_model(
                model,
                train_loader,
                val_loader,
                num_classes=num_classes,
                training_cfg=arch_training_cfg,
                run_dir=run_dir,
                tag=tag,
                device=device,
                resume_from=resume_from,
            )
    else:
        # Classic SMP experts (9 experts: 3 arch × 3 backbones)
        print("=" * 60)
        print("Using classic SMP model configuration (UNet/LinkNet/FPN)")
        print("=" * 60)
        
        experts = list_experts(models_cfg)
        encoder_weights = models_cfg.get("smp", {}).get("encoder_weights", "imagenet")

        for arch, backbone in experts:
            tag = f"{args.layer}/fold{fold}/{expert_name(arch, backbone)}"
            ckpt_dir = Path(run_dir) / "checkpoints" / args.layer / f"fold{fold}" / expert_name(arch, backbone)
            best_ckpt = ckpt_dir / "best.pt"
            last_ckpt = ckpt_dir / "last.pt"
            if args.skip_if_done and best_ckpt.exists():
                print(f"Skip (exists): {best_ckpt}")
                continue

            resume_from = None
            if isinstance(args.resume, str) and args.resume.lower() in {"last", "best"}:
                resume_from = str(last_ckpt if args.resume.lower() == "last" else best_ckpt)
            elif isinstance(args.resume, str) and args.resume.lower() not in {"none", ""}:
                resume_from = args.resume

            model = build_smp_model(arch, backbone, in_channels=in_channels, classes=num_classes, encoder_weights=encoder_weights)
            print(f"Training expert: {tag} on {device}")
            train_model(
                model,
                train_loader,
                val_loader,
                num_classes=num_classes,
                training_cfg=training_cfg,
                run_dir=run_dir,
                tag=tag,
                device=device,
                resume_from=resume_from,
            )


if __name__ == "__main__":
    main()

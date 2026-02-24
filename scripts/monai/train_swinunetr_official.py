#!/usr/bin/env python
"""
SwinUNETR 2D — Official MONAI Recipe Training (standalone).

复现 Tang et al. "Self-Supervised Pre-Training of Swin Transformers
for 3D Medical Image Analysis" (CVPR 2022) 的官方训练策略, 适配 2D:

  Optimizer     : AdamW, lr = 1e-4, weight_decay = 1e-5
  Scheduler     : WarmupCosine (warmup 50 epochs, 300 epochs total)
  Loss          : DiceCELoss (softmax=True, to_onehot_y=True)
  Augmentation  : MONAI transforms — Flip, Rotate90, ScaleIntensity, ShiftIntensity
  Validation    : Mean Dice (foreground classes)

训练完成后使用 import_swinunetr_weights.py 导入权重到 Seg-MoE:
    python scripts/monai/import_swinunetr_weights.py \\
        --source runs/swinunetr_official_msd03/fold0/best_model.pt \\
        --exp  configs/2d/exp/exp_msd_task03_liver.yaml \\
        --models configs/2d/models.yaml --fold 0

Usage:
    # 单折训练
    python scripts/monai/train_swinunetr_official.py \\
        --exp  configs/2d/exp/exp_msd_task03_liver.yaml \\
        --models configs/2d/models.yaml \\
        --fold 0 --gpus 0,1

    # 5 折训练 (PowerShell)
    foreach ($fold in 0..4) {
        python scripts/monai/train_swinunetr_official.py `
            --exp  configs/2d/exp/exp_msd_task03_liver.yaml `
            --models configs/2d/models.yaml `
            --fold $fold --gpus 0,1
    }

Prerequisites:
    pip install monai>=1.3.0
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

# ── MONAI imports ──
from monai.losses import DiceCELoss
from monai.networks.nets import SwinUNETR
from monai.optimizers.lr_scheduler import WarmupCosineSchedule
from monai.transforms import (
    Compose,
    RandFlipd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
)

# ── Seg-MoE shared utilities (data splits / config / seed) ──
from seg_moe.utils.config import load_config
from seg_moe.utils.io import load_jsonl
from seg_moe.utils.seed import seed_everything

# ImageNet normalization (same as Seg-MoE inference pipeline)
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────
#  Dataset (self-contained, reads PNG, applies MONAI transforms)
# ─────────────────────────────────────────────────────────────────────
class _SwinUNETRDataset(Dataset):
    """2D PNG segmentation dataset for MONAI-style training.

    Normalisation pipeline matches Seg-MoE inference exactly:
        grayscale uint8 → float [0,1] → replicate to 3ch → ImageNet norm
    """

    def __init__(
        self,
        rows: List[Dict[str, Any]],
        dataset_cfg: Dict[str, Any],
        monai_transforms=None,
    ):
        self.rows = rows
        self.label_map = dataset_cfg.get("task", {}).get("label_map")
        self.image_size = tuple(dataset_cfg.get("input", {}).get("image_size", [256, 256]))
        self.transforms = monai_transforms

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]

        # ── Load grayscale PNG ──
        img = cv2.imread(row["image"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {row['image']}")
        mask = cv2.imread(row["mask"], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Cannot load mask: {row['mask']}")

        # ── Label remap ──
        if self.label_map:
            new_mask = np.zeros_like(mask)
            for src, dst in self.label_map.items():
                new_mask[mask == int(src)] = int(dst)
            mask = new_mask

        # ── Resize ──
        h, w = img.shape
        th, tw = self.image_size
        if h != th or w != tw:
            img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (tw, th), interpolation=cv2.INTER_NEAREST)

        # ── float [0,1], add channel dim → [1,H,W] ──
        img = (img.astype(np.float32) / 255.0)[np.newaxis, ...]
        mask = mask.astype(np.int64)[np.newaxis, ...]

        # ── MONAI spatial + intensity transforms ──
        if self.transforms is not None:
            data = self.transforms({"image": img, "label": mask})
            img, mask = data["image"], data["label"]

        # ── Grayscale → 3ch + ImageNet normalize (matches Seg-MoE inference) ──
        if isinstance(img, np.ndarray):
            img = np.repeat(img, 3, axis=0)  # [3,H,W]
            for c in range(3):
                img[c] = (img[c] - _IMAGENET_MEAN[c]) / _IMAGENET_STD[c]
            img = torch.from_numpy(img)
            mask = torch.from_numpy(mask).squeeze(0).long()
        else:
            img = img.repeat(3, 1, 1)
            for c in range(3):
                img[c] = (img[c] - _IMAGENET_MEAN[c]) / _IMAGENET_STD[c]
            mask = mask.squeeze(0).long()

        return img, mask, {"id": row.get("id", str(idx))}


# ─────────────────────────────────────────────────────────────────────
#  MONAI official augmentation recipe
# ─────────────────────────────────────────────────────────────────────
def _build_train_transforms() -> Compose:
    """Official MONAI augmentations for SwinUNETR (adapted for 2D)."""
    return Compose([
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
        RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
    ])


# ─────────────────────────────────────────────────────────────────────
#  Split loader (reuse project convention)
# ─────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────
#  Dice metric (foreground-only)
# ─────────────────────────────────────────────────────────────────────
def _mean_dice(logits: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    """Compute mean Dice for foreground classes."""
    preds = logits.argmax(dim=1)  # [B,H,W]
    dices: list[float] = []
    for c in range(1, num_classes):
        p = (preds == c).float()
        t = (targets == c).float()
        inter = (p * t).sum()
        union = p.sum() + t.sum()
        dices.append((2.0 * inter / (union + 1e-8)).item() if union > 0 else 1.0)
    return sum(dices) / len(dices) if dices else 0.0


# ─────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(
        description="SwinUNETR 2D — Official MONAI recipe training"
    )
    ap.add_argument("--exp", required=True, help="Experiment config YAML")
    ap.add_argument("--models", required=True, help="Models YAML (for SwinUNETR params)")
    ap.add_argument("--fold", type=int, default=0, help="CV fold index (0-4)")
    ap.add_argument("--epochs", type=int, default=300, help="Training epochs (default: 300)")
    ap.add_argument("--batch-size", type=int, default=16, help="Batch size per GPU")
    ap.add_argument("--lr", type=float, default=1e-4, help="Learning rate (official: 1e-4)")
    ap.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay (official: 1e-5)")
    ap.add_argument("--warmup-epochs", type=int, default=50, help="Warmup epochs (official: 50)")
    ap.add_argument("--gpus", type=str, default=None, help="GPU IDs, e.g. '0,1'")
    ap.add_argument("--amp", action="store_true", default=True, help="Enable AMP")
    ap.add_argument("--amp-dtype", choices=["float16", "bfloat16"], default="bfloat16")
    ap.add_argument("--val-interval", type=int, default=5, help="Validate every N epochs")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    ap.add_argument("--output-dir", type=str, default=None)
    ap.add_argument("--dataset-config", type=str, default=None)
    args = ap.parse_args()

    # ── Configs ──
    exp_cfg = load_config(args.exp)
    models_cfg = load_config(args.models)
    dataset_cfg = load_config(args.dataset_config or exp_cfg["dataset"]["config"])

    seed = args.seed or exp_cfg.get("seed", 42)
    seed_everything(seed)

    # ── GPU setup ──
    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Find SwinUNETR config in models.yaml ──
    swin_cfg = None
    for ec in models_cfg.get("experts_v2", []):
        if ec.get("type", "").lower() in ("monai_swin_unetr", "swinunetr"):
            swin_cfg = ec
            break
    if swin_cfg is None:
        raise ValueError("No SwinUNETR expert found in models.yaml (experts_v2)")

    params = swin_cfg.get("params", {})
    num_classes: int = dataset_cfg["task"]["num_classes"]
    in_channels = 3  # ImageNet-normalised grayscale → 3ch

    # ── Output directory ──
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path(f"runs/swinunetr_official_{dataset_cfg['name']}")
    fold_dir = out_dir / f"fold{args.fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    # ── Banner ──
    print()
    print("=" * 60)
    print("SwinUNETR 2D — Official MONAI Training Recipe")
    print("=" * 60)
    print(f"  Dataset      : {dataset_cfg['name']}")
    print(f"  Fold         : {args.fold}")
    print(f"  Epochs       : {args.epochs}")
    print(f"  Batch size   : {args.batch_size}")
    print(f"  LR           : {args.lr}")
    print(f"  Weight decay : {args.weight_decay}")
    print(f"  Warmup       : {args.warmup_epochs} epochs")
    print(f"  AMP          : {args.amp} ({args.amp_dtype})")
    print(f"  Seed         : {seed}")
    print(f"  Output       : {fold_dir}")
    print()

    # ── Build SwinUNETR ──
    import inspect
    valid_params = set(inspect.signature(SwinUNETR.__init__).parameters.keys())

    swin_kwargs: Dict[str, Any] = {
        "in_channels": in_channels,
        "out_channels": num_classes,
        "spatial_dims": params.get("spatial_dims", 2),
        "feature_size": params.get("feature_size", 48),
        "depths": params.get("depths", [2, 2, 2, 2]),
        "num_heads": params.get("num_heads", [3, 6, 12, 24]),
        "use_checkpoint": params.get("use_checkpoint", True),
    }
    if "img_size" in valid_params and "img_size" in params:
        swin_kwargs["img_size"] = tuple(params["img_size"])

    model = SwinUNETR(**swin_kwargs)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[SwinUNETR-2D] {n_params:,} parameters")

    # ── DataParallel (Windows compatible) ──
    if torch.cuda.device_count() > 1:
        print(f"  DataParallel on {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model.to(device)

    # ── Official optimizer ──
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # ── Official loss: DiceCE ──
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)

    # ── Data ──
    rows = _load_splits(dataset_cfg)
    train_rows = [r for r in rows if r.get("split") == f"train_fold{args.fold}"]
    val_rows = [r for r in rows if r.get("split") == f"val_fold{args.fold}"]
    print(f"  Train: {len(train_rows)},  Val: {len(val_rows)}")

    train_ds = _SwinUNETRDataset(train_rows, dataset_cfg, monai_transforms=_build_train_transforms())
    val_ds = _SwinUNETRDataset(val_rows, dataset_cfg, monai_transforms=None)

    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_dl = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # ── Official scheduler: WarmupCosine (step-level) ──
    steps_per_epoch = max(len(train_dl), 1)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs
    scheduler = WarmupCosineSchedule(
        optimizer, warmup_steps=warmup_steps, t_total=total_steps
    )

    # ── AMP ──
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
    use_scaler = args.amp and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    # ── TensorBoard ──
    writer = SummaryWriter(log_dir=str(fold_dir / "tb_logs"))

    # ── Resume ──
    start_epoch = 0
    best_dice = 0.0
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location="cpu")
        raw = model.module if isinstance(model, nn.DataParallel) else model
        raw.load_state_dict(ckpt["model"], strict=True)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt.get("epoch", 0)
        best_dice = ckpt.get("best_metric", 0.0)
        print(f"  Resumed from epoch {start_epoch}, best_dice={best_dice:.4f}")

    # ── Training loop ──
    print("\n" + "=" * 60)
    print("Training ...")
    print("=" * 60)

    global_step = start_epoch * steps_per_epoch

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        step_count = 0
        t0 = time.time()

        for batch_img, batch_mask, _ in train_dl:
            batch_img = batch_img.to(device)
            # DiceCELoss expects target [B, 1, H, W] (integer labels, will be one-hot'd)
            batch_mask = batch_mask.to(device).unsqueeze(1)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=args.amp, dtype=amp_dtype):
                logits = model(batch_img)
                loss = loss_fn(logits, batch_mask)

            if use_scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            scheduler.step()
            global_step += 1
            epoch_loss += loss.item()
            step_count += 1

        avg_loss = epoch_loss / max(step_count, 1)
        elapsed = time.time() - t0
        writer.add_scalar("train/loss", avg_loss, epoch)
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)

        # ── Validation ──
        if (epoch + 1) % args.val_interval == 0 or epoch == args.epochs - 1:
            model.eval()
            dice_sum, n_samples = 0.0, 0
            with torch.no_grad():
                for vi, vm, _ in val_dl:
                    vi = vi.to(device)
                    vm = vm.to(device)
                    with torch.amp.autocast("cuda", enabled=args.amp, dtype=amp_dtype):
                        vl = model(vi)
                    d = _mean_dice(vl, vm, num_classes)
                    dice_sum += d * vi.shape[0]
                    n_samples += vi.shape[0]
            val_dice = dice_sum / max(n_samples, 1)
            writer.add_scalar("val/dice", val_dice, epoch)

            raw = model.module if isinstance(model, nn.DataParallel) else model
            ckpt_payload = {
                "model": raw.state_dict(),
                "epoch": epoch + 1,
                "best_metric": max(best_dice, val_dice),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "config": {
                    "in_channels": in_channels,
                    "num_classes": num_classes,
                    "swinunetr_params": params,
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "warmup_epochs": args.warmup_epochs,
                    "epochs": args.epochs,
                },
                "source": "monai_swinunetr_official",
            }

            if val_dice > best_dice:
                best_dice = val_dice
                ckpt_payload["best_metric"] = best_dice
                torch.save(ckpt_payload, fold_dir / "best_model.pt")
                mark = " ◀ best"
            else:
                mark = ""

            torch.save(ckpt_payload, fold_dir / "latest_model.pt")
            print(
                f"  Epoch {epoch+1:>4d}/{args.epochs} | loss={avg_loss:.4f} | "
                f"val_dice={val_dice:.4f}{mark} | "
                f"lr={optimizer.param_groups[0]['lr']:.2e} | {elapsed:.1f}s"
            )
        else:
            print(
                f"  Epoch {epoch+1:>4d}/{args.epochs} | loss={avg_loss:.4f} | "
                f"lr={optimizer.param_groups[0]['lr']:.2e} | {elapsed:.1f}s"
            )

    writer.close()

    # ── Save training log ──
    with open(fold_dir / "training_log.json", "w") as f:
        json.dump({
            "fold": args.fold, "epochs": args.epochs, "best_dice": best_dice,
            "lr": args.lr, "weight_decay": args.weight_decay,
            "warmup_epochs": args.warmup_epochs, "batch_size": args.batch_size,
            "seed": seed, "dataset": dataset_cfg["name"],
            "num_params": n_params,
        }, f, indent=2)

    # ── Done ──
    print("\n" + "=" * 60)
    print(f"✅ Training complete!  Best Dice = {best_dice:.4f}")
    print(f"   Best model : {fold_dir / 'best_model.pt'}")
    print("=" * 60)
    print()
    print("Next step — import weights into Seg-MoE:")
    print(f"  python scripts/monai/import_swinunetr_weights.py \\")
    print(f"    --source {fold_dir / 'best_model.pt'} \\")
    print(f"    --exp {args.exp} \\")
    print(f"    --models {args.models} --fold {args.fold}")


if __name__ == "__main__":
    main()

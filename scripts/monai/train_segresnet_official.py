#!/usr/bin/env python
"""
SegResNet 2D — Official MONAI Auto3DSeg Recipe Training (standalone).

复现 MONAI Auto3DSeg SegResNet2D 官方训练策略:

  Network       : SegResNetDS (deep supervision, dsdepth=2)
  Optimizer     : AdamW, lr = 2e-4, weight_decay = 1e-5
  Scheduler     : WarmupCosineSchedule (warmup 3 epochs, epoch-level)
  Loss          : DeepSupervisionLoss(DiceCELoss(squared_pred, batch))
  Augmentation  : MONAI transforms — Affine, Flip, GaussianSmooth, ScaleIntensity,
                  ShiftIntensity, GaussianNoise
  Validation    : Mean Dice (foreground classes)

Official reference:
  - MONAI Auto3DSeg algorithm_templates/segresnet2d
  - Myronenko A. "3D MRI brain tumor segmentation using autoencoder
    regularization" (MICCAI BrainLes 2018)

训练完成后使用 import_segresnet_weights.py 导入权重到 Seg-MoE:
    python scripts/monai/import_segresnet_weights.py \\
        --source runs/segresnet_official_msd_task03_liver/fold0/best_model.pt \\
        --exp  configs/2d/exp/exp_msd_task03_liver.yaml \\
        --models configs/2d/models.yaml --fold 0

Usage:
    # 单折训练
    python scripts/monai/train_segresnet_official.py \\
        --exp  configs/2d/exp/exp_msd_task03_liver.yaml \\
        --models configs/2d/models.yaml \\
        --fold 0 --gpus 0,1

    # 5 折训练 (PowerShell)
    foreach ($fold in 0..4) {
        python scripts/monai/train_segresnet_official.py `
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
import warnings
from pathlib import Path
from typing import Any, Dict, List

# ── Suppress Windows-specific noise warnings ──
# PyTorch Windows builds exclude NCCL; DataParallel still works via shared memory.
warnings.filterwarnings("ignore", message=".*NCCL.*")
# nll_loss2d has no deterministic CUDA path; suppress warn_only chatter during training.
warnings.filterwarnings("ignore", message=".*nll_loss2d.*deterministic.*")

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

# ── MONAI imports ──
from monai.losses import DiceCELoss
from monai.networks.nets import SegResNetDS
from monai.optimizers.lr_scheduler import WarmupCosineSchedule
from monai.transforms import (
    Compose,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandAffined,
)

# ── Seg-MoE shared utilities (data splits / config / seed) ──
from seg_moe.utils.config import load_config
from seg_moe.utils.io import load_jsonl
from seg_moe.utils.seed import seed_everything

# ImageNet normalization (same as Seg-MoE inference pipeline)
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _deep_supervision_loss(base_loss, logits, target: torch.Tensor) -> torch.Tensor:
    """Deep supervision loss with safe target resizing on Windows/PyTorch.

    Avoids MONAI DeepSupervisionLoss internal interpolate(Long) path, which may
    trigger NotImplementedError on some PyTorch builds.
    """
    if not isinstance(logits, (list, tuple)):
        return base_loss(logits, target)

    num_levels = len(logits)
    weights = [1.0 / (2 ** i) for i in range(num_levels)]
    norm = sum(weights)
    weights = [w / norm for w in weights]

    total = logits[0].new_tensor(0.0)
    for level, pred in enumerate(logits):
        tgt = target
        if pred.shape[2:] != target.shape[2:]:
            tgt = F.interpolate(target.float(), size=pred.shape[2:], mode="nearest").long()
        total = total + weights[level] * base_loss(pred, tgt)
    return total


# ─────────────────────────────────────────────────────────────────────
#  Dataset (self-contained, reads PNG, applies MONAI transforms)
# ─────────────────────────────────────────────────────────────────────
class _SegResNetDataset(Dataset):
    """2D PNG segmentation dataset for MONAI-style training.

    Normalisation pipeline matches Seg-MoE inference exactly:
      - RGB input (multi-modal, e.g. prostate):  RGB uint8 → float [0,1] → ImageNet norm
      - Grayscale input (single-modal, e.g. CT): grayscale uint8 → float [0,1] → replicate to 3ch → ImageNet norm
    Auto-detects from dataset_cfg["input"]["image_channels"].
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
        self.image_channels = int(dataset_cfg.get("input", {}).get("image_channels", 1))
        self.transforms = monai_transforms

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        image_path = row.get("image") or row.get("image_path")
        mask_path = row.get("mask") or row.get("mask_path")
        if image_path is None:
            raise KeyError(f"Missing image path key in row. Expected 'image' or 'image_path'. Row keys: {list(row.keys())}")
        if mask_path is None:
            raise KeyError(f"Missing mask path key in row. Expected 'mask' or 'mask_path'. Row keys: {list(row.keys())}")

        # ── Load image (auto-detect grayscale vs RGB from config) ──
        if self.image_channels >= 3:
            raw = cv2.imread(str(image_path), cv2.IMREAD_COLOR)  # BGR
            if raw is None:
                raise FileNotFoundError(f"Cannot load image: {image_path}")
            img = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)  # [H,W,3]
        else:
            img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)  # [H,W]
            if img is None:
                raise FileNotFoundError(f"Cannot load image: {image_path}")
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Cannot load mask: {mask_path}")

        # ── Label remap ──
        if self.label_map:
            new_mask = np.zeros_like(mask)
            for src, dst in self.label_map.items():
                new_mask[mask == int(src)] = int(dst)
            mask = new_mask

        # ── Resize ──
        th, tw = self.image_size
        if self.image_channels >= 3:
            h, w, _ = img.shape
            if h != th or w != tw:
                img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_LINEAR)
                mask = cv2.resize(mask, (tw, th), interpolation=cv2.INTER_NEAREST)
            # float [0,1], channel-first → [3,H,W]
            img = (img.astype(np.float32) / 255.0).transpose(2, 0, 1)
        else:
            h, w = img.shape
            if h != th or w != tw:
                img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_LINEAR)
                mask = cv2.resize(mask, (tw, th), interpolation=cv2.INTER_NEAREST)
            # float [0,1], add channel dim → [1,H,W]
            img = (img.astype(np.float32) / 255.0)[np.newaxis, ...]
        mask = mask.astype(np.int64)[np.newaxis, ...]

        # ── MONAI spatial + intensity transforms ──
        if self.transforms is not None:
            data = self.transforms({"image": img, "label": mask})
            img, mask = data["image"], data["label"]

        # ── To 3ch + ImageNet normalize ──
        if isinstance(img, np.ndarray):
            if img.shape[0] == 1:
                img = np.repeat(img, 3, axis=0)  # grayscale → 3ch
            for c in range(3):
                img[c] = (img[c] - _IMAGENET_MEAN[c]) / _IMAGENET_STD[c]
            img = torch.from_numpy(img)
            mask = torch.from_numpy(mask).squeeze(0).long()
        else:
            if img.shape[0] == 1:
                img = img.repeat(3, 1, 1)
            for c in range(3):
                img[c] = (img[c] - _IMAGENET_MEAN[c]) / _IMAGENET_STD[c]
            mask = mask.squeeze(0).long()

        return img, mask, {"id": row.get("id", str(idx))}


# ─────────────────────────────────────────────────────────────────────
#  MONAI official augmentation recipe (Auto3DSeg SegResNet2D)
# ─────────────────────────────────────────────────────────────────────
def _build_train_transforms(image_size: tuple) -> Compose:
    """Official MONAI augmentations for SegResNet2D (Auto3DSeg template)."""
    return Compose([
        # Spatial: affine (rotation + scaling)
        RandAffined(
            keys=["image", "label"],
            prob=0.2,
            rotate_range=0.26,
            scale_range=0.2,
            mode=["bilinear", "nearest"],
            spatial_size=tuple(image_size),
            padding_mode="border",
        ),
        # Flip
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        # Gaussian smooth
        RandGaussianSmoothd(
            keys=["image"], prob=0.2,
            sigma_x=[0.5, 1.0], sigma_y=[0.5, 1.0],
        ),
        # Intensity
        RandScaleIntensityd(keys=["image"], factors=0.3, prob=0.5),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
        # Gaussian noise
        RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.1),
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
        description="SegResNet 2D — Official MONAI Auto3DSeg recipe training"
    )
    ap.add_argument("--exp", required=True, help="Experiment config YAML")
    ap.add_argument("--models", required=True, help="Models YAML (for SegResNet params)")
    ap.add_argument("--fold", type=int, default=0, help="CV fold index (0-4)")
    ap.add_argument("--epochs", type=int, default=300, help="Training epochs (default: 300)")
    ap.add_argument("--batch-size", type=int, default=16, help="Batch size per GPU")
    ap.add_argument("--lr", type=float, default=2e-4, help="Learning rate (official: 2e-4)")
    ap.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay (official: 1e-5)")
    ap.add_argument("--warmup-epochs", type=int, default=3, help="Warmup epochs (official: 3)")
    ap.add_argument("--dsdepth", type=int, default=2, help="Deep supervision depth (official: 2)")
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
    seed_everything(seed, deterministic=False)  # deterministic=True triggers nll_loss2d warnings every forward pass

    # ── GPU setup ──
    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Find SegResNet config in models.yaml ──
    seg_cfg = None
    for ec in models_cfg.get("experts_v2", []):
        if ec.get("type", "").lower() in ("monai_segresnet", "monai_segresnet_ds"):
            seg_cfg = ec
            break
    if seg_cfg is None:
        raise ValueError("No SegResNet expert found in models.yaml (experts_v2)")

    params = seg_cfg.get("params", {})
    num_classes: int = dataset_cfg["task"]["num_classes"]
    in_channels = 3  # Always 3ch: RGB multi-modal or grayscale→replicate
    # (RGB for multi-modal datasets like prostate, replicated for single-modal like CT)
    image_size = tuple(dataset_cfg.get("input", {}).get("image_size", [256, 256]))

    # ── Output directory ──
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path(f"runs/segresnet_official_{dataset_cfg['name']}")
    fold_dir = out_dir / f"fold{args.fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    # ── Banner ──
    print()
    print("=" * 60)
    print("SegResNet 2D — Official MONAI Auto3DSeg Training Recipe")
    print("=" * 60)
    print(f"  Dataset      : {dataset_cfg['name']}")
    print(f"  Fold         : {args.fold}")
    print(f"  Epochs       : {args.epochs}")
    print(f"  Batch size   : {args.batch_size}")
    print(f"  LR           : {args.lr}")
    print(f"  Weight decay : {args.weight_decay}")
    print(f"  Warmup       : {args.warmup_epochs} epochs")
    print(f"  DS depth     : {args.dsdepth}")
    print(f"  AMP          : {args.amp} ({args.amp_dtype})")
    print(f"  Seed         : {seed}")
    print(f"  Output       : {fold_dir}")
    print()

    # ── Build SegResNetDS (official architecture with deep supervision) ──
    model = SegResNetDS(
        spatial_dims=params.get("spatial_dims", 2),
        in_channels=in_channels,
        out_channels=num_classes,
        init_filters=params.get("init_filters", 32),
        blocks_down=params.get("blocks_down", [1, 2, 2, 4, 4]),
        blocks_up=params.get("blocks_up", None),
        dsdepth=args.dsdepth,
        norm=params.get("norm", "BATCH"),
        act=params.get("act", "relu"),
        upsample_mode=params.get("upsample_mode", "deconv"),
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[SegResNetDS-2D] {n_params:,} parameters (dsdepth={args.dsdepth})")

    # ── DataParallel (Windows compatible) ──
    if torch.cuda.device_count() > 1:
        print(f"  DataParallel on {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model.to(device)

    # ── Official optimizer: AdamW ──
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # ── Official loss: DiceCELoss (used via _deep_supervision_loss wrapper) ──
    # DeepSupervisionLoss is NOT used directly — _deep_supervision_loss() handles
    # target resizing manually to avoid interpolate(Long) NotImplementedError on Windows.
    base_loss = DiceCELoss(
        include_background=True,
        to_onehot_y=True,
        softmax=True,
        squared_pred=True,  # official Auto3DSeg setting
        smooth_nr=0,
        smooth_dr=1e-5,
        batch=True,
    )

    # ── Data ──
    rows = _load_splits(dataset_cfg)
    train_rows = [r for r in rows if r.get("split") == f"train_fold{args.fold}"]
    val_rows = [r for r in rows if r.get("split") == f"val_fold{args.fold}"]
    print(f"  Train: {len(train_rows)},  Val: {len(val_rows)}")

    train_ds = _SegResNetDataset(train_rows, dataset_cfg, monai_transforms=_build_train_transforms(image_size))
    val_ds = _SegResNetDataset(val_rows, dataset_cfg, monai_transforms=None)

    _use_persistent = args.num_workers > 0
    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=_use_persistent,
    )
    val_dl = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=_use_persistent,
    )

    # ── Official scheduler: WarmupCosine (epoch-level, matching Auto3DSeg Segmenter) ──
    # warmup_multiplier was added in MONAI >= 1.3; guard for older installs
    import inspect as _inspect
    _sched_sig = _inspect.signature(WarmupCosineSchedule.__init__).parameters
    _sched_kwargs: Dict[str, Any] = dict(
        warmup_steps=args.warmup_epochs,
        t_total=args.epochs,
    )
    if "warmup_multiplier" in _sched_sig:
        _sched_kwargs["warmup_multiplier"] = 0.1
    scheduler = WarmupCosineSchedule(optimizer, **_sched_kwargs)

    # ── AMP ──
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
    # GradScaler only needed for float16; bfloat16 has the same exponent range as float32
    # and does not suffer from underflow, so scaling is unnecessary (and misleading).
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

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        step_count = 0
        t0 = time.time()

        nan_steps = 0
        for batch_img, batch_mask, _ in train_dl:
            batch_img = batch_img.to(device)
            # DeepSupervisionLoss + DiceCELoss expects target [B, 1, H, W]
            batch_mask = batch_mask.to(device).unsqueeze(1)

            optimizer.zero_grad(set_to_none=True)

            # Forward pass in half precision for speed
            with torch.amp.autocast("cuda", enabled=args.amp, dtype=amp_dtype):
                logits = model(batch_img)

            # Loss in float32 — DiceCELoss(squared_pred, smooth_nr=0) is
            # numerically sensitive; bfloat16 (8-bit mantissa) causes NaN.
            if isinstance(logits, (list, tuple)):
                logits_fp32 = [l.float() for l in logits]
            else:
                logits_fp32 = logits.float()
            loss = _deep_supervision_loss(base_loss, logits_fp32, batch_mask)

            # NaN guard — skip corrupted steps instead of poisoning the model
            if not torch.isfinite(loss):
                nan_steps += 1
                if nan_steps >= 10:
                    raise RuntimeError(
                        f"Epoch {epoch+1}: {nan_steps} consecutive NaN steps — aborting. "
                        "Try reducing --lr or --batch-size."
                    )
                continue
            nan_steps = 0

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            step_count += 1

        # Epoch-level scheduler step (matching Auto3DSeg Segmenter)
        scheduler.step()

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
                    # At inference, take full-resolution output (first element)
                    if isinstance(vl, (list, tuple)):
                        vl = vl[0]
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
                    "segresnet_params": params,
                    "dsdepth": args.dsdepth,
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "warmup_epochs": args.warmup_epochs,
                    "epochs": args.epochs,
                },
                "source": "monai_segresnet_official",
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
            "dsdepth": args.dsdepth,
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
    print(f"  python scripts/monai/import_segresnet_weights.py \\")
    print(f"    --source {fold_dir / 'best_model.pt'} \\")
    print(f"    --exp {args.exp} \\")
    print(f"    --models {args.models} --fold {args.fold}")


if __name__ == "__main__":
    main()

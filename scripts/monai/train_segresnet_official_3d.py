#!/usr/bin/env python
"""
SegResNet 3D — Official MONAI Auto3DSeg Recipe Training (standalone).

复现 MONAI Auto3DSeg SegResNet 官方 3D 训练策略:

  Network       : SegResNetDS (dsdepth=2, spatial_dims=3)
  Optimizer     : AdamW, lr = 2e-4, weight_decay = 1e-5
  Scheduler     : WarmupCosineSchedule (warmup 3 epochs, epoch-level)
  Loss          : DeepSupervisionLoss(DiceCELoss(squared_pred, batch))
  Augmentation  : MONAI 3D transforms (Affine, Flip, Smooth, Intensity, Noise)
  Validation    : Sliding-window inference + Mean Dice (foreground)

训练完成后使用 import_segresnet_weights_3d.py 导入权重到 Seg-MoE:
    python scripts/monai/import_segresnet_weights_3d.py \\
        --source runs/segresnet_official_3d_prostate/fold0/best_model.pt \\
        --exp configs/3d/exp/exp_prostate_local_3d.yaml \\
        --models configs/3d/models_3d.yaml --fold 0

Usage:
    python scripts/monai/train_segresnet_official_3d.py \\
        --exp  configs/3d/exp/exp_prostate_local_3d.yaml \\
        --models configs/3d/models_3d.yaml \\
        --fold 0 --gpus 0

Hardware target: RTX 5090 32GB, BF16 AMP.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List

warnings.filterwarnings("ignore", message=".*NCCL.*")

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.networks.nets import SegResNetDS
from monai.optimizers.lr_scheduler import WarmupCosineSchedule
from monai.transforms import (
    Compose,
    RandAffined,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandShiftIntensityd,
)

from seg_moe.utils.config import load_config
from seg_moe.utils.io import load_jsonl
from seg_moe.utils.seed import seed_everything
from seg_moe.utils.spatial import parse_3d_size


# ─────────────────────────────────────────────────────────────────────
#  Deep supervision loss (safe for Windows)
# ─────────────────────────────────────────────────────────────────────
def _deep_supervision_loss(base_loss, logits, target: torch.Tensor) -> torch.Tensor:
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
#  3D NIfTI Volume Dataset
# ─────────────────────────────────────────────────────────────────────
class _VolumeDataset3D(Dataset):
    """Load multi-modal 3D NIfTI volumes + label for MONAI-style training."""

    def __init__(self, rows, dataset_cfg, roi_size=(128, 128, 64),
                 monai_transforms=None, is_train=True):
        self.rows = rows
        self.roi_size = roi_size
        self.transforms = monai_transforms
        self.is_train = is_train
        paths = dataset_cfg["paths"]
        raw_structure = dataset_cfg.get("raw_structure", {}) or {}
        raw_dir = paths.get("raw_dir")

        if paths.get("images_dir") and paths.get("labels_dir"):
            self.images_dir = Path(paths["images_dir"])
            self.labels_dir = Path(paths["labels_dir"])
        elif raw_dir:
            raw_root = Path(raw_dir)
            self.images_dir = raw_root / raw_structure.get("images_tr_dir", "imagesTr")
            self.labels_dir = raw_root / raw_structure.get("labels_tr_dir", "labelsTr")
        else:
            raise KeyError("dataset_cfg.paths must define either images_dir/labels_dir or raw_dir with raw_structure")

        self.suffixes = (
            raw_structure.get("modality_suffixes")
            or dataset_cfg.get("input", {}).get("modality_suffixes")
            or ["_0000.nii.gz", "_0001.nii.gz", "_0002.nii.gz"])

        if not self.images_dir.exists() or not self.labels_dir.exists():
            raise FileNotFoundError(
                f"3D dataset directories not found: images={self.images_dir}, labels={self.labels_dir}"
            )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        sid = str(row["id"])
        channels = []
        image_paths = row.get("image_paths")
        if image_paths:
            paths = [Path(p) for p in image_paths]
        else:
            paths = [self.images_dir / f"{sid}{sfx}" for sfx in self.suffixes]

        for path in paths:
            arr = sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.float32)
            lo, hi = np.percentile(arr, [0.5, 99.5])
            arr = np.clip(arr, lo, hi)
            arr = (arr - arr.mean()) / (arr.std() + 1e-8)
            channels.append(arr)
        img = np.stack(channels, axis=0)
        mask_path = Path(row.get("mask_path") or (self.labels_dir / f"{sid}.nii.gz"))
        mask = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path))).astype(np.int64)
        mask = mask[np.newaxis, ...]

        if self.transforms is not None:
            data = self.transforms({"image": img, "label": mask})
            img, mask = data["image"], data["label"]

        img, mask = self._pad_if_needed(img, mask)

        if self.is_train:
            img, mask = self._random_crop(img, mask)

        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img.copy()).float()
            mask = torch.from_numpy(mask.copy())
        mask = mask.squeeze(0).long()
        return img, mask, {"id": sid}

    def _pad_if_needed(self, img, mask):
        rd, rh, rw = self.roi_size
        _, d, h, w = img.shape
        pad_d = max(0, rd - d)
        pad_h = max(0, rh - h)
        pad_w = max(0, rw - w)

        if pad_d == 0 and pad_h == 0 and pad_w == 0:
            return img, mask

        pad_spec = (
            (0, 0),
            (pad_d // 2, pad_d - pad_d // 2),
            (pad_h // 2, pad_h - pad_h // 2),
            (pad_w // 2, pad_w - pad_w // 2),
        )

        if isinstance(img, np.ndarray):
            img = np.pad(img, pad_spec, mode="constant")
            mask = np.pad(mask, pad_spec, mode="constant")
        else:
            pad_flat = (
                pad_w // 2, pad_w - pad_w // 2,
                pad_h // 2, pad_h - pad_h // 2,
                pad_d // 2, pad_d - pad_d // 2,
            )
            img = torch.nn.functional.pad(img, pad_flat, mode="constant", value=0)
            mask = torch.nn.functional.pad(mask, pad_flat, mode="constant", value=0)

        return img, mask

    def _random_crop(self, img, mask):
        if isinstance(img, np.ndarray):
            C, D, H, W = img.shape
        else:
            C, D, H, W = img.shape
        rd, rh, rw = self.roi_size
        if isinstance(mask, np.ndarray) and np.random.rand() < 0.67:
            fg = np.argwhere(mask[0] > 0)
            if len(fg) > 0:
                c = fg[np.random.randint(len(fg))]
                pd = max(0, min(c[0] - rd//2, D - rd)) if D > rd else 0
                ph = max(0, min(c[1] - rh//2, H - rh)) if H > rh else 0
                pw = max(0, min(c[2] - rw//2, W - rw)) if W > rw else 0
            else:
                pd = np.random.randint(0, max(D-rd, 1)) if D > rd else 0
                ph = np.random.randint(0, max(H-rh, 1)) if H > rh else 0
                pw = np.random.randint(0, max(W-rw, 1)) if W > rw else 0
        else:
            pd = np.random.randint(0, max(D-rd, 1)) if D > rd else 0
            ph = np.random.randint(0, max(H-rh, 1)) if H > rh else 0
            pw = np.random.randint(0, max(W-rw, 1)) if W > rw else 0
        return img[:, pd:pd+rd, ph:ph+rh, pw:pw+rw], mask[:, pd:pd+rd, ph:ph+rh, pw:pw+rw]


# ─────────────────────────────────────────────────────────────────────
#  Official augmentation (Auto3DSeg recipe)
# ─────────────────────────────────────────────────────────────────────
def _build_train_transforms_3d() -> Compose:
    return Compose([
        RandAffined(keys=["image", "label"], prob=0.2,
                    rotate_range=(0.26, 0.26, 0.26), scale_range=(0.2, 0.2, 0.2),
                    mode=["bilinear", "nearest"], padding_mode="border"),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandGaussianSmoothd(keys=["image"], prob=0.2,
                            sigma_x=[0.5, 1.0], sigma_y=[0.5, 1.0], sigma_z=[0.5, 1.0]),
        RandScaleIntensityd(keys=["image"], factors=0.3, prob=0.5),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
        RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.1),
    ])


def _load_splits(dataset_cfg: dict) -> list[dict]:
    path = Path(dataset_cfg["paths"]["splits_dir"]) / "splits_train5fold_testfixed.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Splits not found: {path}. Run make_splits_3d.py first.")
    rows = load_jsonl(path)
    valid_rows = []
    dropped = 0
    for row in rows:
        image_paths = row.get("image_paths") or []
        mask_path = row.get("mask_path")
        if image_paths and mask_path and all(Path(p).exists() for p in image_paths) and Path(mask_path).exists():
            valid_rows.append(row)
        else:
            dropped += 1
    if dropped:
        print(f"[splits] Dropped {dropped} invalid rows with missing image/mask files from {path}")
    return valid_rows


def _mean_dice_3d(logits, targets, num_classes):
    preds = logits.argmax(dim=1)
    dices = []
    for c in range(1, num_classes):
        p = (preds == c).float()
        t = (targets == c).float()
        inter = (p * t).sum()
        union = p.sum() + t.sum()
        dices.append((2.0 * inter / (union + 1e-8)).item() if union > 0 else 1.0)
    return sum(dices) / len(dices) if dices else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="SegResNet 3D — Official Auto3DSeg recipe")
    ap.add_argument("--exp", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=2, help="RTX 5090: bs=2 for SegResNet 3D")
    ap.add_argument("--lr", type=float, default=2e-4, help="Official: 2e-4")
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--warmup-epochs", type=int, default=3, help="Official: 3")
    ap.add_argument("--dsdepth", type=int, default=2, help="Deep supervision depth (official: 2)")
    ap.add_argument("--gpus", type=str, default=None)
    ap.add_argument("--amp", action="store_true", default=True, help="Enable AMP")
    ap.add_argument("--amp-dtype", choices=["float16", "bfloat16"], default="bfloat16")
    ap.add_argument("--val-interval", type=int, default=5)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--output-dir", type=str, default=None)
    ap.add_argument("--dataset-config", type=str, default=None)
    args = ap.parse_args()

    exp_cfg = load_config(args.exp)
    models_cfg = load_config(args.models)
    dataset_cfg = load_config(args.dataset_config or exp_cfg["dataset"]["config"])
    seed = args.seed or exp_cfg.get("seed", 42)
    seed_everything(seed, deterministic=False)

    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Find SegResNet 3D config
    seg_cfg = None
    for ec in models_cfg.get("experts_3d", []):
        if ec.get("type", "").lower() in ("segresnet", "seg_resnet", "monai_segresnet"):
            seg_cfg = ec
            break
    if seg_cfg is None:
        raise ValueError("No SegResNet expert found in experts_3d")

    params = seg_cfg.get("params", {})
    num_classes = int(dataset_cfg["task"]["num_classes"])
    in_channels = int(dataset_cfg["input"].get("image_channels", 3))
    roi_size = parse_3d_size(dataset_cfg["input"].get("roi_size", [128, 128, 64]))

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path(f"runs/segresnet_official_3d_{dataset_cfg['name']}")
    fold_dir = out_dir / f"fold{args.fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 60)
    print("SegResNet 3D — Official MONAI Auto3DSeg Training Recipe")
    print("=" * 60)
    print(f"  Dataset      : {dataset_cfg['name']}")
    print(f"  Fold         : {args.fold}")
    print(f"  Epochs       : {args.epochs}")
    print(f"  Batch size   : {args.batch_size} (grad_accum={args.grad_accum} → eff={args.batch_size*args.grad_accum})")
    print(f"  LR           : {args.lr}")
    print(f"  DS depth     : {args.dsdepth}")
    print(f"  ROI size     : {roi_size}")
    print(f"  AMP          : {args.amp_dtype}")
    print(f"  Output       : {fold_dir}")
    print()

    # Build model
    model = SegResNetDS(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=num_classes,
        init_filters=params.get("init_filters", 32),
        blocks_down=params.get("blocks_down", [1, 2, 2, 4]),
        blocks_up=params.get("blocks_up", None),
        dsdepth=args.dsdepth,
        norm=params.get("norm", "INSTANCE"),
        act=params.get("act", "relu"),
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[SegResNetDS-3D] {n_params:,} parameters (dsdepth={args.dsdepth})")

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    base_loss = DiceCELoss(include_background=True, to_onehot_y=True, softmax=True,
                           squared_pred=True, smooth_nr=0, smooth_dr=1e-5, batch=True)

    rows = _load_splits(dataset_cfg)
    train_rows = [r for r in rows if r.get("split") == f"train_fold{args.fold}"]
    val_rows = [r for r in rows if r.get("split") == f"val_fold{args.fold}"]
    print(f"  Train: {len(train_rows)},  Val: {len(val_rows)}")

    train_ds = _VolumeDataset3D(train_rows, dataset_cfg, roi_size=roi_size,
                                monai_transforms=_build_train_transforms_3d(), is_train=True)
    val_ds = _VolumeDataset3D(val_rows, dataset_cfg, roi_size=roi_size,
                              monai_transforms=None, is_train=False)

    _persistent = args.num_workers > 0
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True, drop_last=True,
                          persistent_workers=_persistent)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True,
                        persistent_workers=_persistent)

    # Epoch-level warmup cosine (Auto3DSeg convention)
    import inspect as _inspect
    _sched_sig = _inspect.signature(WarmupCosineSchedule.__init__).parameters
    _sched_kwargs: Dict[str, Any] = dict(warmup_steps=args.warmup_epochs, t_total=args.epochs)
    if "warmup_multiplier" in _sched_sig:
        _sched_kwargs["warmup_multiplier"] = 0.1
    scheduler = WarmupCosineSchedule(optimizer, **_sched_kwargs)

    amp_enabled = bool(args.amp and device.type == "cuda")
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_enabled and amp_dtype == torch.float16))
    writer = SummaryWriter(log_dir=str(fold_dir / "tb_logs"))

    start_epoch, best_dice = 0, 0.0
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

    grad_accum = max(int(args.grad_accum), 1)

    print(f"\nTraining ({args.epochs} epochs) ...")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss, step_count = 0.0, 0
        accum_steps = 0
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)

        for si, (batch_img, batch_mask, _) in enumerate(train_dl):
            batch_img = batch_img.to(device)
            batch_mask = batch_mask.to(device).unsqueeze(1)

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
                logits = model(batch_img)
            if isinstance(logits, (list, tuple)):
                logits_fp32 = [l.float() for l in logits]
            else:
                logits_fp32 = logits.float()
            loss = _deep_supervision_loss(base_loss, logits_fp32, batch_mask) / grad_accum

            if not torch.isfinite(loss):
                continue

            scaler.scale(loss).backward()
            accum_steps += 1
            if accum_steps >= grad_accum:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                accum_steps = 0

            epoch_loss += loss.item() * grad_accum
            step_count += 1

        if accum_steps > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # Epoch-level scheduler step
        if step_count > 0:
            scheduler.step()
        avg_loss = epoch_loss / max(step_count, 1)
        elapsed = time.time() - t0
        writer.add_scalar("train/loss", avg_loss, epoch)

        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)

        if (epoch + 1) % args.val_interval == 0 or epoch == args.epochs - 1:
            model.eval()
            dice_sum, n_samples = 0.0, 0
            raw_model = model.module if isinstance(model, nn.DataParallel) else model
            with torch.no_grad():
                for vi, vm, _ in val_dl:
                    vi, vm = vi.to(device), vm.to(device)
                    with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
                        vl = sliding_window_inference(vi, roi_size, 2, raw_model,
                                                      overlap=0.5, mode="gaussian")
                    # vl may be list (DS); take first
                    if isinstance(vl, (list, tuple)):
                        vl = vl[0]
                    d = _mean_dice_3d(vl, vm, num_classes)
                    dice_sum += d
                    n_samples += 1
            val_dice = dice_sum / max(n_samples, 1)
            writer.add_scalar("val/dice", val_dice, epoch)

            raw = model.module if isinstance(model, nn.DataParallel) else model
            ckpt_payload = {
                "model": raw.state_dict(), "epoch": epoch + 1,
                "best_metric": max(best_dice, val_dice),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "config": {"in_channels": in_channels, "num_classes": num_classes,
                           "segresnet_params": params, "dsdepth": args.dsdepth},
                "source": "monai_segresnet_official_3d",
            }
            if val_dice > best_dice:
                best_dice = val_dice
                ckpt_payload["best_metric"] = best_dice
                torch.save(ckpt_payload, fold_dir / "best_model.pt")
                mark = " ◀ best"
            else:
                mark = ""
            torch.save(ckpt_payload, fold_dir / "latest_model.pt")
            print(f"  Epoch {epoch+1:>4d}/{args.epochs} | loss={avg_loss:.4f} | "
                  f"val_dice={val_dice:.4f}{mark} | lr={optimizer.param_groups[0]['lr']:.2e} | {elapsed:.1f}s")
        else:
            print(f"  Epoch {epoch+1:>4d}/{args.epochs} | loss={avg_loss:.4f} | "
                  f"lr={optimizer.param_groups[0]['lr']:.2e} | {elapsed:.1f}s")

    writer.close()
    with open(fold_dir / "training_log.json", "w") as f:
        json.dump({"fold": args.fold, "epochs": args.epochs, "best_dice": best_dice,
                   "lr": args.lr, "dsdepth": args.dsdepth, "dataset": dataset_cfg["name"],
                   "num_params": n_params}, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Training complete!  Best Dice = {best_dice:.4f}")
    print(f"  Best model : {fold_dir / 'best_model.pt'}")
    print(f"\nNext: import weights into Seg-MoE:")
    print(f"  python scripts/monai/import_segresnet_weights_3d.py \\")
    print(f"    --source {fold_dir / 'best_model.pt'} \\")
    print(f"    --exp {args.exp} --models {args.models} --fold {args.fold}")
    print("=" * 60)


if __name__ == "__main__":
    main()

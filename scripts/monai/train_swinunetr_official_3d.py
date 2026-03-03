#!/usr/bin/env python
"""
SwinUNETR 3D — Official MONAI Recipe Training (standalone).

复现 Tang et al. "Self-Supervised Pre-Training of Swin Transformers
for 3D Medical Image Analysis" (CVPR 2022) 的官方训练策略:

  Network       : SwinUNETR (spatial_dims=3, feature_size=48)
  Optimizer     : AdamW, lr = 1e-4, weight_decay = 1e-5
  Scheduler     : WarmupCosine (warmup 50 epochs, step-level)
  Loss          : DiceCELoss (softmax=True, to_onehot_y=True)
  Augmentation  : MONAI 3D transforms (Flip, Rotate90, Affine, Intensity)
  Validation    : Sliding-window inference + Mean Dice (foreground)

训练完成后使用 import_swinunetr_weights_3d.py 导入权重到 Seg-MoE:
    python scripts/monai/import_swinunetr_weights_3d.py \\
        --source runs/swinunetr_official_3d_prostate/fold0/best_model.pt \\
        --exp configs/3d/exp/exp_prostate_local_3d.yaml \\
        --models configs/3d/models_3d.yaml --fold 0

Usage:
    # 单折训练
    python scripts/monai/train_swinunetr_official_3d.py \\
        --exp  configs/3d/exp/exp_prostate_local_3d.yaml \\
        --models configs/3d/models_3d.yaml \\
        --fold 0 --gpus 0

    # 5 折训练 (PowerShell)
    foreach ($fold in 0..4) {
        python scripts/monai/train_swinunetr_official_3d.py `
            --exp  configs/3d/exp/exp_prostate_local_3d.yaml `
            --models configs/3d/models_3d.yaml `
            --fold $fold --gpus 0
    }

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
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.networks.nets import SwinUNETR
from monai.optimizers.lr_scheduler import WarmupCosineSchedule
from monai.transforms import (
    Compose,
    RandAffined,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
)

from seg_moe.utils.config import load_config
from seg_moe.utils.io import load_jsonl
from seg_moe.utils.seed import seed_everything


# ─────────────────────────────────────────────────────────────────────
#  3D NIfTI Volume Dataset
# ─────────────────────────────────────────────────────────────────────
class _VolumeDataset3D(Dataset):
    """Load multi-modal 3D NIfTI volumes + label for MONAI-style training.

    Normalisation: per-channel percentile clip → z-score.
    Training: random crop to roi_size; Validation: full volume.
    """

    def __init__(
        self,
        rows: List[Dict[str, Any]],
        dataset_cfg: Dict[str, Any],
        roi_size: tuple[int, ...] = (128, 128, 64),
        monai_transforms=None,
        is_train: bool = True,
    ):
        self.rows = rows
        self.cfg = dataset_cfg
        self.roi_size = roi_size
        self.transforms = monai_transforms
        self.is_train = is_train

        paths = dataset_cfg["paths"]
        self.images_dir = Path(paths["images_dir"])
        self.labels_dir = Path(paths["labels_dir"])
        self.suffixes = dataset_cfg["input"].get(
            "modality_suffixes", ["_0000.nii.gz", "_0001.nii.gz", "_0002.nii.gz"]
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        sid = str(row["id"])

        # Load multi-modal images
        channels = []
        for sfx in self.suffixes:
            path = self.images_dir / f"{sid}{sfx}"
            arr = sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.float32)
            # Percentile clip + z-score
            lo, hi = np.percentile(arr, [0.5, 99.5])
            arr = np.clip(arr, lo, hi)
            mean, std = arr.mean(), arr.std() + 1e-8
            arr = (arr - mean) / std
            channels.append(arr)
        img = np.stack(channels, axis=0)  # [C, D, H, W]

        mask_path = self.labels_dir / f"{sid}.nii.gz"
        mask = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path))).astype(np.int64)
        mask = mask[np.newaxis, ...]  # [1, D, H, W]

        # MONAI transforms
        if self.transforms is not None:
            data = self.transforms({"image": img, "label": mask})
            img, mask = data["image"], data["label"]

        # Random crop for training
        if self.is_train:
            img, mask = self._random_crop(img, mask)

        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img.copy())
            mask = torch.from_numpy(mask.copy())

        mask = mask.squeeze(0).long()  # [D, H, W]
        return img.float(), mask, {"id": sid}

    def _random_crop(self, img, mask):
        """RandCropByPosNegLabel-style: 2/3 foreground, 1/3 any."""
        import torch as _t
        if isinstance(img, np.ndarray):
            C, D, H, W = img.shape
        else:
            C, D, H, W = img.shape
        rd, rh, rw = self.roi_size

        if D <= rd:
            pd = 0
        elif isinstance(mask, np.ndarray) and np.random.rand() < 0.67:
            fg = np.argwhere(mask[0] > 0)
            if len(fg) > 0:
                idx = fg[np.random.randint(len(fg))]
                pd = max(0, min(idx[0] - rd // 2, D - rd))
            else:
                pd = np.random.randint(0, max(D - rd, 1))
        else:
            pd = np.random.randint(0, max(D - rd, 1))
        ph = np.random.randint(0, max(H - rh, 1)) if H > rh else 0
        pw = np.random.randint(0, max(W - rw, 1)) if W > rw else 0

        img = img[:, pd:pd+rd, ph:ph+rh, pw:pw+rw]
        mask = mask[:, pd:pd+rd, ph:ph+rh, pw:pw+rw]
        return img, mask


# ─────────────────────────────────────────────────────────────────────
#  Official MONAI 3D augmentation
# ─────────────────────────────────────────────────────────────────────
def _build_train_transforms_3d() -> Compose:
    return Compose([
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(0, 1)),
        RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
    ])


# ─────────────────────────────────────────────────────────────────────
#  Split loader
# ─────────────────────────────────────────────────────────────────────
def _load_splits(dataset_cfg: dict) -> list[dict]:
    splits_dir = Path(dataset_cfg["paths"]["splits_dir"])
    path = splits_dir / "splits_train5fold_testfixed.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Splits not found: {path}. Run make_splits_3d.py first.")
    return load_jsonl(path)


def _mean_dice_3d(logits: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    preds = logits.argmax(dim=1)
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
    ap = argparse.ArgumentParser(description="SwinUNETR 3D — Official MONAI recipe training")
    ap.add_argument("--exp", required=True, help="Experiment config YAML")
    ap.add_argument("--models", required=True, help="Models config YAML")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=2, help="RTX 5090: bs=2 for SwinUNETR 3D")
    ap.add_argument("--lr", type=float, default=1e-4, help="Official: 1e-4")
    ap.add_argument("--weight-decay", type=float, default=1e-5, help="Official: 1e-5")
    ap.add_argument("--warmup-epochs", type=int, default=50, help="Official: 50")
    ap.add_argument("--gpus", type=str, default=None)
    ap.add_argument("--amp-dtype", choices=["float16", "bfloat16"], default="bfloat16")
    ap.add_argument("--val-interval", type=int, default=5)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=2, help="Effective BS = batch-size * grad-accum")
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

    # Find SwinUNETR 3D config
    swin_cfg = None
    for ec in models_cfg.get("experts_3d", []):
        if ec.get("type", "").lower() in ("swin_unetr", "swinunetr"):
            swin_cfg = ec
            break
    if swin_cfg is None:
        raise ValueError("No SwinUNETR expert found in experts_3d")

    params = swin_cfg.get("params", {})
    num_classes = int(dataset_cfg["task"]["num_classes"])
    in_channels = int(dataset_cfg["input"].get("image_channels", 3))
    roi_size = tuple(int(x) for x in dataset_cfg["input"].get("roi_size", [128, 128, 64]))

    # Output
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path(f"runs/swinunetr_official_3d_{dataset_cfg['name']}")
    fold_dir = out_dir / f"fold{args.fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    # Banner
    print()
    print("=" * 60)
    print("SwinUNETR 3D — Official MONAI Training Recipe")
    print("=" * 60)
    print(f"  Dataset      : {dataset_cfg['name']}")
    print(f"  Fold         : {args.fold}")
    print(f"  Epochs       : {args.epochs}")
    print(f"  Batch size   : {args.batch_size} (grad_accum={args.grad_accum} → eff={args.batch_size*args.grad_accum})")
    print(f"  LR           : {args.lr}")
    print(f"  ROI size     : {roi_size}")
    print(f"  AMP          : {args.amp_dtype}")
    print(f"  Output       : {fold_dir}")
    print()

    # Build model
    import inspect
    valid_params = set(inspect.signature(SwinUNETR.__init__).parameters.keys())
    swin_kwargs: Dict[str, Any] = {
        "in_channels": in_channels,
        "out_channels": num_classes,
        "spatial_dims": 3,
        "feature_size": params.get("feature_size", 48),
        "depths": params.get("depths", [2, 2, 2, 2]),
        "num_heads": params.get("num_heads", [3, 6, 12, 24]),
        "use_checkpoint": params.get("use_checkpoint", True),
    }
    if "patch_size" in valid_params:
        swin_kwargs["patch_size"] = params.get("patch_size", 2)

    model = SwinUNETR(**swin_kwargs)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[SwinUNETR-3D] {n_params:,} parameters")

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    # Official optimizer & loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)

    # Data
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

    # Scheduler (step-level warmup cosine)
    steps_per_epoch = max(len(train_dl), 1)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs
    scheduler = WarmupCosineSchedule(optimizer, warmup_steps=warmup_steps, t_total=total_steps)

    # AMP
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))

    writer = SummaryWriter(log_dir=str(fold_dir / "tb_logs"))

    # Resume
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

    grad_accum = args.grad_accum

    print(f"\nTraining ({args.epochs} epochs) ...")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        step_count = 0
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)

        for si, (batch_img, batch_mask, _) in enumerate(train_dl):
            batch_img = batch_img.to(device)
            batch_mask = batch_mask.to(device).unsqueeze(1)  # [B,1,D,H,W]

            with torch.amp.autocast("cuda", dtype=amp_dtype):
                logits = model(batch_img)
            loss = loss_fn(logits.float(), batch_mask) / grad_accum

            if not torch.isfinite(loss):
                continue

            scaler.scale(loss).backward()
            if (si + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            scheduler.step()
            epoch_loss += loss.item() * grad_accum
            step_count += 1

        avg_loss = epoch_loss / max(step_count, 1)
        elapsed = time.time() - t0
        writer.add_scalar("train/loss", avg_loss, epoch)
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)

        # Validation
        if (epoch + 1) % args.val_interval == 0 or epoch == args.epochs - 1:
            model.eval()
            dice_sum, n_samples = 0.0, 0
            raw_model = model.module if isinstance(model, nn.DataParallel) else model
            with torch.no_grad():
                for vi, vm, _ in val_dl:
                    vi = vi.to(device)
                    vm = vm.to(device)
                    with torch.amp.autocast("cuda", dtype=amp_dtype):
                        vl = sliding_window_inference(
                            vi, roi_size, 2, raw_model,
                            overlap=0.5, mode="gaussian"
                        )
                    d = _mean_dice_3d(vl, vm, num_classes)
                    dice_sum += d
                    n_samples += 1
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
                    "in_channels": in_channels, "num_classes": num_classes,
                    "swinunetr_params": params, "lr": args.lr,
                    "weight_decay": args.weight_decay, "warmup_epochs": args.warmup_epochs,
                },
                "source": "monai_swinunetr_official_3d",
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
        json.dump({
            "fold": args.fold, "epochs": args.epochs, "best_dice": best_dice,
            "lr": args.lr, "weight_decay": args.weight_decay,
            "warmup_epochs": args.warmup_epochs, "batch_size": args.batch_size,
            "grad_accum": args.grad_accum, "seed": seed,
            "dataset": dataset_cfg["name"], "num_params": n_params,
        }, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Training complete!  Best Dice = {best_dice:.4f}")
    print(f"  Best model : {fold_dir / 'best_model.pt'}")
    print(f"\nNext: import weights into Seg-MoE:")
    print(f"  python scripts/monai/import_swinunetr_weights_3d.py \\")
    print(f"    --source {fold_dir / 'best_model.pt'} \\")
    print(f"    --exp {args.exp} --models {args.models} --fold {args.fold}")
    print("=" * 60)


if __name__ == "__main__":
    main()

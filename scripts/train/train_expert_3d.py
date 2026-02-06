#!/usr/bin/env python
"""
Train a single 3D expert for Seg-MoE.

Usage:
    python scripts/train/train_expert_3d.py \
        --exp   configs/3d/exp_msd03_liver.yaml \
        --expert segresnet \
        --fold 0 \
        --gpus 0,1

Supports: segresnet | swin_unetr | nnunet
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from seg_moe.models.experts.factory import ExpertFactory
from seg_moe.training.losses import ce_plus_dice
from seg_moe.utils.config import load_config
from seg_moe.utils.io import ensure_dir
from seg_moe.utils.seed import seed_everything


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _get_device(gpus: str) -> tuple[torch.device, list[int]]:
    if not torch.cuda.is_available():
        return torch.device("cpu"), []
    ids = [int(g) for g in gpus.split(",") if g.strip()]
    if not ids:
        ids = list(range(torch.cuda.device_count()))
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in ids)
    return torch.device("cuda:0"), ids


def _build_dummy_loaders(num_classes: int, patch_size: tuple, batch_size: int):
    """Build tiny random DataLoaders for smoke-test / debug runs."""
    from torch.utils.data import TensorDataset

    N = max(batch_size * 4, 8)
    D, H, W = patch_size
    imgs = torch.randn(N, 1, D, H, W)
    masks = torch.randint(0, num_classes, (N, D, H, W))
    ds = TensorDataset(imgs, masks)
    train_dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    return train_dl, val_dl


def _sliding_window_inference(model, x, roi_size, sw_batch_size=4, overlap=0.5, mode="gaussian"):
    """MONAI sliding-window inference wrapper for validation."""
    try:
        from monai.inferers import sliding_window_inference
        return sliding_window_inference(x, roi_size, sw_batch_size, model, overlap=overlap, mode=mode)
    except ImportError:
        # Fallback: direct forward (works if input == roi)
        return model(x)


# ---------------------------------------------------------------------------
#  Main training loop
# ---------------------------------------------------------------------------

def train_expert(
    expert: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int,
    training_cfg: Dict[str, Any],
    run_dir: Path,
    tag: str,
    device: torch.device,
    gpu_ids: list[int],
) -> dict:
    """Train a single 3D expert.

    Returns dict with best_metric, best_ckpt, last_ckpt.
    """
    ckpt_dir = ensure_dir(run_dir / "checkpoints" / tag)

    lr = float(training_cfg["lr"])
    epochs = int(training_cfg["epochs"])
    grad_accum = int(training_cfg.get("gradient_accumulation_steps", 1))

    # AMP
    amp_cfg = training_cfg.get("amp", {}) or {}
    amp_enabled = bool(amp_cfg.get("enabled", False)) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if str(amp_cfg.get("dtype", "float16")).lower() in ("bf16", "bfloat16") else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    # DataParallel
    expert.to(device)
    if len(gpu_ids) > 1 and torch.cuda.device_count() > 1:
        expert = nn.DataParallel(expert, device_ids=list(range(len(gpu_ids))))
        print(f"  DataParallel on {len(gpu_ids)} GPUs")

    # Optimizer
    opt_cfg = training_cfg.get("optimizer", {}) or {}
    opt_name = str(opt_cfg.get("name", "adamw")).lower()
    wd = float(opt_cfg.get("weight_decay", 1e-5))
    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(expert.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "sgd":
        optimizer = torch.optim.SGD(expert.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
    else:
        optimizer = torch.optim.Adam(expert.parameters(), lr=lr, weight_decay=wd)

    # Scheduler
    sched_cfg = training_cfg.get("scheduler", {}) or {}
    sched_name = str(sched_cfg.get("name", "cosine")).lower()
    scheduler = None
    if sched_name == "cosine":
        warmup_ep = int(sched_cfg.get("warmup_epochs", 20))
        min_lr = float(sched_cfg.get("min_lr", 1e-6))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=max(epochs - warmup_ep, 1), eta_min=min_lr
        )

    # Sliding window config
    sw_cfg = training_cfg.get("sliding_window", {}) or {}
    roi_size = tuple(sw_cfg.get("roi_size", [96, 96, 96]))
    sw_batch = int(sw_cfg.get("sw_batch_size", 4))
    sw_overlap = float(sw_cfg.get("overlap", 0.5))

    best_dice = -1.0
    best_path = None
    last_path = None

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # ---- Train ----
        expert.train()
        train_loss = 0.0
        n_steps = 0
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(train_loader):
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                img, mask = batch[0].to(device), batch[1].to(device)
            else:
                img, mask = batch["image"].to(device), batch["label"].to(device)

            # Squeeze label channel dim [B,1,D,H,W] → [B,D,H,W]
            if mask.ndim == 5 and mask.shape[1] == 1:
                mask = mask.squeeze(1).long()
            elif mask.dtype != torch.long:
                mask = mask.long()

            with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
                logits = expert(img)
                loss = ce_plus_dice(logits, mask, num_classes=num_classes) / grad_accum

            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()

            if (batch_idx + 1) % grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            train_loss += loss.item() * grad_accum
            n_steps += 1

        train_loss /= max(n_steps, 1)
        if scheduler is not None:
            scheduler.step()

        # ---- Val (sliding-window) ----
        expert.eval()
        val_dices = []
        raw_model = expert.module if isinstance(expert, nn.DataParallel) else expert

        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    img, mask = batch[0].to(device), batch[1].to(device)
                else:
                    img, mask = batch["image"].to(device), batch["label"].to(device)

                if mask.ndim == 5 and mask.shape[1] == 1:
                    mask = mask.squeeze(1).long()
                elif mask.dtype != torch.long:
                    mask = mask.long()

                with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
                    logits = _sliding_window_inference(raw_model, img, roi_size, sw_batch, sw_overlap)

                pred = logits.argmax(dim=1)
                # per-class dice (skip bg=0)
                for c in range(1, num_classes):
                    p = (pred == c).float()
                    t = (mask == c).float()
                    inter = (p * t).sum()
                    union = p.sum() + t.sum()
                    val_dices.append((2.0 * inter / (union + 1e-7)).item())

        val_dice = float(np.mean(val_dices)) if val_dices else 0.0
        elapsed = time.time() - t0

        print(f"  epoch {epoch}/{epochs}  train_loss={train_loss:.4f}  val_dice={val_dice:.4f}  time={elapsed:.0f}s")

        # Save
        raw_sd = (expert.module if isinstance(expert, nn.DataParallel) else expert).state_dict()
        ckpt = {"epoch": epoch, "model": raw_sd, "best_metric": best_dice, "metrics": {"val_dice_mean": val_dice}}
        last_path = str(ckpt_dir / "last.pt")
        torch.save(ckpt, last_path)

        if val_dice > best_dice:
            best_dice = val_dice
            ckpt["best_metric"] = best_dice
            best_path = str(ckpt_dir / "best.pt")
            torch.save(ckpt, best_path)
            print(f"  → new best dice={best_dice:.4f}")

    return {"best_metric": best_dice, "best_ckpt": best_path, "last_ckpt": last_path}


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Train a single 3D expert")
    ap.add_argument("--exp", required=True, help="Experiment YAML (configs/3d/exp_msd03_liver.yaml)")
    ap.add_argument("--expert", required=True, choices=["segresnet", "swin_unetr", "nnunet"],
                    help="Which expert to train")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--gpus", default="0", help="GPU ids, e.g. 0,1")
    ap.add_argument("--epochs", type=int, default=None, help="Override epochs")
    ap.add_argument("--smoke", action="store_true", help="Smoke test with random data (2 epochs)")
    args = ap.parse_args()

    exp_cfg = load_config(args.exp)
    seed_everything(exp_cfg.get("seed", 42))

    experts_cfg = load_config(exp_cfg.get("experts_config", "configs/3d/experts.yaml"))
    training_cfg = load_config(exp_cfg.get("training_config", "configs/3d/training.yaml"))

    if args.epochs is not None:
        training_cfg["epochs"] = args.epochs
    if args.smoke:
        training_cfg["epochs"] = 2
        training_cfg["batch_size"] = 1

    device, gpu_ids = _get_device(args.gpus)

    num_classes = int(exp_cfg.get("dataset", {}).get("num_classes", 3))
    in_channels = int(exp_cfg.get("dataset", {}).get("in_channels", 1))

    # Build expert
    factory = ExpertFactory(experts_cfg)
    expert = factory.build_one(args.expert, in_channels=in_channels, classes=num_classes)

    # Run dir
    exp_name = exp_cfg.get("exp_name", "segmoe_3d")
    run_dir = Path(exp_cfg.get("output", {}).get("run_dir", f"runs/{exp_name}").replace("${exp_name}", exp_name))
    tag = f"fold{args.fold}/{expert.name}"

    print(f"\n[train_expert_3d] expert={expert.name}  classes={num_classes}  device={device}  gpus={gpu_ids}")
    print(f"  run_dir={run_dir}  tag={tag}")

    # Data loaders
    if args.smoke:
        patch = tuple(training_cfg.get("sliding_window", {}).get("roi_size", [96, 96, 96]))
        train_dl, val_dl = _build_dummy_loaders(num_classes, patch, training_cfg.get("batch_size", 1))
    else:
        # TODO: plug in real 3D dataset loader here.
        # For now require --smoke or provide your own DataLoader.
        raise NotImplementedError(
            "Real 3D data loading not yet implemented. Use --smoke for testing, "
            "or implement a 3D NIfTI DataLoader and plug it in here."
        )

    result = train_expert(expert, train_dl, val_dl, num_classes, training_cfg, run_dir, tag, device, gpu_ids)
    print(f"\n[train_expert_3d] Done. best_dice={result['best_metric']:.4f}")
    print(f"  best_ckpt: {result['best_ckpt']}")
    print(f"  last_ckpt: {result['last_ckpt']}")


if __name__ == "__main__":
    main()

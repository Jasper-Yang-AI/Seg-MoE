"""
3D Layer1 expert training script.

流程: 与 train_2d_experts.py 完全一致
  1. 加载 splits (volume-level JSONL)
  2. 构建 SegmentationDataset3D (random crop training / full volume val)
  3. 遍历三专家, 调用统一训练引擎 train_model_3d()
  4. 保存 checkpoints/layer1/fold{k}/{expert}/best.pt

Usage:
    python scripts/train/train_layer1_3d.py \\
        --exp      configs/3d/exp/exp_prostate_local_3d.yaml \\
        --training configs/3d/training_layer1_3d.yaml \\
        --models   configs/3d/models_3d.yaml \\
        --augs     configs/3d/augs_3d.yaml \\
        --fold 0   --gpus 0
"""
from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from seg_moe.data.dataset_3d import SegmentationDataset3D
from seg_moe.evaluation.metrics_3d import compute_dice_batch_3d
from seg_moe.models.factory_3d import build_expert_3d, expert_name_3d, list_experts_3d
from seg_moe.training.losses import build_loss_fn, ce_plus_dice
from seg_moe.utils.config import load_config, resolve_run_dir
from seg_moe.utils.io import ensure_dir, load_jsonl, save_jsonl
from seg_moe.utils.seed import seed_everything
from seg_moe.utils.spatial import parse_3d_size

_IS_WINDOWS = platform.system() == "Windows"
_DEFAULT_WORKERS = 2 if _IS_WINDOWS else 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_gpu_ids(gpus_str: str | None):
    if not torch.cuda.is_available():
        return torch.device("cpu"), []
    if gpus_str:
        ids = [int(x.strip()) for x in gpus_str.split(",")]
    else:
        ids = list(range(torch.cuda.device_count()))
    torch.cuda.set_device(ids[0])
    return torch.device(f"cuda:{ids[0]}"), ids


def _wrap_dp(model: nn.Module, gpu_ids: list) -> nn.Module:
    if len(gpu_ids) > 1 and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)
        print(f"  DataParallel on GPUs {gpu_ids}")
    return model


def _unwrap(m: nn.Module) -> nn.Module:
    return m.module if isinstance(m, nn.DataParallel) else m


def _load_splits(dataset_cfg: dict) -> list[dict]:
    splits_dir = Path(dataset_cfg["paths"]["splits_dir"])
    path = splits_dir / "splits_train5fold_testfixed.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Splits not found: {path}. Run scripts/data/make_splits_3d.py first.")
    return load_jsonl(path)


def _sliding_window_inference(model: nn.Module, x: torch.Tensor,
                               roi_size: tuple, sw_batch: int,
                               overlap: float, device: torch.device) -> torch.Tensor:
    try:
        from monai.inferers import sliding_window_inference
        return sliding_window_inference(x, roi_size, sw_batch, model,
                                        overlap=overlap, mode="gaussian",
                                        device=device)
    except ImportError:
        return model(x)


# ---------------------------------------------------------------------------
# 3D training loop
# ---------------------------------------------------------------------------

def train_model_3d(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int,
    training_cfg: Dict[str, Any],
    run_dir: Path,
    tag: str,
    device: torch.device,
    resume_from: str | None = None,
) -> dict:
    """Unified 3D training loop (mirrors 2D engine but without TensorBoard dep)."""
    from torch.utils.tensorboard import SummaryWriter

    ckpt_dir = ensure_dir(run_dir / "checkpoints" / tag)
    log_dir = ensure_dir(run_dir / "logs" / tag)
    writer = SummaryWriter(log_dir=str(log_dir))

    lr = float(training_cfg["lr"])
    epochs = int(training_cfg["epochs"])
    grad_accum = int(training_cfg.get("gradient_accumulation_steps", 1))

    # Build loss function (config-driven, defaults to ce_plus_dice)
    loss_cfg = training_cfg.get("loss", {}) or {}
    loss_fn = build_loss_fn(loss_cfg, num_classes)

    amp_cfg = training_cfg.get("amp", {}) or {}
    amp_on = bool(amp_cfg.get("enabled", True)) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if str(amp_cfg.get("dtype", "bfloat16")).lower() in ("bf16", "bfloat16") else torch.float16
    # bf16 doesn't need GradScaler
    use_scaler = amp_on and (amp_dtype == torch.float16)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    # Optimizer
    opt_cfg = training_cfg.get("optimizer", {}) or {}
    wd = float(opt_cfg.get("weight_decay", 1e-5))
    opt_name = str(opt_cfg.get("name", "adamw")).lower()
    if opt_name == "adamw":
        betas = tuple(opt_cfg.get("betas", [0.9, 0.999]))
        eps = float(opt_cfg.get("eps", 1e-8))
        optimizer = torch.optim.AdamW(_unwrap(model).parameters(), lr=lr,
                                      weight_decay=wd, betas=betas, eps=eps)
    elif opt_name == "sgd":
        momentum = float(opt_cfg.get("momentum", 0.99))
        nesterov = bool(opt_cfg.get("nesterov", True))
        optimizer = torch.optim.SGD(_unwrap(model).parameters(), lr=lr,
                                    weight_decay=wd, momentum=momentum, nesterov=nesterov)
    else:
        optimizer = torch.optim.Adam(_unwrap(model).parameters(), lr=lr, weight_decay=wd)

    # Scheduler
    sched_cfg = training_cfg.get("scheduler", {}) or {}
    sched_name = str(sched_cfg.get("name", "cosine")).lower()
    warmup_ep = int(sched_cfg.get("warmup_epochs", 20))
    min_lr = float(sched_cfg.get("min_lr", 1e-6))
    total_steps = epochs * len(train_loader)
    warmup_steps = warmup_ep * len(train_loader)

    if sched_name == "poly" and len(train_loader) > 0:
        poly_power = float(sched_cfg.get("poly_power", 0.9))
        scheduler = torch.optim.lr_scheduler.PolynomialLR(
            optimizer, total_iters=total_steps, power=poly_power
        )
    else:
        from seg_moe.training.engine import get_cosine_schedule_with_warmup
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, warmup_steps, total_steps, min_lr_ratio=min_lr / lr
        ) if len(train_loader) > 0 else None

    # Sliding window config
    sw_cfg = training_cfg.get("sliding_window", {}) or {}
    roi_size = parse_3d_size(sw_cfg.get("roi_size", [128, 128, 64]))
    sw_batch = int(sw_cfg.get("sw_batch_size", 2))
    sw_overlap = float(sw_cfg.get("overlap", 0.5))

    best_dice = -1.0
    start_epoch = 1
    best_path = None
    last_path = None

    # Resume
    if resume_from and Path(resume_from).exists():
        ckpt = torch.load(resume_from, map_location="cpu", weights_only=True)
        _unwrap(model).load_state_dict({k.removeprefix("module."): v for k, v in ckpt["model"].items()})
        if "opt" in ckpt:
            optimizer.load_state_dict(ckpt["opt"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_dice = float(ckpt.get("best_metric", -1.0))
        print(f"  Resumed from {resume_from} epoch {start_epoch}")

    model.to(device)
    log_interval = int(training_cfg.get("logging", {}).get("log_interval", 10))

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()

        # ---- Train ----
        model.train()
        total_loss = 0.0
        n_steps = 0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader):
            img, mask, _meta = batch
            img  = img.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            if mask.ndim == 5 and mask.shape[1] == 1:    # [B,1,D,H,W] → [B,D,H,W]
                mask = mask.squeeze(1)
            mask = mask.long()

            with torch.amp.autocast("cuda", enabled=amp_on, dtype=amp_dtype):
                logits = _unwrap(model)(img)
                if isinstance(logits, (list, tuple)):
                    logits = logits[0]
                loss = loss_fn(logits, mask) / grad_accum

            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                continue

            if use_scaler:
                scaler.scale(loss).backward()
                if (step + 1) % grad_accum == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            else:
                loss.backward()
                if (step + 1) % grad_accum == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            if scheduler is not None:
                scheduler.step()

            total_loss += float(loss.item()) * grad_accum
            n_steps += 1

        train_loss = total_loss / max(n_steps, 1)

        # ---- Val (sliding window) ----
        model.eval()
        val_dices: list[float] = []
        raw_model = _unwrap(model)

        with torch.no_grad():
            for batch in val_loader:
                img, mask, _meta = batch
                img  = img.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                if mask.ndim == 5 and mask.shape[1] == 1:
                    mask = mask.squeeze(1)
                mask = mask.long()

                with torch.amp.autocast("cuda", enabled=amp_on, dtype=amp_dtype):
                    logits = _sliding_window_inference(raw_model, img, roi_size, sw_batch, sw_overlap, device)

                m = compute_dice_batch_3d(logits, mask, num_classes)
                if not np.isnan(m["dice_mean"]):
                    val_dices.append(m["dice_mean"])

        val_dice = float(np.mean(val_dices)) if val_dices else 0.0
        elapsed = time.time() - t0

        if epoch % log_interval == 0 or epoch == 1:
            print(f"  [3D-L1] epoch {epoch}/{epochs}  loss={train_loss:.4f}  val_dice={val_dice:.4f}  {elapsed:.0f}s")

        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("val/dice_mean", val_dice, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        raw_sd = _unwrap(model).state_dict()
        ckpt = {"epoch": epoch, "model": raw_sd, "opt": optimizer.state_dict(),
                "best_metric": best_dice, "val_dice": val_dice}
        last_path = str(ckpt_dir / "last.pt")
        torch.save(ckpt, last_path)

        if val_dice > best_dice:
            best_dice = val_dice
            ckpt["best_metric"] = best_dice
            best_path = str(ckpt_dir / "best.pt")
            torch.save(ckpt, best_path)
            print(f"  ★ New best dice={best_dice:.4f}")

    writer.close()
    return {"best_metric": best_dice, "best_ckpt": best_path, "last_ckpt": last_path}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Train 3D Layer1 experts")
    ap.add_argument("--exp",       required=True, help="Experiment config YAML")
    ap.add_argument("--training",  required=True, help="Training config YAML")
    ap.add_argument("--models",    required=True, help="Models config YAML")
    ap.add_argument("--augs",      required=True, help="Augmentation config YAML")
    ap.add_argument("--fold",      type=int, default=0)
    ap.add_argument("--gpus",      type=str, default=None)
    ap.add_argument("--resume",    default="none", help="none|last|best|path/to/ckpt.pt")
    ap.add_argument("--skip-if-done", action="store_true")
    ap.add_argument("--epochs",    type=int, default=None, help="Override epochs")
    ap.add_argument("--smoke",     action="store_true", help="Smoke test (2 epochs, dummy data)")
    ap.add_argument("--num-workers", type=int, default=None)
    args = ap.parse_args()

    exp_cfg      = load_config(args.exp)
    training_cfg = load_config(args.training)
    models_cfg   = load_config(args.models)
    augs_cfg     = load_config(args.augs)
    dataset_cfg  = load_config(exp_cfg["dataset"]["config"])

    seed = int(exp_cfg.get("seed", 42))
    seed_everything(seed)

    if args.epochs is not None:
        training_cfg["epochs"] = args.epochs
    if args.num_workers is not None:
        training_cfg.setdefault("dataloader", {})["num_workers"] = args.num_workers
    if args.smoke:
        training_cfg["epochs"] = 2
        training_cfg["batch_size"] = 1

    run_dir = Path(resolve_run_dir(exp_cfg))
    ensure_dir(run_dir)
    device, gpu_ids = _parse_gpu_ids(args.gpus)

    num_classes  = int(dataset_cfg["task"]["num_classes"])
    in_channels  = int(dataset_cfg["input"].get("image_channels", 1))
    fold = int(args.fold)

    expert_cfgs = list_experts_3d(models_cfg)
    print(f"3D Layer1 | fold={fold} | experts: {[expert_name_3d(e) for e in expert_cfgs]}")
    print(f"device={device}  num_classes={num_classes}  in_channels={in_channels}")

    if args.smoke:
        # Smoke: tiny random DataLoader
        from torch.utils.data import TensorDataset
        sz = training_cfg.get("sliding_window", {}).get("roi_size", [128, 128, 64])
        D, H, W = sz
        imgs  = torch.randn(4, in_channels, D, H, W)
        masks = torch.randint(0, num_classes, (4, D, H, W))
        ds = TensorDataset(imgs, masks)
        _dummy_meta: list = [{}] * 4
        train_loader = DataLoader(
            torch.utils.data.TensorDataset(imgs, masks),
            batch_size=1, collate_fn=lambda b: (torch.stack([x[0] for x in b]), torch.stack([x[1] for x in b]), {})
        )
        val_loader = train_loader
    else:
        rows = _load_splits(dataset_cfg)
        nw = int(training_cfg.get("dataloader", {}).get("num_workers", _DEFAULT_WORKERS))
        pm = bool(training_cfg.get("dataloader", {}).get("pin_memory", True))
        bs = int(training_cfg.get("batch_size", 2))

        train_ds = SegmentationDataset3D(
            [r for r in rows if r.get("split") == f"train_fold{fold}"],
            dataset_cfg, augs_cfg, is_train=True)
        val_ds = SegmentationDataset3D(
            [r for r in rows if r.get("split") == f"val_fold{fold}"],
            dataset_cfg, augs_cfg, is_train=False)

        print(f"  Train volumes: {len(train_ds)}  Val volumes: {len(val_ds)}")
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                                  num_workers=nw, pin_memory=pm, drop_last=True)
        val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False,
                                  num_workers=nw, pin_memory=pm)

    for ec in expert_cfgs:
        name = expert_name_3d(ec)
        tag  = f"layer1/fold{fold}/{name}"
        ckpt_dir = run_dir / "checkpoints" / "layer1" / f"fold{fold}" / name
        best_ckpt = ckpt_dir / "best.pt"
        last_ckpt = ckpt_dir / "last.pt"

        if args.skip_if_done and best_ckpt.exists():
            print(f"  Skip (exists): {best_ckpt}")
            continue

        resume_from: str | None = None
        if args.resume.lower() == "last" and last_ckpt.exists():
            resume_from = str(last_ckpt)
        elif args.resume.lower() == "best" and best_ckpt.exists():
            resume_from = str(best_ckpt)
        elif args.resume.lower() not in ("none", ""):
            resume_from = args.resume

        print(f"\n  Building 3D expert: {name}  in_ch={in_channels}  classes={num_classes}")
        model = build_expert_3d(ec, in_channels=in_channels, num_classes=num_classes)
        model = _wrap_dp(model, gpu_ids)

        result = train_model_3d(
            model, train_loader, val_loader, num_classes,
            training_cfg, run_dir, tag, device, resume_from=resume_from
        )
        print(f"  Done {name}: best_dice={result['best_metric']:.4f}")


if __name__ == "__main__":
    main()

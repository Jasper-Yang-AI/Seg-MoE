from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from seg_moe.evaluation.metrics_2d import compute_segmentation_metrics_batch
from seg_moe.training.losses import ce_plus_dice
from seg_moe.utils.io import ensure_dir


@dataclass
class TrainResult:
    best_metric: float
    best_ckpt_path: Optional[str]
    last_ckpt_path: Optional[str]


def _to_device(batch, device: torch.device):
    img, mask, meta = batch
    return img.to(device, non_blocking=True), mask.to(device, non_blocking=True), meta


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int,
    training_cfg: Dict[str, Any],
    run_dir: Path,
    tag: str,
    device: torch.device,
    ignore_index: Optional[int] = None,
    resume_from: Optional[str] = None,
) -> TrainResult:
    ensure_dir(run_dir)
    ckpt_dir = ensure_dir(run_dir / "checkpoints" / tag)
    log_dir = ensure_dir(run_dir / "logs" / tag)

    writer = SummaryWriter(log_dir=str(log_dir))

    lr = float(training_cfg["lr"])
    epochs = int(training_cfg["epochs"])
    wd = float(training_cfg.get("optimizer", {}).get("weight_decay", 0.0))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    best_metric = -1.0
    best_path: Optional[str] = None
    last_path: Optional[str] = None

    start_epoch = 1

    model.to(device)

    amp_cfg = training_cfg.get("amp", {}) or {}
    amp_enabled = bool(amp_cfg.get("enabled", False)) and device.type == "cuda"
    amp_dtype_name = str(amp_cfg.get("dtype", "fp16")).lower()
    if amp_dtype_name in {"bf16", "bfloat16"}:
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    if resume_from:
        resume_path = Path(resume_from)
        if resume_path.exists():
            state = torch.load(resume_path, map_location="cpu")
            if "model" in state:
                model.load_state_dict(state["model"], strict=True)
            if "opt" in state:
                try:
                    opt.load_state_dict(state["opt"])
                except Exception:
                    # If optimizer state is incompatible, continue with fresh optimizer.
                    pass
            if "epoch" in state:
                start_epoch = int(state["epoch"]) + 1
            if "best_metric" in state:
                try:
                    best_metric = float(state["best_metric"])
                except Exception:
                    pass
            # If resuming from best.pt, keep best_path pointing there.
            if resume_path.name.lower() == "best.pt":
                best_path = str(resume_path)
            print(f"[train_model] Resumed from: {resume_path} (start_epoch={start_epoch}, best_metric={best_metric})")
        else:
            print(f"[train_model] Resume checkpoint not found: {resume_path}; training from scratch")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        running_loss = 0.0
        n_steps = 0

        pbar = tqdm(train_loader, desc=f"train[{tag}] e{epoch}/{epochs}")
        for batch in pbar:
            img, mask, _ = _to_device(batch, device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=amp_dtype):
                logits = model(img)
                loss = ce_plus_dice(
                    logits,
                    mask,
                    num_classes=num_classes,
                    dice_smooth=float(training_cfg.get("loss", {}).get("dice_smooth", 1.0)),
                    ce_weight=float(training_cfg.get("loss", {}).get("ce_weight", 1.0)),
                    dice_weight=float(training_cfg.get("loss", {}).get("dice_weight", 1.0)),
                    ignore_index=ignore_index,
                )
            if amp_enabled:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()

            running_loss += float(loss.item())
            n_steps += 1
            pbar.set_postfix(loss=running_loss / max(1, n_steps))

        train_loss = running_loss / max(1, n_steps)
        writer.add_scalar("loss/train", train_loss, epoch)

        # val
        model.eval()
        val_losses = []
        val_metrics_accum = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"val[{tag}] e{epoch}/{epochs}"):
                img, mask, _ = _to_device(batch, device)
                with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=amp_dtype):
                    logits = model(img)
                    loss = ce_plus_dice(
                        logits,
                        mask,
                        num_classes=num_classes,
                        dice_smooth=float(training_cfg.get("loss", {}).get("dice_smooth", 1.0)),
                        ce_weight=float(training_cfg.get("loss", {}).get("ce_weight", 1.0)),
                        dice_weight=float(training_cfg.get("loss", {}).get("dice_weight", 1.0)),
                        ignore_index=ignore_index,
                    )
                val_losses.append(float(loss.item()))
                probs = torch.softmax(logits, dim=1)
                val_metrics_accum.append(compute_segmentation_metrics_batch(probs, mask, num_classes=num_classes))

        val_loss = float(np.mean(val_losses)) if val_losses else 0.0
        writer.add_scalar("loss/val", val_loss, epoch)

        # aggregate metrics
        agg = _aggregate_metrics(val_metrics_accum)
        val_dice = float(agg.get("dice_mean", 0.0))
        writer.add_scalar("metrics/val_dice_mean", val_dice, epoch)

        # checkpoints
        monitor_key = training_cfg.get("checkpoint", {}).get("monitor", "val_dice_mean")
        current_metric = float(agg.get(monitor_key.replace("val_", ""), val_dice))

        last_path = str(ckpt_dir / "last.pt")
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "metrics": agg,
                "best_metric": best_metric,
            },
            last_path,
        )

        if current_metric > best_metric:
            best_metric = current_metric
            best_path = str(ckpt_dir / "best.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "metrics": agg,
                    "best_metric": best_metric,
                },
                best_path,
            )

    writer.close()
    return TrainResult(best_metric=best_metric, best_ckpt_path=best_path, last_ckpt_path=last_path)


def _aggregate_metrics(metrics_list: list[dict[str, float]]) -> dict[str, float]:
    if not metrics_list:
        return {}
    keys = metrics_list[0].keys()
    out: dict[str, float] = {}
    for k in keys:
        out[k] = float(np.mean([m.get(k, 0.0) for m in metrics_list]))
    return out

"""
Unified training engine for Seg-MoE

Design decisions (Windows DataParallel, no DDP):
- DataParallel multi-GPU: 单进程多卡, 无需 torchrun / init_process_group
- checkpoint 保存去 module. 前缀的 state_dict, 加载时自动兼容旧 DDP 权重
- AMP (FP16/BF16) + GradScaler
- 梯度累积 (gradient_accumulation_steps)
- Cosine warmup / StepLR / 无 scheduler
- AdamW / Adam / SGD
- NaN/Inf loss 检测 + 警告
- 文件日志 + TensorBoard
- GPU 显存跟踪
"""
from __future__ import annotations

import logging
import math
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from seg_moe.evaluation.metrics_2d import compute_segmentation_metrics_batch
from seg_moe.training.losses import ce_plus_dice, build_loss_fn
from seg_moe.utils.io import ensure_dir


# ---------------------------------------------------------------------------
# Checkpoint 兼容工具
# ---------------------------------------------------------------------------

def normalize_state_dict_keys(state_dict: dict) -> dict:
    """Strip 'module.' prefix from state_dict keys.

    兼容旧版 DDP checkpoint 和 DataParallel checkpoint.
    如果 key 以 'module.' 开头则删前缀, 否则原样保留.
    """
    new_sd = {}
    for k, v in state_dict.items():
        new_key = k[len("module."):] if k.startswith("module.") else k
        new_sd[new_key] = v
    return new_sd


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return the raw model regardless of DataParallel wrapping."""
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def load_checkpoint_file(path: Path) -> Dict[str, Any]:
    """Load a full training checkpoint across torch versions.

    PyTorch 2.6 changed torch.load default to weights_only=True, which breaks
    resuming checkpoints that contain optimizer/scheduler/scaler states.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        # Backward compatibility for torch versions without weights_only arg.
        return torch.load(path, map_location="cpu")


# ---------------------------------------------------------------------------
# LR Scheduler helpers
# ---------------------------------------------------------------------------

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps: int, num_training_steps: int, min_lr_ratio: float = 0.01):
    """Cosine annealing with linear warmup (step-level)."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + np.cos(np.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# File logger
# ---------------------------------------------------------------------------

def _setup_file_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger(f"seg_moe.train.{log_path.stem}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(str(log_path), encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(fh)
    return logger


# ---------------------------------------------------------------------------
# TrainResult
# ---------------------------------------------------------------------------

@dataclass
class TrainResult:
    best_metric: float
    best_ckpt_path: Optional[str]
    last_ckpt_path: Optional[str]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_device(batch, device: torch.device):
    img, mask, meta = batch
    return img.to(device, non_blocking=True), mask.to(device, non_blocking=True), meta


def _gpu_mem_mb() -> str:
    """Return current / max GPU memory in MB (cuda:0). Empty string if no CUDA."""
    if not torch.cuda.is_available():
        return ""
    alloc = torch.cuda.memory_allocated(0) / 1024**2
    reserved = torch.cuda.max_memory_reserved(0) / 1024**2
    return f"gpu_mem={alloc:.0f}/{reserved:.0f}MB"


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

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
    """
    统一训练入口, 支持单卡 / DataParallel 双卡.

    training_cfg 可选字段 (均有合理默认值):
        lr, epochs, optimizer.{name,weight_decay,betas,eps,momentum},
        scheduler.{name,warmup_epochs,min_lr,step_size,gamma},
        amp.{enabled,dtype}, gradient_accumulation_steps,
        dataloader.{num_workers,pin_memory}, loss.{dice_smooth,ce_weight,dice_weight},
        checkpoint.{monitor}, logging.{log_interval}
    """
    ensure_dir(run_dir)
    ckpt_dir = ensure_dir(run_dir / "checkpoints" / tag)
    log_dir = ensure_dir(run_dir / "logs" / tag)

    writer = SummaryWriter(log_dir=str(log_dir))
    flog = _setup_file_logger(log_dir / "train.log")

    # ---- Hyper-params ----
    lr = float(training_cfg["lr"])
    epochs = int(training_cfg["epochs"])
    grad_accum_steps = int(training_cfg.get("gradient_accumulation_steps", 1))
    log_interval = int(training_cfg.get("logging", {}).get("log_interval", 20))
    val_interval = max(1, int(training_cfg.get("logging", {}).get("val_interval", 1)))

    early_stop_cfg = training_cfg.get("early_stopping", {}) or {}
    early_stop_enabled = bool(early_stop_cfg.get("enabled", False))
    early_stop_patience = max(1, int(early_stop_cfg.get("patience", 10)))
    early_stop_min_epochs = max(0, int(early_stop_cfg.get("min_epochs", 0)))
    early_stop_min_delta = float(early_stop_cfg.get("min_delta", 0.0))

    # ---- Optimizer (AdamW / Adam / SGD) ----
    opt_cfg = training_cfg.get("optimizer", {}) or {}
    opt_name = str(opt_cfg.get("name", "adam")).lower()
    wd = float(opt_cfg.get("weight_decay", 0.0))

    if opt_name == "adamw":
        betas = tuple(opt_cfg.get("betas", [0.9, 0.999]))
        eps = float(opt_cfg.get("eps", 1e-8))
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=betas, eps=eps)
    elif opt_name == "sgd":
        momentum = float(opt_cfg.get("momentum", 0.9))
        opt = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=momentum)
    else:  # default: adam
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # ---- LR Scheduler ----
    scheduler = None
    sched_cfg = training_cfg.get("scheduler", {}) or {}
    sched_name = str(sched_cfg.get("name", "none")).lower()

    if sched_name == "cosine":
        warmup_epochs = int(sched_cfg.get("warmup_epochs", 10))
        min_lr_ratio = float(sched_cfg.get("min_lr", 1e-6)) / max(lr, 1e-12)
        num_warmup_steps = warmup_epochs * len(train_loader)
        num_training_steps = epochs * len(train_loader)
        scheduler = get_cosine_schedule_with_warmup(opt, num_warmup_steps, num_training_steps, min_lr_ratio)
    elif sched_name == "poly":
        poly_power = float(sched_cfg.get("poly_power", sched_cfg.get("power", 0.9)))
        total_iters = max(epochs * math.ceil(len(train_loader) / max(grad_accum_steps, 1)), 1)
        scheduler = torch.optim.lr_scheduler.PolynomialLR(opt, total_iters=total_iters, power=poly_power)
    elif sched_name == "step":
        step_size = int(sched_cfg.get("step_size", 30))
        gamma = float(sched_cfg.get("gamma", 0.1))
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=gamma)

    # ---- AMP ----
    amp_cfg = training_cfg.get("amp", {}) or {}
    amp_enabled = bool(amp_cfg.get("enabled", False)) and device.type == "cuda"
    amp_dtype_name = str(amp_cfg.get("dtype", "float16")).lower()
    amp_dtype = torch.bfloat16 if amp_dtype_name in {"bf16", "bfloat16"} else torch.float16
    # BF16 动态范围等同 FP32, 不需要 GradScaler; FP16 才需要
    use_scaler = amp_enabled and (amp_dtype == torch.float16)
    scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)

    # ---- State ----
    best_metric = -1.0
    best_path: Optional[str] = None
    last_path: Optional[str] = None
    start_epoch = 1
    global_step = 0
    nan_count = 0
    epochs_since_improvement = 0

    model.to(device)

    # ---- Resume checkpoint (兼容旧 DDP / DP / 裸模型权重) ----
    if resume_from:
        resume_path = Path(resume_from)
        if resume_path.exists():
            state = load_checkpoint_file(resume_path)
            if "model" in state:
                raw_sd = normalize_state_dict_keys(state["model"])
                raw_model = unwrap_model(model)
                info = raw_model.load_state_dict(raw_sd, strict=False)
                if info.missing_keys:
                    print(f"[resume] WARNING missing keys: {info.missing_keys}")
                if info.unexpected_keys:
                    print(f"[resume] WARNING unexpected keys: {info.unexpected_keys}")
            if "opt" in state:
                try:
                    opt.load_state_dict(state["opt"])
                except Exception as e:
                    print(f"[resume] optimizer state incompatible, using fresh optimizer: {e}")
            if "scheduler" in state and scheduler is not None:
                try:
                    scheduler.load_state_dict(state["scheduler"])
                except Exception:
                    pass
            if "scaler" in state and amp_enabled:
                try:
                    scaler.load_state_dict(state["scaler"])
                except Exception:
                    pass
            if "epoch" in state:
                start_epoch = int(state["epoch"]) + 1
            if "global_step" in state:
                global_step = int(state["global_step"])
            if "best_metric" in state:
                try:
                    best_metric = float(state["best_metric"])
                except Exception:
                    pass
            if "epochs_since_improvement" in state:
                try:
                    epochs_since_improvement = int(state["epochs_since_improvement"])
                except Exception:
                    pass
            if resume_path.name.lower() == "best.pt":
                best_path = str(resume_path)
            print(f"[resume] Loaded {resume_path} | epoch→{start_epoch} best={best_metric:.4f} global_step={global_step}")
            flog.info(f"Resumed from {resume_path} epoch={start_epoch} best_metric={best_metric:.4f}")
        else:
            print(f"[resume] Checkpoint not found: {resume_path}; training from scratch")

    # ---- Log training config ----
    n_params = sum(p.numel() for p in unwrap_model(model).parameters()) / 1e6
    is_dp = isinstance(model, torch.nn.DataParallel)
    gpu_count = len(model.device_ids) if is_dp else 1
    header = (
        f"Training [{tag}] | device={device} | GPUs={gpu_count} | DP={is_dp} | "
        f"params={n_params:.1f}M | epochs={epochs} | bs={training_cfg.get('batch_size','?')} | "
        f"lr={lr} | opt={opt_name} | amp={amp_enabled}({amp_dtype_name}) | "
        f"grad_accum={grad_accum_steps} | scheduler={sched_name} | val_interval={val_interval}"
    )
    print(header)
    flog.info(header)
    if early_stop_enabled:
        flog.info(
            "Early stopping enabled: patience=%d min_epochs=%d min_delta=%.6f",
            early_stop_patience,
            early_stop_min_epochs,
            early_stop_min_delta,
        )

    # ---- Build loss function (B3: supports ce_dice_boundary via config) ----
    loss_cfg = training_cfg.get("loss", {})
    loss_fn = build_loss_fn(loss_cfg, num_classes, ignore_index=ignore_index)

    # ---- Training loop ----
    for epoch in range(start_epoch, epochs + 1):
        epoch_t0 = time.time()
        model.train()
        running_loss = 0.0
        n_steps = 0
        pending_grads = False

        pbar = tqdm(train_loader, desc=f"train[{tag}] e{epoch}/{epochs}")
        for batch_idx, batch in enumerate(pbar):
            img, mask, _ = _to_device(batch, device)

            with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=amp_dtype):
                logits = model(img)
                loss = loss_fn(logits, mask)
                loss = loss / grad_accum_steps  # 梯度累积: 等效放大 batch

            # NaN/Inf 检测
            if not torch.isfinite(loss):
                nan_count += 1
                print(f"  ⚠ NaN/Inf loss at epoch {epoch} batch {batch_idx} (count={nan_count})")
                flog.warning(f"NaN/Inf loss epoch={epoch} batch={batch_idx} nan_count={nan_count}")
                opt.zero_grad(set_to_none=True)
                continue

            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            pending_grads = True

            # 梯度累积: 每 grad_accum_steps 步才更新
            if (batch_idx + 1) % grad_accum_steps == 0:
                if use_scaler:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)
                pending_grads = False
                # Step-level schedulers
                if scheduler is not None and sched_name in {"cosine", "poly"}:
                    scheduler.step()
                global_step += 1

            running_loss += float(loss.item()) * grad_accum_steps
            n_steps += 1
            pbar.set_postfix(loss=running_loss / max(1, n_steps), lr=f"{opt.param_groups[0]['lr']:.2e}")

        if pending_grads:
            if use_scaler:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            opt.zero_grad(set_to_none=True)
            if scheduler is not None and sched_name in {"cosine", "poly"}:
                scheduler.step()
            global_step += 1

        train_loss = running_loss / max(1, n_steps)

        # Epoch-level scheduler (StepLR etc.)
        if scheduler is not None and sched_name not in {"cosine", "poly"}:
            scheduler.step()

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("lr", opt.param_groups[0]["lr"], epoch)

        # ---- Validation ----
        should_validate = (epoch % val_interval == 0) or (epoch == epochs)
        val_loss: Optional[float] = None
        agg: dict[str, float] = {}
        val_dice: Optional[float] = None
        current_metric: Optional[float] = None

        if should_validate:
            model.eval()
            val_losses = []
            val_metrics_accum = []

            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"val[{tag}] e{epoch}/{epochs}"):
                    img, mask, _ = _to_device(batch, device)
                    with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=amp_dtype):
                        logits = model(img)
                        loss = loss_fn(logits, mask)
                    val_losses.append(float(loss.item()))
                    probs = torch.softmax(logits, dim=1)
                    val_metrics_accum.append(compute_segmentation_metrics_batch(probs, mask, num_classes=num_classes))

            val_loss = float(np.mean(val_losses)) if val_losses else 0.0
            agg = _aggregate_metrics(val_metrics_accum)
            val_dice = float(agg.get("dice_mean", 0.0))
            writer.add_scalar("loss/val", val_loss, epoch)
            writer.add_scalar("metrics/val_dice_mean", val_dice, epoch)

        epoch_time = time.time() - epoch_t0
        mem_str = _gpu_mem_mb()
        if should_validate and val_loss is not None and val_dice is not None:
            log_line = (
                f"epoch={epoch}/{epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"val_dice={val_dice:.4f} lr={opt.param_groups[0]['lr']:.2e} "
                f"time={epoch_time:.0f}s {mem_str}"
            )
        else:
            log_line = (
                f"epoch={epoch}/{epochs} train_loss={train_loss:.4f} val=skipped "
                f"lr={opt.param_groups[0]['lr']:.2e} time={epoch_time:.0f}s {mem_str}"
            )
        flog.info(log_line)

        # ---- Checkpoints: 统一保存去 module. 前缀的权重 ----
        monitor_key = training_cfg.get("checkpoint", {}).get("monitor", "val_dice_mean")
        improved = False
        if should_validate:
            current_metric = float(agg.get(monitor_key.replace("val_", ""), val_dice or 0.0))
            if current_metric > (best_metric + early_stop_min_delta):
                best_metric = current_metric
                epochs_since_improvement = 0
                improved = True
            elif epoch >= early_stop_min_epochs:
                epochs_since_improvement += 1

        raw_model_sd = unwrap_model(model).state_dict()
        save_dict = {
            "epoch": epoch,
            "global_step": global_step,
            "model": raw_model_sd,
            "opt": opt.state_dict(),
            "metrics": agg,
            "best_metric": best_metric,
            "epochs_since_improvement": epochs_since_improvement,
        }
        if scheduler is not None:
            save_dict["scheduler"] = scheduler.state_dict()
        if amp_enabled:
            save_dict["scaler"] = scaler.state_dict()

        last_path = str(ckpt_dir / "last.pt")
        torch.save(save_dict, last_path)

        if improved:
            save_dict["best_metric"] = best_metric
            best_path = str(ckpt_dir / "best.pt")
            torch.save(save_dict, best_path)
            print(f"  → New best: {monitor_key}={best_metric:.4f}")
            flog.info(f"New best: {monitor_key}={best_metric:.4f}")

        if early_stop_enabled and should_validate and epoch >= early_stop_min_epochs and epochs_since_improvement >= early_stop_patience:
            msg = (
                f"Early stopping at epoch {epoch}: no improvement in {epochs_since_improvement} "
                f"validation rounds on {monitor_key}"
            )
            print(msg)
            flog.info(msg)
            break

    writer.close()
    flog.info(f"Training complete. best_metric={best_metric:.4f}")
    return TrainResult(best_metric=best_metric, best_ckpt_path=best_path, last_ckpt_path=last_path)


# ---------------------------------------------------------------------------
# Metric aggregation (单进程, 无 DDP all_reduce)
# ---------------------------------------------------------------------------

def _aggregate_metrics(metrics_list: list[dict[str, float]]) -> dict[str, float]:
    if not metrics_list:
        return {}
    keys = metrics_list[0].keys()
    out: dict[str, float] = {}
    for k in keys:
        out[k] = float(np.mean([m.get(k, 0.0) for m in metrics_list]))
    return out

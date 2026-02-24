"""
Train patch-level gating network for dynamic expert fusion.

核心流程 (Seg-MoE 课题设计):
  Layer1 train → L1 OOF → Layer2 train → **L2 OOF** → Gating → Eval
  1. 加载 **Layer2** OOF 概率图 [K, M, H, W] (非 Layer1!)
  2. 切分为 patches [K*M, pH, pW]
  3. 门控网络预测 per-patch 专家权重 [K]
  4. 加权融合: fused = Σ_k w_k · probs_k
  5. 与 GT mask patch 计算 Dice + CE loss (端到端)
  6. 可选: 负载平衡正则化 (防止专家坍缩)
  7. 温度退火: τ 从 2.0 → 0.5 (训练初期均匀探索, 后期锐化)

Usage:
    python scripts/train/train_gating.py \\
      --exp configs/2d/exp/exp_msd_task03_liver.yaml \\
      --gating-config configs/2d/gating.yaml \\
      --models configs/2d/models.yaml \\
      --fold 0 --gpus 0,1
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from seg_moe.data.gating_patch_dataset import GatingPatchDataset
from seg_moe.data.indexing import infer_num_classes
from seg_moe.gating.patch_gating_2d import (
    PatchConvGate2D,
    PatchGatingConfig,
    compute_load_balance_loss,
    compute_temperature,
)
from seg_moe.models.factory_2d import list_experts
from seg_moe.training.engine import get_cosine_schedule_with_warmup
from seg_moe.training.losses import build_loss_fn
from seg_moe.utils.config import load_config, resolve_run_dir
from seg_moe.utils.io import ensure_dir, load_jsonl
from seg_moe.utils.seed import seed_everything


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_splits(dataset_cfg: dict) -> list[dict]:
    splits_dir = Path(dataset_cfg["paths"]["splits_dir"])
    stype = dataset_cfg["split"]["type"]
    if stype == "holdout20_then_5fold":
        path = splits_dir / "splits_holdout20_5fold.jsonl"
    elif stype == "train_5fold_test_fixed":
        path = splits_dir / "splits_train5fold_testfixed.jsonl"
    else:
        path = splits_dir / "splits_5fold.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Splits not found: {path}")
    return load_jsonl(path)


def _parse_gpu_ids(gpus_str: str | None) -> tuple[torch.device, list[int]]:
    if not torch.cuda.is_available():
        return torch.device("cpu"), []
    if gpus_str:
        gpu_ids = [int(x.strip()) for x in gpus_str.split(",")]
    else:
        gpu_ids = list(range(torch.cuda.device_count()))
    torch.cuda.set_device(gpu_ids[0])
    return torch.device(f"cuda:{gpu_ids[0]}"), gpu_ids


def _wrap_dp(model: nn.Module, gpu_ids: list[int]) -> nn.Module:
    if len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)
    return model


def _unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Train patch-level gating network")
    ap.add_argument("--exp", required=True, help="Experiment config")
    ap.add_argument("--gating-config", required=True, help="Gating config (configs/2d/gating.yaml)")
    ap.add_argument("--models", required=True, help="Expert model config")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--gpus", type=str, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from")
    ap.add_argument("--skip-if-done", action="store_true")
    args = ap.parse_args()

    # ---- Config loading ----
    exp_cfg = load_config(args.exp)
    gating_cfg_raw = load_config(args.gating_config)
    models_cfg = load_config(args.models)
    dataset_cfg = load_config(exp_cfg["dataset"]["config"])

    gcfg = gating_cfg_raw["gating"]
    tcfg = gating_cfg_raw["training"]

    fold = args.fold
    seed_everything(args.seed)

    device, gpu_ids = _parse_gpu_ids(args.gpus)

    # ---- Dataset params ----
    num_classes = infer_num_classes(dataset_cfg)
    expert_cfgs = list_experts(models_cfg)
    K = len(expert_cfgs)

    patch_size = int(gcfg.get("patch_size", 64))
    stride = int(gcfg.get("stride", 32))

    # ---- OOF paths (Layer2 OOF, NOT Layer1) ----
    cache_root = Path(
        exp_cfg["layering"]["cache_root"].replace("${exp_name}", exp_cfg["exp_name"])
    )
    oof_manifest_path = Path(
        str(
            exp_cfg.get("layering", {}).get(
                "l2_oof_manifest_path",
                cache_root / "oof" / "layer2" / "oof_manifest_layer2.jsonl",
            )
        ).replace("${exp_name}", exp_cfg["exp_name"])
    )
    if not oof_manifest_path.exists():
        raise FileNotFoundError(
            f"Layer2 OOF manifest not found: {oof_manifest_path}. "
            "Run scripts/inference/generate_layer2_oof.py first."
        )

    # ---- Run dir ----
    run_dir = Path(resolve_run_dir(exp_cfg))
    ckpt_dir = run_dir / "checkpoints" / "gating" / f"fold{fold}"
    log_dir = run_dir / "logs" / "gating" / f"fold{fold}"
    ensure_dir(ckpt_dir)
    ensure_dir(log_dir)

    best_ckpt = ckpt_dir / "best.pt"
    last_ckpt = ckpt_dir / "last.pt"

    if args.skip_if_done and best_ckpt.exists():
        print(f"Skip: {best_ckpt}")
        return

    # ---- Datasets ----
    rows = _load_splits(dataset_cfg)
    train_rows = [r for r in rows if r.get("split") == f"train_fold{fold}"]
    val_rows = [r for r in rows if r.get("split") == f"val_fold{fold}"]

    train_ds = GatingPatchDataset(
        train_rows, dataset_cfg, oof_manifest_path,
        expected_num_experts=K, patch_size=patch_size, stride=stride,
        is_train=True,
    )
    val_ds = GatingPatchDataset(
        val_rows, dataset_cfg, oof_manifest_path,
        expected_num_experts=K, patch_size=patch_size, stride=stride,
        is_train=False,
    )

    print(f"Gating | K={K} M={num_classes} patch={patch_size} stride={stride}")
    print(f"Train patches: {len(train_ds)}, Val patches: {len(val_ds)}")

    bs = int(tcfg.get("batch_size", 512))
    nw = int(tcfg.get("dataloader", {}).get("num_workers", 4))
    pm = bool(tcfg.get("dataloader", {}).get("pin_memory", True))

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=nw, pin_memory=pm, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                            num_workers=nw, pin_memory=pm)

    # ---- Build gating model ----
    gate_cfg = PatchGatingConfig(
        num_experts=K,
        num_classes=num_classes,
        patch_size=patch_size,
        stride=stride,
        hidden_dim=int(gcfg.get("hidden_dim", 64)),
        dropout=float(gcfg.get("dropout", 0.1)),
        per_class=bool(gcfg.get("per_class", False)),
        temperature_start=float(gcfg.get("temperature_start", 2.0)),
        temperature_end=float(gcfg.get("temperature_end", 0.5)),
        load_balance_weight=float(gcfg.get("load_balance_weight", 0.01)),
        blend_mode=str(gcfg.get("blend_mode", "gaussian")),
    )
    model = PatchConvGate2D(gate_cfg)
    n_params = sum(p.numel() for p in model.parameters()) / 1e3
    print(f"Gating network: {n_params:.1f}K params")
    model.to(device)
    model = _wrap_dp(model, gpu_ids)

    # ---- Loss ----
    loss_cfg = tcfg.get("loss", {})
    seg_loss_fn = build_loss_fn(loss_cfg, num_classes)
    lb_weight = gate_cfg.load_balance_weight

    # ---- Optimizer ----
    lr = float(tcfg.get("lr", 1e-3))
    opt_cfg = tcfg.get("optimizer", {}) or {}
    opt_name = str(opt_cfg.get("name", "adamw")).lower()
    wd = float(opt_cfg.get("weight_decay", 0.01))
    if opt_name == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "sgd":
        mom = float(opt_cfg.get("momentum", 0.9))
        opt = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=mom)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # ---- Scheduler ----
    epochs = int(tcfg.get("epochs", 50))
    sched_cfg = tcfg.get("scheduler", {}) or {}
    sched_name = str(sched_cfg.get("name", "cosine")).lower()
    scheduler = None
    if sched_name == "cosine":
        warmup_epochs = int(sched_cfg.get("warmup_epochs", 5))
        min_lr_ratio = float(sched_cfg.get("min_lr", 1e-6)) / max(lr, 1e-12)
        scheduler = get_cosine_schedule_with_warmup(
            opt,
            warmup_epochs * len(train_loader),
            epochs * len(train_loader),
            min_lr_ratio,
        )

    # ---- AMP ----
    amp_cfg = tcfg.get("amp", {}) or {}
    amp_enabled = bool(amp_cfg.get("enabled", False)) and device.type == "cuda"
    amp_dtype_name = str(amp_cfg.get("dtype", "float16")).lower()
    amp_dtype = torch.bfloat16 if amp_dtype_name in {"bf16", "bfloat16"} else torch.float16
    use_scaler = amp_enabled and (amp_dtype == torch.float16)
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    # ---- Resume ----
    start_epoch = 1
    best_metric = -1.0
    if args.resume and Path(args.resume).exists():
        state = torch.load(args.resume, map_location="cpu", weights_only=True)
        _unwrap(model).load_state_dict(state["model"])
        if "opt" in state:
            opt.load_state_dict(state["opt"])
        if "epoch" in state:
            start_epoch = int(state["epoch"]) + 1
        if "best_metric" in state:
            best_metric = float(state["best_metric"])
        print(f"Resumed from {args.resume}, epoch {start_epoch}")

    # ---- Logging ----
    writer = SummaryWriter(log_dir=str(log_dir))
    flog = logging.getLogger(f"gating_fold{fold}")
    flog.addHandler(logging.FileHandler(log_dir / "train.log"))
    flog.setLevel(logging.INFO)

    t_start = gate_cfg.temperature_start
    t_end = gate_cfg.temperature_end

    print(f"Training gating: fold={fold} device={device} epochs={epochs} bs={bs} lr={lr}")
    flog.info(f"Training gating: fold={fold} K={K} M={num_classes} patch={patch_size}")

    # ================================================================
    # Training loop
    # ================================================================
    for epoch in range(start_epoch, epochs + 1):
        epoch_t0 = time.time()
        τ = compute_temperature(epoch - 1, epochs, t_start, t_end)

        # ---- Train ----
        model.train()
        running_loss = 0.0
        running_lb = 0.0
        n_steps = 0

        pbar = tqdm(train_loader, desc=f"gating e{epoch}/{epochs}")
        for prob_flat, probs, mask in pbar:
            prob_flat = prob_flat.to(device)             # [B, K*M, pH, pW]
            probs = probs.to(device)                     # [B, K, M, pH, pW]
            mask = mask.to(device)                       # [B, pH, pW]

            with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=amp_dtype):
                weights = model(prob_flat, temperature=τ)  # [B, K] or [B, K, M]
                fused = _unwrap(model).fuse_probs(probs, weights)  # [B, M, pH, pW]

                seg_loss = seg_loss_fn(fused, mask)
                lb_loss = compute_load_balance_loss(
                    weights if weights.dim() == 2 else weights.mean(dim=2)
                )
                loss = seg_loss + lb_weight * lb_loss

            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True)
                continue

            if use_scaler:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            opt.zero_grad(set_to_none=True)

            if scheduler is not None:
                scheduler.step()

            running_loss += float(seg_loss.item())
            running_lb += float(lb_loss.item())
            n_steps += 1
            pbar.set_postfix(loss=running_loss / max(1, n_steps), τ=f"{τ:.2f}")

        train_loss = running_loss / max(1, n_steps)
        train_lb = running_lb / max(1, n_steps)

        # ---- Validation ----
        model.eval()
        val_losses = []
        val_dices: list[float] = []

        with torch.no_grad():
            for prob_flat, probs, mask in tqdm(val_loader, desc=f"val e{epoch}"):
                prob_flat = prob_flat.to(device)
                probs = probs.to(device)
                mask = mask.to(device)

                with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=amp_dtype):
                    weights = model(prob_flat, temperature=τ)
                    fused = _unwrap(model).fuse_probs(probs, weights)
                    loss = seg_loss_fn(fused, mask)

                val_losses.append(float(loss.item()))

                # Per-patch Dice
                pred = fused.argmax(dim=1)  # [B, pH, pW]
                for c in range(1, num_classes):  # skip background
                    p = (pred == c).float()
                    g = (mask == c).float()
                    inter = (p * g).sum()
                    union = p.sum() + g.sum()
                    if union > 0:
                        val_dices.append(float(2 * inter / (union + 1e-8)))

        val_loss = float(np.mean(val_losses)) if val_losses else 0.0
        val_dice = float(np.mean(val_dices)) if val_dices else 0.0

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("loss/load_balance", train_lb, epoch)
        writer.add_scalar("metrics/val_dice", val_dice, epoch)
        writer.add_scalar("temperature", τ, epoch)
        writer.add_scalar("lr", opt.param_groups[0]["lr"], epoch)

        elapsed = time.time() - epoch_t0
        log_line = (
            f"epoch={epoch}/{epochs} τ={τ:.2f} train_loss={train_loss:.4f} "
            f"lb_loss={train_lb:.4f} val_loss={val_loss:.4f} val_dice={val_dice:.4f} "
            f"lr={opt.param_groups[0]['lr']:.2e} time={elapsed:.0f}s"
        )
        print(log_line)
        flog.info(log_line)

        # ---- Checkpoint ----
        raw_model = _unwrap(model)
        ckpt = {
            "model": raw_model.state_dict(),
            "opt": opt.state_dict(),
            "epoch": epoch,
            "best_metric": best_metric,
            "val_dice": val_dice,
            "temperature": τ,
            "gate_cfg": {
                "num_experts": K, "num_classes": num_classes,
                "patch_size": patch_size, "stride": stride,
                "hidden_dim": gate_cfg.hidden_dim,
                "dropout": gate_cfg.dropout,
                "per_class": gate_cfg.per_class,
                "blend_mode": gate_cfg.blend_mode,
            },
        }
        torch.save(ckpt, last_ckpt)

        if val_dice > best_metric:
            best_metric = val_dice
            ckpt["best_metric"] = best_metric
            torch.save(ckpt, best_ckpt)
            print(f"  ★ New best: val_dice={val_dice:.4f}")

    writer.close()
    print(f"Gating training done. Best val_dice={best_metric:.4f}")
    print(f"  Best ckpt: {best_ckpt}")


if __name__ == "__main__":
    main()

"""
Train 3D patch-level gating network.

流程 (与 train_gating.py 完全对应):
  Layer2 OOF logits [K, M, D, H, W] → 3D patches → PatchConvGate3D → Dice+CE loss

Usage:
    python scripts/train/train_gating_3d.py \\
        --exp    configs/3d/exp/exp_prostate_local_3d.yaml \\
        --gating-config configs/3d/gating_3d.yaml \\
        --models configs/3d/models_3d.yaml \\
        --fold 0 --gpus 0
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

from seg_moe.data.gating_patch_dataset_3d import GatingPatchDataset3D
from seg_moe.gating.patch_gating_3d import (
    PatchConvGate3D, PatchGatingConfig3D,
    compute_load_balance_loss_3d, compute_spatial_smooth_loss_3d,
    compute_temperature_3d,
)
from seg_moe.models.factory_3d import list_experts_3d
from seg_moe.training.engine import get_cosine_schedule_with_warmup
from seg_moe.training.losses import build_loss_fn
from seg_moe.utils.checkpoint import extract_model_state_dict, load_trusted_torch_checkpoint
from seg_moe.utils.config import load_config, resolve_run_dir
from seg_moe.utils.io import ensure_dir, load_jsonl
from seg_moe.utils.seed import seed_everything


# ---------------------------------------------------------------------------
# Helpers (copied locally to avoid circular import issues)
# ---------------------------------------------------------------------------

def _parse_gpu_ids(gpus_str):
    if not torch.cuda.is_available():
        return torch.device("cpu"), []
    ids = [int(x.strip()) for x in gpus_str.split(",")] if gpus_str else list(range(torch.cuda.device_count()))
    torch.cuda.set_device(ids[0])
    return torch.device(f"cuda:{ids[0]}"), ids


def _wrap_dp(model, gpu_ids):
    if len(gpu_ids) > 1 and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)
    return model


def _unwrap(m):
    return m.module if isinstance(m, nn.DataParallel) else m


def _load_splits(dataset_cfg):
    path = Path(dataset_cfg["paths"]["splits_dir"]) / "splits_train5fold_testfixed.jsonl"
    return load_jsonl(path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Train 3D gating network")
    ap.add_argument("--exp",           required=True)
    ap.add_argument("--gating-config", required=True, help="configs/3d/gating_3d.yaml")
    ap.add_argument("--models",        required=True)
    ap.add_argument("--fold",          type=int, default=0)
    ap.add_argument("--gpus",          type=str, default=None)
    ap.add_argument("--seed",          type=int, default=42)
    ap.add_argument("--resume",        type=str, default=None)
    ap.add_argument("--skip-if-done",  action="store_true")
    args = ap.parse_args()

    exp_cfg       = load_config(args.exp)
    gating_raw    = load_config(args.gating_config)
    models_cfg    = load_config(args.models)
    dataset_cfg   = load_config(exp_cfg["dataset"]["config"])

    gcfg = gating_raw["gating"]
    tcfg = gating_raw["training"]

    fold = args.fold
    seed_everything(args.seed)
    device, gpu_ids = _parse_gpu_ids(args.gpus)

    num_classes = int(dataset_cfg["task"]["num_classes"])
    expert_cfgs = list_experts_3d(models_cfg)
    K = len(expert_cfgs)

    # Patch size / stride
    patch_size = tuple(int(p) for p in gcfg.get("patch_size", [32, 32, 16]))
    stride     = tuple(int(s) for s in gcfg.get("stride",     [16, 16,  8]))
    fg_ratio   = float(gcfg.get("foreground_oversample_ratio", 0.5))

    # OOF manifest (Layer2)
    cache_root = Path(
        exp_cfg["layering"]["cache_root"].replace("${exp_name}", exp_cfg["exp_name"])
    )
    l2_oof_path = Path(
        str(exp_cfg["layering"].get(
            "l2_oof_manifest_path",
            cache_root / "oof" / "layer2" / "oof_manifest_layer2.jsonl"
        )).replace("${exp_name}", exp_cfg["exp_name"])
    )
    if not l2_oof_path.exists():
        raise FileNotFoundError(
            f"Layer2 OOF manifest not found: {l2_oof_path}\n"
            "Run scripts/inference/generate_layer2_oof_3d.py first."
        )

    run_dir  = Path(resolve_run_dir(exp_cfg))
    ckpt_dir = ensure_dir(run_dir / "checkpoints" / "gating" / f"fold{fold}")
    log_dir  = ensure_dir(run_dir / "logs" / "gating" / f"fold{fold}")
    best_ckpt = ckpt_dir / "best.pt"
    last_ckpt = ckpt_dir / "last.pt"

    if args.skip_if_done and best_ckpt.exists():
        print(f"Skip: {best_ckpt}")
        return

    rows      = _load_splits(dataset_cfg)
    train_rows = [r for r in rows if r.get("split") == f"train_fold{fold}"]
    val_rows   = [r for r in rows if r.get("split") == f"val_fold{fold}"]

    train_ds = GatingPatchDataset3D(
        train_rows, dataset_cfg, l2_oof_path,
        expected_num_experts=K, patch_size=patch_size, stride=stride,
        is_train=True, foreground_oversample_ratio=fg_ratio,
    )
    val_ds = GatingPatchDataset3D(
        val_rows, dataset_cfg, l2_oof_path,
        expected_num_experts=K, patch_size=patch_size, stride=stride,
        is_train=False,
    )

    print(f"Gating3D | K={K} M={num_classes} patch={patch_size} stride={stride}")
    print(f"  Train patches: {len(train_ds)}  Val patches: {len(val_ds)}")

    bs = int(tcfg.get("batch_size", 64))
    nw = int(tcfg.get("dataloader", {}).get("num_workers", 2))
    pm = bool(tcfg.get("dataloader", {}).get("pin_memory", True))

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=nw, pin_memory=pm, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False,
                              num_workers=nw, pin_memory=pm)

    # Build gating model
    gate_cfg = PatchGatingConfig3D(
        num_experts=K,
        num_classes=num_classes,
        patch_size=patch_size,
        stride=stride,
        hidden_dim=int(gcfg.get("hidden_dim", 64)),
        score_hidden_dim=int(gcfg.get("score_hidden_dim", gcfg.get("hidden_dim", 64))),
        dropout=float(gcfg.get("dropout", 0.1)),
        per_class=bool(gcfg.get("per_class", False)),
        use_residual_head=bool(gcfg.get("use_residual_head", True)),
        use_entropy=bool(gcfg.get("use_entropy", True)),
        use_consensus_features=bool(gcfg.get("use_consensus_features", True)),
        use_disagreement_features=bool(gcfg.get("use_disagreement_features", True)),
        use_confidence_features=bool(gcfg.get("use_confidence_features", True)),
        temperature_start=float(gcfg.get("temperature_start", 2.0)),
        temperature_end=float(gcfg.get("temperature_end", 0.5)),
        load_balance_weight=float(gcfg.get("load_balance_weight", 0.01)),
        spatial_smooth_weight=float(gcfg.get("spatial_smooth_weight", 0.0)),
        blend_mode=str(gcfg.get("blend_mode", "gaussian")),
    )
    model = PatchConvGate3D(gate_cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Gating network: {n_params / 1e3:.1f}K params")
    model.to(device)
    model = _wrap_dp(model, gpu_ids)

    # Loss
    loss_cfg = tcfg.get("loss", {})
    seg_loss_fn = build_loss_fn(loss_cfg, num_classes)
    lb_weight   = gate_cfg.load_balance_weight
    tv_weight   = gate_cfg.spatial_smooth_weight

    # Optimizer
    lr = float(tcfg.get("lr", 1e-3))
    opt_cfg = tcfg.get("optimizer", {}) or {}
    wd = float(opt_cfg.get("weight_decay", 0.01))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    # Scheduler
    epochs    = int(tcfg.get("epochs", 50))
    sched_cfg = tcfg.get("scheduler", {}) or {}
    warmup_ep = int(sched_cfg.get("warmup_epochs", 5))
    min_lr_r  = float(sched_cfg.get("min_lr", 1e-6)) / max(lr, 1e-12)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        warmup_ep * max(len(train_loader), 1),
        epochs   * max(len(train_loader), 1),
        min_lr_r,
    )

    # AMP
    amp_cfg   = tcfg.get("amp", {}) or {}
    amp_on    = bool(amp_cfg.get("enabled", True)) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if str(amp_cfg.get("dtype", "bfloat16")).lower() in ("bf16", "bfloat16") else torch.float16
    use_scaler = amp_on and (amp_dtype == torch.float16)
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    # Resume
    start_epoch = 1
    best_metric = -1.0
    if args.resume and Path(args.resume).exists():
        state = load_trusted_torch_checkpoint(args.resume, map_location="cpu")
        _unwrap(model).load_state_dict(extract_model_state_dict(state))
        if "opt" in state:
            optimizer.load_state_dict(state["opt"])
        start_epoch = int(state.get("epoch", 0)) + 1
        best_metric = float(state.get("best_metric", -1.0))
        print(f"Resumed from {args.resume} epoch {start_epoch}")

    writer = SummaryWriter(log_dir=str(log_dir))
    flog = logging.getLogger(f"gating3d_fold{fold}")
    flog.addHandler(logging.FileHandler(log_dir / "train.log"))
    flog.setLevel(logging.INFO)

    t_start = gate_cfg.temperature_start
    t_end   = gate_cfg.temperature_end

    print(f"Training 3D gating: fold={fold} device={device} epochs={epochs} bs={bs}")

    # ================================================================
    # Training loop
    # ================================================================
    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        tau = compute_temperature_3d(epoch - 1, epochs, t_start, t_end)

        # ---- Train ----
        model.train()
        running_loss = running_lb = 0.0
        n_steps = 0

        for logits_cache, mask, sample_ids, positions in tqdm(train_loader, desc=f"e{epoch}/{epochs}"):
            logits_cache = logits_cache.to(device)     # [B, K, M, pd, ph, pw]
            mask = mask.to(device)                     # [B, pd, ph, pw]
            sample_ids = sample_ids.to(device)
            positions = positions.to(device)

            with torch.cuda.amp.autocast(enabled=amp_on, dtype=amp_dtype):
                weights = model(logits_cache, temperature=tau)   # [B, K]
                fused   = _unwrap(model).fuse_logits(logits_cache, weights)  # [B, M, ...]
                seg_loss = seg_loss_fn(fused, mask)
                w_per   = _unwrap(model).weights_per_expert(weights)
                lb_loss  = compute_load_balance_loss_3d(w_per)
                tv_loss  = (
                    compute_spatial_smooth_loss_3d(w_per, sample_ids=sample_ids, positions=positions)
                    if tv_weight > 0 else w_per.new_tensor(0.0)
                )
                loss = seg_loss + lb_weight * lb_loss + tv_weight * tv_loss

            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                continue

            if use_scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            running_loss += float(seg_loss.item())
            running_lb   += float(lb_loss.item())
            n_steps += 1

        train_loss = running_loss / max(n_steps, 1)
        train_lb   = running_lb   / max(n_steps, 1)

        # ---- Validation ----
        model.eval()
        val_losses, val_dices = [], []
        with torch.no_grad():
            for logits_cache, mask, sample_ids, positions in val_loader:
                logits_cache = logits_cache.to(device)
                mask = mask.to(device)
                with torch.cuda.amp.autocast(enabled=amp_on, dtype=amp_dtype):
                    weights = model(logits_cache, temperature=tau)
                    fused   = _unwrap(model).fuse_logits(logits_cache, weights)
                    val_losses.append(float(seg_loss_fn(fused, mask).item()))
                pred = fused.argmax(dim=1)
                for c in range(1, num_classes):
                    p = (pred == c).float()
                    g = (mask == c).float()
                    inter = (p * g).sum()
                    union = p.sum() + g.sum()
                    if union > 0:
                        val_dices.append(float(2 * inter / (union + 1e-8)))

        val_loss = float(np.mean(val_losses)) if val_losses else 0.0
        val_dice = float(np.mean(val_dices))  if val_dices  else 0.0
        elapsed  = time.time() - t0

        log_str = (
            f"epoch={epoch}/{epochs} τ={tau:.2f} "
            f"loss={train_loss:.4f} lb={train_lb:.4f} "
            f"val_loss={val_loss:.4f} val_dice={val_dice:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.2e} {elapsed:.0f}s"
        )
        print(log_str)
        flog.info(log_str)

        writer.add_scalar("loss/train",     train_loss, epoch)
        writer.add_scalar("loss/val",       val_loss,   epoch)
        writer.add_scalar("loss/lb",        train_lb,   epoch)
        writer.add_scalar("metrics/val_dice", val_dice, epoch)
        writer.add_scalar("temperature",    tau,        epoch)

        # Checkpoint
        raw_m = _unwrap(model)
        ckpt = {
            "model": raw_m.state_dict(), "opt": optimizer.state_dict(),
            "epoch": epoch, "best_metric": best_metric, "val_dice": val_dice,
            "temperature": tau,
            "gate_cfg": {"num_experts": K, "num_classes": num_classes,
                         "patch_size": patch_size, "stride": stride,
                         "hidden_dim": gate_cfg.hidden_dim,
                         "score_hidden_dim": gate_cfg.score_hidden_dim,
                         "dropout": gate_cfg.dropout,
                         "use_residual_head": gate_cfg.use_residual_head,
                         "use_entropy": gate_cfg.use_entropy,
                         "use_consensus_features": gate_cfg.use_consensus_features,
                         "use_disagreement_features": gate_cfg.use_disagreement_features,
                         "use_confidence_features": gate_cfg.use_confidence_features,
                         "per_class": gate_cfg.per_class,
                         "blend_mode": gate_cfg.blend_mode,
                         "temperature_end": gate_cfg.temperature_end},
        }
        torch.save(ckpt, last_ckpt)
        if val_dice > best_metric:
            best_metric = val_dice
            ckpt["best_metric"] = best_metric
            torch.save(ckpt, best_ckpt)
            print(f"  ★ New best val_dice={val_dice:.4f}")

    writer.close()
    print(f"3D Gating done. Best val_dice={best_metric:.4f}")
    print(f"  Best: {best_ckpt}")


if __name__ == "__main__":
    main()

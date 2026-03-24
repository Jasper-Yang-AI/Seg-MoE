"""Train patch-level gating network for dynamic expert fusion."""
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

from seg_moe.data.gating_patch_dataset import GatingPatchDataset, SampleGroupedBatchSampler
from seg_moe.data.indexing import infer_num_classes
from seg_moe.gating.patch_gating_2d import (
    PatchConvGate2D,
    PatchGatingConfig,
    compute_load_balance_loss,
    compute_spatial_smooth_loss,
    compute_temperature,
)
from seg_moe.models.factory_2d import list_experts
from seg_moe.training.engine import get_cosine_schedule_with_warmup
from seg_moe.training.losses import build_loss_fn
from seg_moe.utils.config import load_config, resolve_run_dir
from seg_moe.utils.io import ensure_dir, load_jsonl
from seg_moe.utils.seed import seed_everything


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


def _move_extra_to_device(extra: dict, device: torch.device) -> dict:
    if not extra:
        return {}
    out = {}
    for key, value in extra.items():
        out[key] = value.to(device) if torch.is_tensor(value) else value
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Train patch-level gating network")
    ap.add_argument("--exp", required=True, help="Experiment config")
    ap.add_argument("--gating-config", required=True, help="Gating config YAML")
    ap.add_argument("--models", required=True, help="Expert model config")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--gpus", type=str, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from")
    ap.add_argument("--skip-if-done", action="store_true")
    args = ap.parse_args()

    exp_cfg = load_config(args.exp)
    gating_cfg_raw = load_config(args.gating_config)
    models_cfg = load_config(args.models)
    dataset_cfg = load_config(exp_cfg["dataset"]["config"])

    gcfg = gating_cfg_raw["gating"]
    tcfg = gating_cfg_raw["training"]

    fold = int(args.fold)
    seed_everything(args.seed)
    device, gpu_ids = _parse_gpu_ids(args.gpus)

    num_classes = infer_num_classes(dataset_cfg)
    raw_image_channels = int(dataset_cfg["input"].get("image_channels", 3))
    image_channels = 3 if raw_image_channels == 1 else raw_image_channels
    expert_cfgs = list_experts(models_cfg)
    num_experts = len(expert_cfgs)

    patch_size = int(gcfg.get("patch_size", 64))
    stride = int(gcfg.get("stride", 32))
    fg_oversample = float(gcfg.get("foreground_oversample_ratio", 0.0))

    cache_root = Path(exp_cfg["layering"]["cache_root"].replace("${exp_name}", exp_cfg["exp_name"]))
    l2_oof_manifest_path = Path(
        str(
            exp_cfg.get("layering", {}).get(
                "l2_oof_manifest_path",
                cache_root / "oof" / "layer2" / "oof_manifest_layer2.jsonl",
            )
        ).replace("${exp_name}", exp_cfg["exp_name"])
    )
    if not l2_oof_manifest_path.exists():
        raise FileNotFoundError(
            f"Layer2 OOF manifest not found: {l2_oof_manifest_path}. "
            "Run scripts/inference/generate_layer2_oof.py first."
        )

    l1_oof_manifest_path = Path(
        str(
            exp_cfg.get("layering", {}).get(
                "oof_manifest_path",
                cache_root / "oof" / "layer1" / "oof_manifest.jsonl",
            )
        ).replace("${exp_name}", exp_cfg["exp_name"])
    )

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

    rows = _load_splits(dataset_cfg)
    train_rows = [r for r in rows if r.get("split") == f"train_fold{fold}"]
    val_rows = [r for r in rows if r.get("split") == f"val_fold{fold}"]

    use_layer1_semantics = bool(gcfg.get("use_layer1_semantics", False))
    use_image_context = bool(gcfg.get("use_image_context", False))
    use_position_channels = bool(gcfg.get("use_position_channels", False))
    use_slice_position = bool(gcfg.get("use_slice_position", False))
    data_cfg = gating_cfg_raw.get("data", {}) or {}
    cache_in_memory = bool(data_cfg.get("cache_in_memory", True))
    cache_max_items = data_cfg.get("cache_max_items")
    train_sampler_name = str(data_cfg.get("train_sampler", "random")).lower()
    shuffle_patches_within_sample = bool(data_cfg.get("shuffle_patches_within_sample", True))
    if train_sampler_name not in {"random", "sample_grouped"}:
        raise ValueError(f"Unsupported train_sampler={train_sampler_name!r}")

    train_ds = GatingPatchDataset(
        train_rows,
        dataset_cfg,
        l2_oof_manifest_path,
        expected_num_experts=num_experts,
        patch_size=patch_size,
        stride=stride,
        is_train=True,
        foreground_oversample_ratio=fg_oversample if train_sampler_name == "random" else 0.0,
        cache_in_memory=cache_in_memory,
        cache_max_items=cache_max_items,
        layer1_oof_manifest_path=l1_oof_manifest_path,
        use_layer1_semantics=use_layer1_semantics,
        use_image_context=use_image_context,
        use_position_channels=use_position_channels,
        use_slice_position=use_slice_position,
    )
    val_ds = GatingPatchDataset(
        val_rows,
        dataset_cfg,
        l2_oof_manifest_path,
        expected_num_experts=num_experts,
        patch_size=patch_size,
        stride=stride,
        is_train=False,
        cache_in_memory=cache_in_memory,
        cache_max_items=cache_max_items,
        layer1_oof_manifest_path=l1_oof_manifest_path,
        use_layer1_semantics=use_layer1_semantics,
        use_image_context=use_image_context,
        use_position_channels=use_position_channels,
        use_slice_position=use_slice_position,
    )

    print(
        f"Gating | K={num_experts} M={num_classes} patch={patch_size} stride={stride} "
        f"fg_oversample={fg_oversample}"
    )
    print(f"Train patches: {len(train_ds)}, Val patches: {len(val_ds)}")
    cache_desc = "full" if (cache_in_memory and cache_max_items is None) else (
        f"lru[{cache_max_items}]" if cache_in_memory else "disabled"
    )
    print(f"Runtime | train_sampler={train_sampler_name} cache={cache_desc}")

    batch_size = int(tcfg.get("batch_size", 512))
    num_workers = int(tcfg.get("dataloader", {}).get("num_workers", 4))
    pin_memory = bool(tcfg.get("dataloader", {}).get("pin_memory", True))
    persistent_workers = bool(tcfg.get("dataloader", {}).get("persistent_workers", False)) if num_workers > 0 else False
    prefetch_factor = tcfg.get("dataloader", {}).get("prefetch_factor")

    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
    }
    if num_workers > 0 and prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = int(prefetch_factor)

    if train_sampler_name == "sample_grouped":
        train_batch_sampler = SampleGroupedBatchSampler(
            train_ds.sample_patch_ranges,
            batch_size=batch_size,
            sample_fg_indices=train_ds.sample_fg_indices,
            drop_last=True,
            shuffle_samples=True,
            shuffle_patches_within_sample=shuffle_patches_within_sample,
            foreground_oversample_ratio=fg_oversample,
        )
        train_loader = DataLoader(train_ds, batch_sampler=train_batch_sampler, **loader_kwargs)
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            **loader_kwargs,
        )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    gate_cfg = PatchGatingConfig(
        num_experts=num_experts,
        num_classes=num_classes,
        image_channels=image_channels,
        patch_size=patch_size,
        stride=stride,
        hidden_dim=int(gcfg.get("hidden_dim", 64)),
        score_hidden_dim=int(gcfg.get("score_hidden_dim", gcfg.get("hidden_dim", 64))),
        context_hidden_dim=int(gcfg.get("context_hidden_dim", 32)),
        dropout=float(gcfg.get("dropout", 0.1)),
        per_class=bool(gcfg.get("per_class", False)),
        use_residual_head=bool(gcfg.get("use_residual_head", True)),
        use_entropy=bool(gcfg.get("use_entropy", True)),
        use_consensus_features=bool(gcfg.get("use_consensus_features", True)),
        use_disagreement_features=bool(gcfg.get("use_disagreement_features", True)),
        use_confidence_features=bool(gcfg.get("use_confidence_features", True)),
        use_prior_agreement_features=bool(gcfg.get("use_prior_agreement_features", False)),
        use_layer1_semantics=use_layer1_semantics,
        use_image_context=use_image_context,
        use_position_channels=use_position_channels,
        use_slice_position=use_slice_position,
        use_context_film=bool(gcfg.get("use_context_film", True)),
        temperature_start=float(gcfg.get("temperature_start", 2.0)),
        temperature_end=float(gcfg.get("temperature_end", 0.5)),
        load_balance_weight=float(gcfg.get("load_balance_weight", 0.01)),
        spatial_smooth_weight=float(gcfg.get("spatial_smooth_weight", 0.0)),
        blend_mode=str(gcfg.get("blend_mode", "gaussian")),
    )

    model = PatchConvGate2D(gate_cfg)
    print(f"Gating network: {sum(p.numel() for p in model.parameters()) / 1e3:.1f}K params")
    model.to(device)
    model = _wrap_dp(model, gpu_ids)

    seg_loss_fn = build_loss_fn(tcfg.get("loss", {}), num_classes)
    lb_weight = gate_cfg.load_balance_weight
    tv_weight = gate_cfg.spatial_smooth_weight

    lr = float(tcfg.get("lr", 1e-3))
    opt_cfg = tcfg.get("optimizer", {}) or {}
    opt_name = str(opt_cfg.get("name", "adamw")).lower()
    weight_decay = float(opt_cfg.get("weight_decay", 0.01))
    if opt_name == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == "sgd":
        momentum = float(opt_cfg.get("momentum", 0.9))
        opt = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    epochs = int(tcfg.get("epochs", 50))
    scheduler = None
    sched_cfg = tcfg.get("scheduler", {}) or {}
    if str(sched_cfg.get("name", "cosine")).lower() == "cosine":
        warmup_epochs = int(sched_cfg.get("warmup_epochs", 5))
        min_lr_ratio = float(sched_cfg.get("min_lr", 1e-6)) / max(lr, 1e-12)
        scheduler = get_cosine_schedule_with_warmup(
            opt,
            warmup_epochs * len(train_loader),
            epochs * len(train_loader),
            min_lr_ratio,
        )

    amp_cfg = tcfg.get("amp", {}) or {}
    amp_enabled = bool(amp_cfg.get("enabled", False)) and device.type == "cuda"
    amp_dtype_name = str(amp_cfg.get("dtype", "float16")).lower()
    amp_dtype = torch.bfloat16 if amp_dtype_name in {"bf16", "bfloat16"} else torch.float16
    use_scaler = amp_enabled and (amp_dtype == torch.float16)
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

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

    writer = SummaryWriter(log_dir=str(log_dir))
    flog = logging.getLogger(f"gating_fold{fold}")
    flog.addHandler(logging.FileHandler(log_dir / "train.log"))
    flog.setLevel(logging.INFO)

    print(f"Training gating: fold={fold} device={device} epochs={epochs} bs={batch_size} lr={lr}")
    flog.info(f"Training gating: fold={fold} K={num_experts} M={num_classes} patch={patch_size}")
    early_cfg = tcfg.get("early_stopping", {}) or {}
    early_enabled = bool(early_cfg.get("enabled", False))
    early_patience = int(early_cfg.get("patience", 0))
    early_min_epochs = int(early_cfg.get("min_epochs", 0))
    early_min_delta = float(early_cfg.get("min_delta", 0.0))
    epochs_without_improve = 0
    if early_enabled:
        print(
            f"Early stopping enabled: patience={early_patience} "
            f"min_epochs={early_min_epochs} min_delta={early_min_delta:.6f}"
        )

    for epoch in range(start_epoch, epochs + 1):
        epoch_t0 = time.time()
        tau = compute_temperature(epoch - 1, epochs, gate_cfg.temperature_start, gate_cfg.temperature_end)

        model.train()
        running_loss = 0.0
        running_lb = 0.0
        n_steps = 0

        pbar = tqdm(train_loader, desc=f"gating e{epoch}/{epochs}")
        for logits_cache, mask, sample_ids, positions, extra in pbar:
            logits_cache = logits_cache.to(device)
            mask = mask.to(device)
            sample_ids = sample_ids.to(device)
            positions = positions.to(device)
            extra = _move_extra_to_device(extra, device)

            with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=amp_dtype):
                weights = model(logits_cache, extra=extra, temperature=tau)
                fused = _unwrap(model).fuse_logits(logits_cache, weights)
                seg_loss = seg_loss_fn(fused, mask)

                w_per_expert = _unwrap(model).weights_per_expert(weights)
                lb_loss = compute_load_balance_loss(w_per_expert)
                tv_loss = (
                    compute_spatial_smooth_loss(w_per_expert, sample_ids=sample_ids, positions=positions)
                    if tv_weight > 0 else w_per_expert.new_tensor(0.0)
                )
                loss = seg_loss + lb_weight * lb_loss + tv_weight * tv_loss

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
            pbar.set_postfix(loss=running_loss / max(1, n_steps), tau=f"{tau:.2f}")

        train_loss = running_loss / max(1, n_steps)
        train_lb = running_lb / max(1, n_steps)

        model.eval()
        val_losses: list[float] = []
        val_dices: list[float] = []

        with torch.no_grad():
            for logits_cache, mask, sample_ids, positions, extra in tqdm(val_loader, desc=f"val e{epoch}"):
                logits_cache = logits_cache.to(device)
                mask = mask.to(device)
                extra = _move_extra_to_device(extra, device)

                with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=amp_dtype):
                    weights = model(logits_cache, extra=extra, temperature=tau)
                    fused = _unwrap(model).fuse_logits(logits_cache, weights)
                    loss = seg_loss_fn(fused, mask)

                val_losses.append(float(loss.item()))
                pred = fused.argmax(dim=1)
                for cls in range(1, num_classes):
                    p = (pred == cls).float()
                    g = (mask == cls).float()
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
        writer.add_scalar("temperature", tau, epoch)
        writer.add_scalar("lr", opt.param_groups[0]["lr"], epoch)

        elapsed = time.time() - epoch_t0
        log_line = (
            f"epoch={epoch}/{epochs} tau={tau:.2f} train_loss={train_loss:.4f} "
            f"lb_loss={train_lb:.4f} val_loss={val_loss:.4f} val_dice={val_dice:.4f} "
            f"lr={opt.param_groups[0]['lr']:.2e} time={elapsed:.0f}s"
        )
        print(log_line)
        flog.info(log_line)

        raw_model = _unwrap(model)
        ckpt = {
            "model": raw_model.state_dict(),
            "opt": opt.state_dict(),
            "epoch": epoch,
            "best_metric": best_metric,
            "val_dice": val_dice,
            "temperature": tau,
            "gate_cfg": {
                "num_experts": gate_cfg.num_experts,
                "num_classes": gate_cfg.num_classes,
                "image_channels": gate_cfg.image_channels,
                "patch_size": gate_cfg.patch_size,
                "stride": gate_cfg.stride,
                "hidden_dim": gate_cfg.hidden_dim,
                "score_hidden_dim": gate_cfg.score_hidden_dim,
                "context_hidden_dim": gate_cfg.context_hidden_dim,
                "dropout": gate_cfg.dropout,
                "per_class": gate_cfg.per_class,
                "use_residual_head": gate_cfg.use_residual_head,
                "use_entropy": gate_cfg.use_entropy,
                "use_consensus_features": gate_cfg.use_consensus_features,
                "use_disagreement_features": gate_cfg.use_disagreement_features,
                "use_confidence_features": gate_cfg.use_confidence_features,
                "use_prior_agreement_features": gate_cfg.use_prior_agreement_features,
                "use_layer1_semantics": gate_cfg.use_layer1_semantics,
                "use_image_context": gate_cfg.use_image_context,
                "use_position_channels": gate_cfg.use_position_channels,
                "use_slice_position": gate_cfg.use_slice_position,
                "use_context_film": gate_cfg.use_context_film,
                "blend_mode": gate_cfg.blend_mode,
            },
        }
        torch.save(ckpt, last_ckpt)

        if val_dice > best_metric + early_min_delta:
            best_metric = val_dice
            ckpt["best_metric"] = best_metric
            torch.save(ckpt, best_ckpt)
            print(f"  New best: val_dice={val_dice:.4f}")
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        if early_enabled and epoch >= early_min_epochs and epochs_without_improve >= early_patience:
            msg = (
                f"Early stopping at epoch {epoch}: no improvement in {epochs_without_improve} "
                f"epochs on val_dice"
            )
            print(msg)
            flog.info(msg)
            break

    writer.close()
    print(f"Gating training done. Best val_dice={best_metric:.4f}")
    print(f"  Best ckpt: {best_ckpt}")


if __name__ == "__main__":
    main()

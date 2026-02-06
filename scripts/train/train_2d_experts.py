"""
2D expert training script (nnUNet / SwinUNETR / SegResNet).

支持单卡/DataParallel双卡, Windows原生兼容, 不依赖torchrun/DDP.

Usage:
    python scripts/train/train_2d_experts.py \
        --exp configs/2d/exp/exp_msd_task03_liver.yaml \
        --training configs/2d/training.yaml \
        --models configs/2d/models.yaml \
        --augs configs/2d/augs.yaml --fold 0 --gpus 0,1
"""
from __future__ import annotations

import argparse
import platform
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from seg_moe.data.dataset_2d import SegmentationDataset2D
from seg_moe.data.indexing import infer_image_channels, infer_num_classes
from seg_moe.models.factory_2d import build_expert, expert_name, list_experts
from seg_moe.training.engine import train_model
from seg_moe.utils.config import apply_debug_overrides, load_config, merge_configs, resolve_run_dir
from seg_moe.utils.io import ensure_dir, load_jsonl
from seg_moe.utils.seed import seed_everything

_IS_WINDOWS = platform.system() == "Windows"
_DEFAULT_NUM_WORKERS = 2 if _IS_WINDOWS else 4


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
        raise FileNotFoundError(f"Splits not found: {path}. Run scripts/data/make_splits.py first.")
    return load_jsonl(path)


def _parse_gpu_ids(gpus_str: str | None) -> tuple[torch.device, list[int]]:
    if not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        return torch.device("cpu"), []
    if gpus_str:
        gpu_ids = [int(x.strip()) for x in gpus_str.split(",")]
    else:
        gpu_ids = list(range(torch.cuda.device_count()))
    torch.cuda.set_device(gpu_ids[0])
    device = torch.device(f"cuda:{gpu_ids[0]}")
    print(f"Using GPU(s): {gpu_ids}")
    return device, gpu_ids


def _wrap_dp(model: torch.nn.Module, gpu_ids: list[int]) -> torch.nn.Module:
    if len(gpu_ids) > 1:
        print(f"  -> DataParallel on GPUs {gpu_ids}")
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    return model


def _resolve_resume(resume_arg: str, last_ckpt: Path, best_ckpt: Path) -> str | None:
    if not isinstance(resume_arg, str):
        return None
    low = resume_arg.lower()
    if low in {"last", "best"}:
        return str(last_ckpt if low == "last" else best_ckpt)
    elif low not in {"none", ""}:
        return resume_arg
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Train 2D experts (DataParallel, Windows-compatible)")
    ap.add_argument("--exp", required=True, help="Experiment config YAML")
    ap.add_argument("--training", required=True, help="Training config YAML")
    ap.add_argument("--models", required=True, help="Models config YAML (models.yaml)")
    ap.add_argument("--augs", required=True, help="Augmentation config YAML")
    ap.add_argument("--debug", default=None, help="Debug overrides YAML")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--layer", default="layer1", help="layer1|layer2")
    ap.add_argument("--dataset-config", default=None)
    ap.add_argument("--resume", default="none", help="none|last|best|/path/to/ckpt.pt")
    ap.add_argument("--skip-if-done", action="store_true")
    ap.add_argument("--gpus", type=str, default=None)
    ap.add_argument("--amp", action="store_true", default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    exp_cfg = load_config(args.exp)
    dataset_cfg = load_config(args.dataset_config or exp_cfg["dataset"]["config"])
    training_cfg = load_config(args.training)
    models_cfg = load_config(args.models)
    augs_cfg = load_config(args.augs)
    debug_cfg = load_config(args.debug) if args.debug else None

    merged = merge_configs(exp_cfg, {"training": training_cfg, "models": models_cfg,
                                     "augs": augs_cfg, "dataset_cfg": dataset_cfg})
    merged = apply_debug_overrides(merged, debug_cfg)

    if args.amp is not None:
        training_cfg.setdefault("amp", {})["enabled"] = True
    if args.grad_accum is not None:
        training_cfg["gradient_accumulation_steps"] = args.grad_accum
    if args.num_workers is not None:
        training_cfg.setdefault("dataloader", {})["num_workers"] = args.num_workers

    seed = args.seed or int(merged.get("seed", 42))
    det = merged.get("deterministic", {})
    seed_everything(seed, deterministic=bool(det.get("torch_deterministic", True)),
                    cudnn_benchmark=bool(det.get("cudnn_benchmark", False)))

    run_dir = resolve_run_dir(exp_cfg)
    ensure_dir(run_dir)
    device, gpu_ids = _parse_gpu_ids(args.gpus)

    num_classes = infer_num_classes(dataset_cfg)
    in_channels = infer_image_channels(dataset_cfg)
    rows = _load_splits(dataset_cfg)
    fold = int(args.fold)

    limits = merged.get("datalimits", {}) or {}
    train_ds = SegmentationDataset2D(
        [r for r in rows if r.get("split") == f"train_fold{fold}"],
        dataset_cfg, augs_cfg=augs_cfg, is_train=True, limit=limits.get("limit_train_samples"))
    val_ds = SegmentationDataset2D(
        [r for r in rows if r.get("split") == f"val_fold{fold}"],
        dataset_cfg, augs_cfg=augs_cfg, is_train=False, limit=limits.get("limit_val_samples"))

    bs = int((merged.get("training", {}) or {}).get("batch_size", training_cfg.get("batch_size", 8)))
    epochs = int((merged.get("training", {}) or {}).get("epochs", training_cfg.get("epochs", 300)))
    training_cfg = {**training_cfg, "batch_size": bs, "epochs": epochs}

    nw = int(training_cfg.get("dataloader", {}).get("num_workers", _DEFAULT_NUM_WORKERS))
    pm = bool(training_cfg.get("dataloader", {}).get("pin_memory", True))
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=pm)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pm)

    layer = getattr(args, 'layer', 'layer1') or 'layer1'
    expert_cfgs = list_experts(models_cfg)

    print("=" * 60)
    print(f"Training {len(expert_cfgs)} experts")
    for ec in expert_cfgs:
        print(f"  - {ec['name']} ({ec['type']})")
    print("=" * 60)

    for ec in expert_cfgs:
        name = expert_name(ec)
        tag = f"{layer}/fold{fold}/{name}"
        ckpt_dir = Path(run_dir) / "checkpoints" / layer / f"fold{fold}" / name
        best_ckpt = ckpt_dir / "best.pt"
        last_ckpt = ckpt_dir / "last.pt"

        if args.skip_if_done and best_ckpt.exists():
            print(f"Skip (exists): {best_ckpt}")
            continue

        resume_from = _resolve_resume(args.resume, last_ckpt, best_ckpt)
        print(f"\nBuilding expert: {name} (type={ec['type']})")
        model = build_expert(ec, in_channels=in_channels, num_classes=num_classes)
        model = _wrap_dp(model, gpu_ids)

        train_model(model, train_loader, val_loader, num_classes=num_classes,
                    training_cfg=training_cfg, run_dir=run_dir, tag=tag,
                    device=device, resume_from=resume_from)


if __name__ == "__main__":
    main()

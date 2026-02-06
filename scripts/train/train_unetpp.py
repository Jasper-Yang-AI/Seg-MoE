"""
UNet++ training script (DataParallel, Windows-compatible).

Usage:
    python scripts/train/train_unetpp.py --exp configs/2d/exp/exp_msd_task03_liver.yaml \
        --training configs/2d/training.yaml --models configs/2d/models.yaml \
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
from seg_moe.models.factory_2d import build_smp_model
from seg_moe.training.engine import train_model
from seg_moe.utils.config import load_config, resolve_run_dir
from seg_moe.utils.io import ensure_dir, load_jsonl

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
        raise FileNotFoundError(f"Splits not found: {path}")
    return load_jsonl(path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train UNet++ (DataParallel, Windows-compatible)")
    ap.add_argument("--exp", required=True)
    ap.add_argument("--training", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--augs", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--which", choices=["best", "last"], default="best")
    ap.add_argument("--dataset-config", default=None)
    ap.add_argument("--resume", default="none", help="none|last|best|/path/to/ckpt.pt")
    ap.add_argument("--skip-if-done", action="store_true")
    ap.add_argument("--gpus", type=str, default=None, help="GPU IDs, e.g. '0,1'")
    args = ap.parse_args()

    exp_cfg = load_config(args.exp)
    dataset_cfg = load_config(args.dataset_config or exp_cfg["dataset"]["config"])
    training_cfg = load_config(args.training)
    models_cfg = load_config(args.models)
    augs_cfg = load_config(args.augs)

    run_dir = resolve_run_dir(exp_cfg)
    ensure_dir(run_dir)

    # GPU setup
    if torch.cuda.is_available():
        if args.gpus:
            gpu_ids = [int(x.strip()) for x in args.gpus.split(",")]
        else:
            gpu_ids = list(range(torch.cuda.device_count()))
        torch.cuda.set_device(gpu_ids[0])
        device = torch.device(f"cuda:{gpu_ids[0]}")
        print(f"Using GPU(s): {gpu_ids}")
    else:
        device = torch.device("cpu")
        gpu_ids = []
        print("CUDA not available, using CPU")

    rows = _load_splits(dataset_cfg)
    fold = int(args.fold)
    train_rows = [r for r in rows if r.get("split") == f"train_fold{fold}"]
    val_rows = [r for r in rows if r.get("split") == f"val_fold{fold}"]

    num_classes = infer_num_classes(dataset_cfg)
    in_channels = infer_image_channels(dataset_cfg)

    train_ds = SegmentationDataset2D(train_rows, dataset_cfg, augs_cfg=augs_cfg, is_train=True)
    val_ds = SegmentationDataset2D(val_rows, dataset_cfg, augs_cfg=augs_cfg, is_train=False)

    bs = int(training_cfg.get("batch_size", 8))
    num_workers = int(training_cfg.get("dataloader", {}).get("num_workers", _DEFAULT_NUM_WORKERS))
    pin_memory = bool(training_cfg.get("dataloader", {}).get("pin_memory", True))

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    unetpp_cfg = models_cfg.get("unetpp", {})
    backbone = str(unetpp_cfg.get("backbone", "resnet34")).lower()
    encoder_weights = models_cfg.get("smp", {}).get("encoder_weights", "imagenet")

    tag = f"unetpp/fold{fold}/unetpp-{backbone}"
    ckpt_dir = Path(run_dir) / "checkpoints" / "unetpp" / f"fold{fold}" / f"unetpp-{backbone}"
    best_ckpt = ckpt_dir / "best.pt"
    last_ckpt = ckpt_dir / "last.pt"
    if args.skip_if_done and best_ckpt.exists():
        print(f"Skip (exists): {best_ckpt}")
        return

    resume_from = None
    if isinstance(args.resume, str) and args.resume.lower() in {"last", "best"}:
        resume_from = str(last_ckpt if args.resume.lower() == "last" else best_ckpt)
    elif isinstance(args.resume, str) and args.resume.lower() not in {"none", ""}:
        resume_from = args.resume

    model = build_smp_model("unetplusplus", backbone, in_channels=in_channels, classes=num_classes, encoder_weights=encoder_weights)

    # DataParallel for multi-GPU
    if len(gpu_ids) > 1:
        print(f"  → DataParallel on GPUs {gpu_ids}")
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)

    print(f"Training UNet++ on {device} with {len(gpu_ids)} GPU(s)")
    train_model(
        model, train_loader, val_loader,
        num_classes=num_classes, training_cfg=training_cfg,
        run_dir=run_dir, tag=tag, device=device, resume_from=resume_from,
    )


if __name__ == "__main__":
    main()
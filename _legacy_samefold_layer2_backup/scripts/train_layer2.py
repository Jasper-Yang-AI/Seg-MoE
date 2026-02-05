from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from seg_moe.data.dataset_2d import SegmentationDataset2D
from seg_moe.data.indexing import infer_image_channels, infer_num_classes
from seg_moe.models.factory_2d import build_smp_model, expert_name, list_experts
from seg_moe.training.engine import train_model
from seg_moe.utils.config import load_config, resolve_run_dir
from seg_moe.utils.io import ensure_dir, load_jsonl

from seg_moe.data.layer2_dataset import Layer2Dataset


def _load_splits(dataset_cfg: dict) -> list[dict]:
    splits_dir = Path(dataset_cfg["paths"]["splits_dir"])
    stype = dataset_cfg["split"]["type"]
    if stype == "holdout20_then_5fold":
        path = splits_dir / "splits_holdout20_5fold.jsonl"
    elif stype == "train_5fold_test_fixed":
        path = splits_dir / "splits_train5fold_testfixed.jsonl"
    else:
        path = splits_dir / "splits_5fold.jsonl"
    return load_jsonl(path)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--training", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--augs", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--which", choices=["best", "last"], default="best")
    ap.add_argument("--dataset-config", default=None, help="Override dataset config YAML (instead of exp.dataset.config)")
    ap.add_argument(
        "--resume",
        default="none",
        help="Resume mode: none|last|best|/path/to/ckpt.pt (per-expert).",
    )
    ap.add_argument("--skip-if-done", action="store_true", help="Skip training if best checkpoint already exists")
    args = ap.parse_args()

    exp_cfg = load_config(args.exp)
    dataset_cfg = load_config(args.dataset_config or exp_cfg["dataset"]["config"])
    training_cfg = load_config(args.training)
    models_cfg = load_config(args.models)
    augs_cfg = load_config(args.augs)

    run_dir = resolve_run_dir(exp_cfg)
    ensure_dir(run_dir)

    cache_root = Path(exp_cfg["layering"]["cache_root"].replace("${exp_name}", exp_cfg["exp_name"]))
    layer1_cache = cache_root / "layer1_probs" / dataset_cfg["name"]

    rows = _load_splits(dataset_cfg)
    fold = int(args.fold)
    train_split = f"train_fold{fold}"
    val_split = f"val_fold{fold}"

    train_rows = [r for r in rows if r.get("split") == train_split]
    val_rows = [r for r in rows if r.get("split") == val_split]

    num_classes = infer_num_classes(dataset_cfg)
    base_in = infer_image_channels(dataset_cfg)

    experts = list_experts(models_cfg)
    K = len(experts)
    in_channels = base_in + K * num_classes

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build datasets
    train_base = SegmentationDataset2D(train_rows, dataset_cfg, augs_cfg=augs_cfg, is_train=True)
    val_base = SegmentationDataset2D(val_rows, dataset_cfg, augs_cfg=augs_cfg, is_train=False)

    train_ds = Layer2Dataset(train_base, layer1_cache / train_split, num_experts=K, num_classes=num_classes)
    val_ds = Layer2Dataset(val_base, layer1_cache / val_split, num_experts=K, num_classes=num_classes)

    bs = int(training_cfg.get("batch_size", 8))
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=int(training_cfg.get("dataloader", {}).get("num_workers", 4)))
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=int(training_cfg.get("dataloader", {}).get("num_workers", 4)))

    encoder_weights = models_cfg.get("smp", {}).get("encoder_weights", "imagenet")

    for arch, backbone in experts:
        tag = f"layer2/fold{fold}/{expert_name(arch, backbone)}"
        ckpt_dir = Path(run_dir) / "checkpoints" / "layer2" / f"fold{fold}" / expert_name(arch, backbone)
        best_ckpt = ckpt_dir / "best.pt"
        last_ckpt = ckpt_dir / "last.pt"
        if args.skip_if_done and best_ckpt.exists():
            print(f"Skip (exists): {best_ckpt}")
            continue

        resume_from = None
        if isinstance(args.resume, str) and args.resume.lower() in {"last", "best"}:
            resume_from = str(last_ckpt if args.resume.lower() == "last" else best_ckpt)
        elif isinstance(args.resume, str) and args.resume.lower() not in {"none", ""}:
            resume_from = args.resume

        model = build_smp_model(arch, backbone, in_channels=in_channels, classes=num_classes, encoder_weights=encoder_weights)
        print(f"Training layer2 expert: {tag}")
        train_model(
            model,
            train_loader,
            val_loader,
            num_classes=num_classes,
            training_cfg=training_cfg,
            run_dir=run_dir,
            tag=tag,
            device=device,
            resume_from=resume_from,
        )


if __name__ == "__main__":
    main()

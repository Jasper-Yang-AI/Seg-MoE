"""
Cache per-sample logit maps [K, M, H, W] as .npz files.

Usage:
    python scripts/inference/cache_probs.py \
        --exp configs/2d/exp/exp_msd_task03_liver.yaml \
        --models configs/2d/models.yaml --layer layer1 --fold 0
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from seg_moe.data.dataset_2d import SegmentationDataset2D
from seg_moe.data.indexing import infer_image_channels, infer_num_classes
from seg_moe.models.factory_2d import build_expert, expert_name, list_experts
from seg_moe.utils.checkpoint import load_trusted_model_state_dict
from seg_moe.utils.config import load_config, resolve_run_dir
from seg_moe.utils.io import ensure_dir, load_jsonl


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


def _ckpt_path(run_dir: Path, layer: str, fold: int, ex: str, which: str = "best") -> Path:
    return run_dir / "checkpoints" / layer / f"fold{fold}" / ex / f"{which}.pt"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--layer", choices=["layer1", "layer2"], required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--which", choices=["best", "last"], default="best")
    ap.add_argument("--split", default=None)
    ap.add_argument("--dataset-config", default=None)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    exp_cfg = load_config(args.exp)
    dataset_cfg = load_config(args.dataset_config or exp_cfg["dataset"]["config"])
    models_cfg = load_config(args.models)

    run_dir = resolve_run_dir(exp_cfg)
    cache_root = Path(exp_cfg["layering"]["cache_root"].replace("${exp_name}", exp_cfg["exp_name"]))
    cache_dtype_str = str(exp_cfg.get("layering", {}).get("cache_dtype", "float16")).lower()
    cache_dtype = np.float16 if cache_dtype_str in {"float16", "fp16", "f16"} else np.float32

    out_root = cache_root / f"{args.layer}_logits" / dataset_cfg["name"]
    ensure_dir(out_root)

    rows = _load_splits(dataset_cfg)
    splits = sorted({r["split"] for r in rows})
    if args.split:
        target_splits = [args.split]
    else:
        target_splits = [s for s in splits if s.startswith("train_fold") or s.startswith("val_fold") or s == "test"]

    num_classes = infer_num_classes(dataset_cfg)
    in_channels = infer_image_channels(dataset_cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    expert_cfgs = list_experts(models_cfg)
    expert_names = [expert_name(ec) for ec in expert_cfgs]

    # ── Pre-load all expert models (outside the sample loop) ──
    expert_models: list[torch.nn.Module] = []
    for ec in expert_cfgs:
        ex = expert_name(ec)
        ckpt = _ckpt_path(run_dir, args.layer, args.fold, ex, which=args.which)
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
        model = build_expert(ec, in_channels=in_channels, num_classes=num_classes)
        model.load_state_dict(load_trusted_model_state_dict(ckpt), strict=True)
        model.to(device).eval()
        print(f"  Loaded {ex} from {ckpt}")
        expert_models.append(model)

    for split in target_splits:
        split_rows = [r for r in rows if r.get("split") == split]
        ds = SegmentationDataset2D(split_rows, dataset_cfg, augs_cfg=None, is_train=False)
        dl = DataLoader(ds, batch_size=1, shuffle=False)

        split_dir = out_root / split
        ensure_dir(split_dir)

        for img_t, _, meta in tqdm(dl, desc=f"cache {args.layer} {split}"):
            sample_id = meta["id"][0] if isinstance(meta["id"], list) else meta["id"]
            out_path = split_dir / f"{sample_id}.npz"
            if args.skip_existing and out_path.exists():
                continue

            logits_k = []
            for model in expert_models:
                with torch.no_grad():
                    logits = model(img_t.to(device))
                    logit_np = logits.detach().cpu().numpy()[0]
                logits_k.append(logit_np.astype(cache_dtype))

            np.savez_compressed(out_path, logits=np.stack(logits_k, axis=0))


if __name__ == "__main__":
    main()

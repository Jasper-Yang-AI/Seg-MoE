from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from seg_moe.data.dataset_2d import SegmentationDataset2D
from seg_moe.data.layer2_dataset import Layer2Dataset
from seg_moe.data.indexing import infer_image_channels, infer_num_classes
from seg_moe.models.factory_2d import build_smp_model, expert_name, list_experts
from seg_moe.utils.config import load_config, resolve_run_dir
from seg_moe.utils.io import ensure_dir, load_jsonl


def _load_splits(dataset_cfg: dict) -> tuple[list[dict], str]:
    splits_dir = Path(dataset_cfg["paths"]["splits_dir"])
    stype = dataset_cfg["split"]["type"]
    if stype == "holdout20_then_5fold":
        path = splits_dir / "splits_holdout20_5fold.jsonl"
    elif stype == "train_5fold_test_fixed":
        path = splits_dir / "splits_train5fold_testfixed.jsonl"
    else:
        path = splits_dir / "splits_5fold.jsonl"
    return load_jsonl(path), stype


def _ckpt_path(run_dir: Path, layer: str, fold: int, expert: str, which: str = "best") -> Path:
    return run_dir / "checkpoints" / layer / f"fold{fold}" / expert / f"{which}.pt"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--layer", choices=["layer1", "layer2"], required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--which", choices=["best", "last"], default="best")
    ap.add_argument("--split", default=None, help="split name (e.g., train_fold0/val_fold0/test). If omitted caches train/val/test if present.")
    ap.add_argument("--dataset-config", default=None, help="Override dataset config YAML (instead of exp.dataset.config)")
    ap.add_argument("--skip-existing", action="store_true", help="Skip caching if the .npz file already exists")
    args = ap.parse_args()

    exp_cfg = load_config(args.exp)
    dataset_cfg = load_config(args.dataset_config or exp_cfg["dataset"]["config"])
    models_cfg = load_config(args.models)

    run_dir = resolve_run_dir(exp_cfg)
    cache_root = Path(exp_cfg["layering"]["cache_root"].replace("${exp_name}", exp_cfg["exp_name"]))
    out_root = cache_root / f"{args.layer}_probs" / dataset_cfg["name"]
    ensure_dir(out_root)

    rows, _ = _load_splits(dataset_cfg)
    splits = sorted({r["split"] for r in rows})

    target_splits = [args.split] if args.split else [s for s in splits if s.startswith("train_fold") or s.startswith("val_fold") or s == "test"]

    num_classes = infer_num_classes(dataset_cfg)
    base_in_channels = infer_image_channels(dataset_cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    experts = list_experts(models_cfg)
    num_experts = len(experts)
    encoder_weights = models_cfg.get("smp", {}).get("encoder_weights", "imagenet")

    for split in target_splits:
        split_rows = [r for r in rows if r.get("split") == split]
        base_ds = SegmentationDataset2D(split_rows, dataset_cfg, augs_cfg=None, is_train=False)

        if args.layer == "layer2":
            # Layer2 consumes I* = concat(image, layer1_probs[K,M,H,W] flattened to K*M channels)
            layer1_split_dir = cache_root / "layer1_probs" / dataset_cfg["name"] / split
            if not layer1_split_dir.exists():
                raise FileNotFoundError(
                    f"Missing layer1 cache for split '{split}': {layer1_split_dir}. "
                    f"Run cache_probs with --layer layer1 for the same split first."
                )
            ds = Layer2Dataset(base_ds, layer1_split_dir, num_experts=num_experts, num_classes=num_classes)
            in_channels = base_in_channels + num_experts * num_classes
        else:
            ds = base_ds
            in_channels = base_in_channels
        dl = DataLoader(ds, batch_size=1, shuffle=False)

        split_dir = out_root / split
        ensure_dir(split_dir)

        for arch, backbone in experts:
            ex_name = expert_name(arch, backbone)
            ckpt = _ckpt_path(run_dir, args.layer, args.fold, ex_name, which=args.which)
            if not ckpt.exists():
                raise FileNotFoundError(f"Missing checkpoint: {ckpt}")

        # cache per-sample: [K,M,H,W]
        for img_t, _, meta in tqdm(dl, desc=f"cache {args.layer} {split}"):
            sample_id = meta["id"][0] if isinstance(meta["id"], list) else meta["id"]
            out_path = split_dir / f"{sample_id}.npz"
            if args.skip_existing and out_path.exists():
                continue
            probs_k = []
            for arch, backbone in experts:
                ex_name = expert_name(arch, backbone)
                ckpt = _ckpt_path(run_dir, args.layer, args.fold, ex_name, which=args.which)
                model = build_smp_model(arch, backbone, in_channels=in_channels, classes=num_classes, encoder_weights=encoder_weights)
                state = torch.load(ckpt, map_location="cpu")
                model.load_state_dict(state["model"], strict=True)
                model.to(device)
                model.eval()

                with torch.no_grad():
                    logits = model(img_t.to(device))
                    probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]  # [M,H,W]
                probs_k.append(probs.astype(np.float16))

            stacked = np.stack(probs_k, axis=0)  # [K,M,H,W]
            np.savez_compressed(out_path, probs=stacked)


if __name__ == "__main__":
    main()

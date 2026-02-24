"""
Generate Layer2 Out-of-Fold (OOF) probability predictions.

正确流程:  Layer1 train → L1 OOF → Layer2 train → **L2 OOF** → Gating train → Eval

For each fold k, loads the Layer2 checkpoints trained on train_fold{k}
and predicts on val_fold{k}.  Layer2 experts take concatenated input:
  x = [image, L1_oof_probs, entropy, disagreement]  (in_channels=16)

The L1 OOF probs are already generated (no leakage) and serve as
the additional channels for Layer2 input.

Output: cache/oof/layer2/fold_{k}/{sample_id}.npz  — probs shape [K, M, H, W]
Manifest: cache/oof/layer2/oof_manifest_layer2.jsonl

Usage:
    python scripts/inference/generate_layer2_oof.py \\
        --exp configs/2d/exp/exp_msd_task03_liver.yaml \\
        --models configs/2d/models.yaml --which best

    # With TTA & batch inference:
    python scripts/inference/generate_layer2_oof.py \\
        --exp configs/2d/exp/exp_msd_task03_liver.yaml \\
        --models configs/2d/models.yaml --which best \\
        --batch-size 32 --tta
"""
from __future__ import annotations

import argparse
import datetime as _dt
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from seg_moe.data.indexing import infer_image_channels, infer_num_classes
from seg_moe.data.layer2_oof_dataset import Layer2OOFDataset
from seg_moe.models.factory_2d import build_expert, expert_name, list_experts
from seg_moe.utils.config import load_config, resolve_run_dir
from seg_moe.utils.io import ensure_dir, load_jsonl, save_jsonl


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


def _l2_ckpt_path(run_dir: Path, fold: int, ex: str, which: str) -> Path:
    """Layer2 checkpoint: checkpoints/layer2/fold{k}/{expert}/best.pt"""
    return run_dir / "checkpoints" / "layer2" / f"fold{fold}" / ex / f"{which}.pt"


def _find_folds(rows: list[dict]) -> list[int]:
    folds = set()
    for r in rows:
        s = str(r.get("split", ""))
        if s.startswith("val_fold"):
            try:
                folds.add(int(s.replace("val_fold", "")))
            except Exception:
                pass
    return sorted(folds)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate Layer2 OOF probability predictions")
    ap.add_argument("--exp", required=True, help="Experiment config")
    ap.add_argument("--models", required=True, help="Expert model config")
    ap.add_argument("--which", choices=["best", "last"], default="best")
    ap.add_argument("--fold", type=int, default=None, help="Single fold (default: all)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=32,
                    help="Inference batch size (default: 32, 2D slices)")
    ap.add_argument("--tta", action="store_true",
                    help="Enable Test-Time Augmentation (horizontal + vertical flip)")
    ap.add_argument("--no-uncertainty", action="store_true",
                    help="Disable uncertainty channels (match Layer2 training)")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    exp_cfg = load_config(args.exp)
    dataset_cfg = load_config(exp_cfg["dataset"]["config"])
    models_cfg = load_config(args.models)
    run_dir = Path(resolve_run_dir(exp_cfg))

    cache_root = Path(
        str(exp_cfg["layering"].get("cache_root", f"runs/${{exp_name}}/cache"))
        .replace("${exp_name}", exp_cfg["exp_name"])
    )
    cache_dtype_str = str(exp_cfg.get("layering", {}).get("cache_dtype", "float16")).lower()
    cache_dtype = np.float16 if cache_dtype_str in {"float16", "fp16", "f16"} else np.float32

    # ── Layer1 OOF manifest (needed as input to Layer2) ──
    l1_oof_manifest_path = Path(
        str(exp_cfg.get("layering", {}).get(
            "oof_manifest_path",
            cache_root / "oof" / "layer1" / "oof_manifest.jsonl",
        )).replace("${exp_name}", exp_cfg["exp_name"])
    )
    if not l1_oof_manifest_path.exists():
        raise FileNotFoundError(
            f"Layer1 OOF manifest not found: {l1_oof_manifest_path}. "
            "Run scripts/inference/generate_layer1_oof.py first."
        )

    # ── Layer2 OOF output paths ──
    l2_oof_cache_dir = Path(
        str(exp_cfg.get("layering", {}).get(
            "l2_oof_cache_dir",
            cache_root / "oof" / "layer2",
        )).replace("${exp_name}", exp_cfg["exp_name"])
    )
    l2_oof_manifest_path = Path(
        str(exp_cfg.get("layering", {}).get(
            "l2_oof_manifest_path",
            l2_oof_cache_dir / "oof_manifest_layer2.jsonl",
        )).replace("${exp_name}", exp_cfg["exp_name"])
    )
    ensure_dir(l2_oof_cache_dir)

    rows = _load_splits(dataset_cfg)
    folds = [int(args.fold)] if args.fold is not None else _find_folds(rows)

    num_classes = infer_num_classes(dataset_cfg)
    base_in = infer_image_channels(dataset_cfg)
    expert_cfgs = list_experts(models_cfg)
    K = len(expert_cfgs)

    # ── Layer2 input channels (must match training) ──
    add_uncertainty = not args.no_uncertainty
    extra_uncertainty_ch = (1 + num_classes) if add_uncertainty else 0
    in_channels = base_in + K * num_classes + extra_uncertainty_ch
    unc_str = f" + uncertainty({extra_uncertainty_ch}ch)" if add_uncertainty else ""
    print(f"Layer2 OOF | in_channels={in_channels} (base={base_in} + {K}x{num_classes}{unc_str})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    existing_map: Dict[str, dict] = {}
    if args.skip_existing and l2_oof_manifest_path.exists():
        for r in load_jsonl(l2_oof_manifest_path):
            existing_map[(int(r.get("sample_fold", -1)), str(r.get("sample_id")))] = r

    all_records: List[Dict[str, Any]] = []
    ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for fold in folds:
        val_split = f"val_fold{fold}"
        val_rows = [r for r in rows if r.get("split") == val_split]
        fold_dir = ensure_dir(l2_oof_cache_dir / f"fold_{fold}")

        # ── Pre-load all Layer2 expert models for this fold ──
        ckpt_map: Dict[str, str] = {}
        expert_models: List[torch.nn.Module] = []
        for ec in expert_cfgs:
            ex = expert_name(ec)
            ckpt = _l2_ckpt_path(run_dir, fold, ex, args.which)
            if not ckpt.exists():
                raise FileNotFoundError(
                    f"Missing Layer2 checkpoint: {ckpt}. "
                    "Run scripts/train/train_layer2.py first."
                )
            ckpt_map[ex] = str(ckpt)
            model = build_expert(ec, in_channels=in_channels, num_classes=num_classes)
            state = torch.load(ckpt, map_location="cpu", weights_only=True)
            model.load_state_dict(state["model"], strict=True)
            model.to(device).eval()
            print(f"  [fold{fold}] Loaded L2-{ex} from {ckpt}")
            expert_models.append(model)

        # ── Layer2OOFDataset: provides [image + L1_probs + uncertainty] ──
        ds = Layer2OOFDataset(
            val_rows, dataset_cfg, l1_oof_manifest_path,
            expected_num_experts=K,
            augs_cfg=None,
            is_train=False,
            limit=args.limit,
            add_uncertainty=add_uncertainty,
        )
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

        for x_batch, _, meta_batch in tqdm(dl, desc=f"L2-OOF fold{fold}"):
            # x_batch: [B, in_channels, H, W]  (image + L1_probs + uncertainty)
            B = x_batch.shape[0]
            ids = meta_batch["id"] if isinstance(meta_batch["id"], list) else [meta_batch["id"]]
            if len(ids) != B:
                ids = [ids[0]] * B

            # ── Collect per-expert Layer2 predictions ──
            batch_expert_probs = []  # List[np.ndarray], each [B, M, H, W]
            for model in expert_models:
                with torch.no_grad():
                    logits = model(x_batch.to(device))  # [B, M, H, W]
                    probs = torch.softmax(logits, dim=1)

                    if args.tta:
                        # Horizontal flip
                        x_h = torch.flip(x_batch.to(device), dims=[-1])
                        logits_h = model(x_h)
                        probs_h = torch.softmax(torch.flip(logits_h, dims=[-1]), dim=1)
                        # Vertical flip
                        x_v = torch.flip(x_batch.to(device), dims=[-2])
                        logits_v = model(x_v)
                        probs_v = torch.softmax(torch.flip(logits_v, dims=[-2]), dim=1)
                        probs = (probs + probs_h + probs_v) / 3.0

                    batch_expert_probs.append(probs.cpu().numpy())  # [B, M, H, W]

            # ── Save per-sample ──
            for b_idx in range(B):
                sample_id = ids[b_idx]
                out_path = fold_dir / f"{sample_id}.npz"
                rel_path = Path(f"fold_{fold}") / f"{sample_id}.npz"

                rec: Dict[str, Any] = {
                    "sample_id": str(sample_id),
                    "sample_fold": int(fold),
                    "predictor_fold": int(fold),
                    "split": val_split,
                    "prob_path": rel_path.as_posix(),
                    "num_classes": int(num_classes),
                    "num_experts": len(expert_cfgs),
                    "experts": list(ckpt_map.keys()),
                    "model_ckpt_paths": dict(ckpt_map),
                    "which": args.which,
                    "seed": int(exp_cfg.get("seed", 0)),
                    "timestamp": ts,
                    "exp_name": str(exp_cfg.get("exp_name")),
                    "dataset": str(dataset_cfg.get("name")),
                    "tta": args.tta,
                    "layer": "layer2",
                    "l1_oof_manifest": str(l1_oof_manifest_path),
                }

                if args.skip_existing and out_path.exists():
                    k = (int(fold), str(sample_id))
                    all_records.append(existing_map.get(k, rec))
                    continue

                probs_k = [ep[b_idx].astype(cache_dtype) for ep in batch_expert_probs]
                np.savez_compressed(out_path, probs=np.stack(probs_k, axis=0))  # [K,M,H,W]
                all_records.append(rec)

    # ── Save manifest ──
    merged = dict(existing_map)
    for r in all_records:
        merged[(int(r.get("sample_fold", -1)), str(r.get("sample_id")))] = r
    ensure_dir(l2_oof_manifest_path.parent)
    save_jsonl(l2_oof_manifest_path, list(merged.values()))
    print(f"Saved Layer2 OOF manifest: {l2_oof_manifest_path} (rows={len(merged)})")


if __name__ == "__main__":
    main()

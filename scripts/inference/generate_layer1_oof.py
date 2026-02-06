"""
Generate Layer1 Out-of-Fold (OOF) probability predictions.

For each fold k, loads the layer1 checkpoints trained on train_fold{k}
and predicts on val_fold{k}.  This ensures no data leakage for Layer2 training.

Usage:
    python scripts/inference/generate_layer1_oof.py \
        --exp configs/2d/exp/exp_msd_task03_liver.yaml \
        --models configs/2d/models.yaml --which best
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

from seg_moe.data.dataset_2d import SegmentationDataset2D
from seg_moe.data.indexing import infer_image_channels, infer_num_classes
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


def _ckpt_path(run_dir: Path, fold: int, ex: str, which: str) -> Path:
    return run_dir / "checkpoints" / "layer1" / f"fold{fold}" / ex / f"{which}.pt"


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--which", choices=["best", "last"], default="best")
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    exp_cfg = load_config(args.exp)
    dataset_cfg = load_config(exp_cfg["dataset"]["config"])
    models_cfg = load_config(args.models)
    run_dir = resolve_run_dir(exp_cfg)

    cache_root = Path(str(exp_cfg["layering"].get("cache_root", f"runs/${{exp_name}}/cache")).replace("${exp_name}", exp_cfg["exp_name"]))
    cache_dtype_str = str(exp_cfg.get("layering", {}).get("cache_dtype", "float16")).lower()
    cache_dtype = np.float16 if cache_dtype_str in {"float16", "fp16", "f16"} else np.float32

    oof_cache_dir = Path(str(exp_cfg["layering"].get("oof_cache_dir", cache_root / "oof" / "layer1")).replace("${exp_name}", exp_cfg["exp_name"]))
    oof_manifest_path = Path(str(exp_cfg["layering"].get("oof_manifest_path", oof_cache_dir / "oof_manifest.jsonl")).replace("${exp_name}", exp_cfg["exp_name"]))
    ensure_dir(oof_cache_dir)

    rows = _load_splits(dataset_cfg)
    folds = [int(args.fold)] if args.fold is not None else _find_folds(rows)

    num_classes = infer_num_classes(dataset_cfg)
    in_channels = infer_image_channels(dataset_cfg)
    expert_cfgs = list_experts(models_cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    existing_map = {}
    if args.skip_existing and oof_manifest_path.exists():
        for r in load_jsonl(oof_manifest_path):
            existing_map[(int(r.get("sample_fold", -1)), str(r.get("sample_id")))] = r

    all_records: List[Dict[str, Any]] = []
    ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for fold in folds:
        val_split = f"val_fold{fold}"
        val_rows = [r for r in rows if r.get("split") == val_split]
        fold_dir = ensure_dir(oof_cache_dir / f"fold_{fold}")

        # Check all checkpoints exist
        ckpt_map = {}
        for ec in expert_cfgs:
            ex = expert_name(ec)
            ckpt = _ckpt_path(run_dir, fold, ex, args.which)
            if not ckpt.exists():
                raise FileNotFoundError(f"Missing layer1 checkpoint: {ckpt}")
            ckpt_map[ex] = str(ckpt)

        ds = SegmentationDataset2D(val_rows, dataset_cfg, augs_cfg=None, is_train=False, limit=args.limit)
        dl = DataLoader(ds, batch_size=1, shuffle=False)

        for img_t, _, meta in tqdm(dl, desc=f"OOF fold{fold}"):
            sample_id = meta["id"][0] if isinstance(meta["id"], list) else meta["id"]
            out_path = fold_dir / f"{sample_id}.npz"
            rel_path = Path(f"fold_{fold}") / f"{sample_id}.npz"

            rec: Dict[str, Any] = {
                "sample_id": str(sample_id), "sample_fold": int(fold),
                "predictor_fold": int(fold), "split": val_split,
                "prob_path": rel_path.as_posix(),
                "num_classes": int(num_classes),
                "num_experts": len(expert_cfgs),
                "experts": list(ckpt_map.keys()),
                "model_ckpt_paths": dict(ckpt_map),
                "which": args.which, "seed": int(exp_cfg.get("seed", 0)),
                "timestamp": ts, "exp_name": str(exp_cfg.get("exp_name")),
                "dataset": str(dataset_cfg.get("name")),
            }

            if args.skip_existing and out_path.exists():
                k = (int(fold), str(sample_id))
                all_records.append(existing_map.get(k, rec))
                continue

            probs_k = []
            for ec in expert_cfgs:
                ex = expert_name(ec)
                model = build_expert(ec, in_channels=in_channels, num_classes=num_classes)
                state = torch.load(Path(ckpt_map[ex]), map_location="cpu")
                model.load_state_dict(state["model"], strict=True)
                model.to(device).eval()
                with torch.no_grad():
                    probs = torch.softmax(model(img_t.to(device)), dim=1).detach().cpu().numpy()[0]
                probs_k.append(probs.astype(cache_dtype))

            np.savez_compressed(out_path, probs=np.stack(probs_k, axis=0))
            all_records.append(rec)

    merged = dict(existing_map)
    for r in all_records:
        merged[(int(r.get("sample_fold", -1)), str(r.get("sample_id")))] = r

    ensure_dir(oof_manifest_path.parent)
    save_jsonl(oof_manifest_path, list(merged.values()))
    print(f"Saved OOF manifest: {oof_manifest_path} (rows={len(merged)})")


if __name__ == "__main__":
    main()

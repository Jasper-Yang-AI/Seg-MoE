"""
Generate Layer1 cached logits for validation or test splits.

Validation folds keep using the shared OOF cache/manifest because those
artifacts are consumed by Layer2 training without leakage.

For explicit non-validation splits such as ``test``, predictions are written to
an inference-only manifest so downstream evaluation can reuse the same cache
loading logic without polluting training OOF artifacts.
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
from seg_moe.data.oof import parse_val_fold, resolve_prediction_cache_paths
from seg_moe.models.factory_2d import build_expert, expert_name, list_experts
from seg_moe.utils.checkpoint import load_trusted_model_state_dict
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


def _record_key(record: dict) -> tuple[str, int, str]:
    return (
        str(record.get("split", "")),
        int(record.get("predictor_fold", -1)),
        str(record.get("sample_id", "")),
    )


def _resolve_targets(rows: list[dict], fold: int | None, split: str | None) -> list[tuple[int, str]]:
    if split is not None:
        split = str(split).strip()
        split_fold = parse_val_fold(split)
        if split == "test":
            if fold is None:
                raise ValueError("--fold is required when --split test to choose predictor checkpoints")
            return [(int(fold), split)]
        if split_fold is None:
            raise ValueError(f"Unsupported split={split!r}; expected val_fold{{k}} or test")
        if fold is not None and int(fold) != split_fold:
            raise ValueError(f"--fold {fold} conflicts with --split {split}")
        return [(split_fold, split)]

    if fold is not None:
        return [(int(fold), f"val_fold{int(fold)}")]

    return [(f, f"val_fold{f}") for f in _find_folds(rows)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--which", choices=["best", "last"], default="best")
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--split", type=str, default=None,
                    help="Target split to cache (for example val_fold0 or test)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=32,
                    help="Inference batch size (default: 32, 2D slices are small)")
    ap.add_argument("--tta", action="store_true",
                    help="Enable Test-Time Augmentation (horizontal + vertical flip)")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    exp_cfg = load_config(args.exp)
    dataset_cfg = load_config(exp_cfg["dataset"]["config"])
    models_cfg = load_config(args.models)
    run_dir = Path(resolve_run_dir(exp_cfg))

    rows = _load_splits(dataset_cfg)
    targets = _resolve_targets(rows, args.fold, args.split)
    manifest_paths = {
        resolve_prediction_cache_paths(exp_cfg, "layer1", predictor_fold=fold, split=split)[1]
        for fold, split in targets
    }
    if len(manifest_paths) != 1:
        raise ValueError(
            "This command can write only one manifest per run. "
            "Use separate invocations when mixing validation and non-validation splits."
        )
    manifest_path = next(iter(manifest_paths))
    ensure_dir(manifest_path.parent)

    cache_dtype_str = str(exp_cfg.get("layering", {}).get("cache_dtype", "float16")).lower()
    cache_dtype = np.float16 if cache_dtype_str in {"float16", "fp16", "f16"} else np.float32

    num_classes = infer_num_classes(dataset_cfg)
    in_channels = infer_image_channels(dataset_cfg)
    expert_cfgs = list_experts(models_cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    existing_map: Dict[tuple[str, int, str], dict] = {}
    if args.skip_existing and manifest_path.exists():
        for row in load_jsonl(manifest_path):
            existing_map[_record_key(row)] = row

    all_records: List[Dict[str, Any]] = []
    ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for fold, split in targets:
        split_rows = [r for r in rows if r.get("split") == split]
        if not split_rows:
            raise ValueError(f"No samples found for split={split}")

        cache_dir, _ = resolve_prediction_cache_paths(exp_cfg, "layer1", predictor_fold=fold, split=split)
        fold_dir = ensure_dir(cache_dir / f"fold_{fold}") if parse_val_fold(split) is not None else ensure_dir(cache_dir)
        sample_fold = int(fold) if parse_val_fold(split) is not None else -1

        ckpt_map: Dict[str, str] = {}
        expert_models: List[torch.nn.Module] = []
        for ec in expert_cfgs:
            ex = expert_name(ec)
            ckpt = _ckpt_path(run_dir, fold, ex, args.which)
            if not ckpt.exists():
                raise FileNotFoundError(f"Missing layer1 checkpoint: {ckpt}")
            ckpt_map[ex] = str(ckpt)
            model = build_expert(ec, in_channels=in_channels, num_classes=num_classes)
            model.load_state_dict(load_trusted_model_state_dict(ckpt), strict=True)
            model.to(device).eval()
            print(f"  [fold{fold}] Loaded {ex} from {ckpt}")
            expert_models.append(model)

        ds = SegmentationDataset2D(split_rows, dataset_cfg, augs_cfg=None, is_train=False, limit=args.limit)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

        for img_batch, _, meta_batch in tqdm(dl, desc=f"L1 cache fold{fold} {split}"):
            batch_size = img_batch.shape[0]
            ids = meta_batch["id"] if isinstance(meta_batch["id"], list) else [meta_batch["id"]]
            if len(ids) != batch_size:
                ids = [ids[0]] * batch_size

            batch_expert_logits = []
            img_batch = img_batch.to(device)
            for model in expert_models:
                with torch.no_grad():
                    logits = model(img_batch)

                    if args.tta:
                        logits_h = model(torch.flip(img_batch, dims=[-1]))
                        logits_h = torch.flip(logits_h, dims=[-1])
                        logits_v = model(torch.flip(img_batch, dims=[-2]))
                        logits_v = torch.flip(logits_v, dims=[-2])
                        logits = (logits + logits_h + logits_v) / 3.0

                    batch_expert_logits.append(logits.cpu().numpy())

            for batch_idx in range(batch_size):
                sample_id = ids[batch_idx]
                out_path = fold_dir / f"{sample_id}.npz"
                rel_path = out_path.relative_to(manifest_path.parent)

                record: Dict[str, Any] = {
                    "sample_id": str(sample_id),
                    "sample_fold": int(sample_fold),
                    "predictor_fold": int(fold),
                    "split": split,
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
                }

                if args.skip_existing and out_path.exists():
                    all_records.append(existing_map.get(_record_key(record), record))
                    continue

                logits_k = [arr[batch_idx].astype(cache_dtype) for arr in batch_expert_logits]
                np.savez_compressed(out_path, logits=np.stack(logits_k, axis=0))
                all_records.append(record)

    merged = dict(existing_map)
    for record in all_records:
        merged[_record_key(record)] = record

    save_jsonl(manifest_path, list(merged.values()))
    print(f"Saved Layer1 manifest: {manifest_path} (rows={len(merged)})")


if __name__ == "__main__":
    main()

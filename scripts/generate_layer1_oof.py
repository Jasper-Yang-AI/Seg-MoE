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
from seg_moe.models.factory_2d import build_smp_model, expert_name, list_experts
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


def _ckpt_path(run_dir: Path, fold: int, expert: str, which: str) -> Path:
    return run_dir / "checkpoints" / "layer1" / f"fold{fold}" / expert / f"{which}.pt"


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
    ap.add_argument("--fold", type=int, default=None, help="Only generate one fold (default: all folds found)")
    ap.add_argument("--limit", type=int, default=None, help="Optional limit per fold for smoke testing")
    ap.add_argument(
        "--experts-limit",
        type=int,
        default=None,
        help="Optional: only use the first N experts (debug only; must match layer2 if used for training)",
    )
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    exp_cfg = load_config(args.exp)
    dataset_cfg = load_config(exp_cfg["dataset"]["config"])
    models_cfg = load_config(args.models)

    run_dir = resolve_run_dir(exp_cfg)

    cache_root = Path(str(exp_cfg["layering"].get("cache_root", f"runs/${{exp_name}}/cache")).replace("${exp_name}", exp_cfg["exp_name"]))
    cache_dtype_str = str(exp_cfg.get("layering", {}).get("cache_dtype", "float16")).lower()
    if cache_dtype_str in {"float16", "fp16", "f16"}:
        cache_dtype = np.float16
    elif cache_dtype_str in {"float32", "fp32", "f32"}:
        cache_dtype = np.float32
    else:
        raise ValueError(f"Unsupported cache_dtype: {cache_dtype_str} (use float16 or float32)")
    oof_cache_dir = Path(str(exp_cfg["layering"].get("oof_cache_dir", cache_root / "oof" / "layer1")).replace("${exp_name}", exp_cfg["exp_name"]))
    oof_manifest_path = Path(str(exp_cfg["layering"].get("oof_manifest_path", oof_cache_dir / "oof_manifest.jsonl")).replace("${exp_name}", exp_cfg["exp_name"]))

    ensure_dir(oof_cache_dir)

    rows = _load_splits(dataset_cfg)
    folds = _find_folds(rows)
    if args.fold is not None:
        folds = [int(args.fold)]

    num_classes = infer_num_classes(dataset_cfg)
    in_channels = infer_image_channels(dataset_cfg)

    experts = list_experts(models_cfg)
    if args.experts_limit is not None:
        experts = experts[: int(args.experts_limit)]
    encoder_weights = models_cfg.get("smp", {}).get("encoder_weights", "imagenet")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    existing = []   
    existing_map = {}

    if args.skip_existing and oof_manifest_path.exists():
        existing = load_jsonl(oof_manifest_path)
        for r in existing :
            k = (int(r.get('sample_fold', -1)), str(r.get('sample_id')))
            existing_map[k] = r


    all_records: List[Dict[str, Any]] = []
    ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Validate split disjointness and generate OOF per fold
    for fold in folds:
        train_split = f"train_fold{fold}"
        val_split = f"val_fold{fold}"

        train_rows = [r for r in rows if r.get("split") == train_split]
        val_rows = [r for r in rows if r.get("split") == val_split]

        train_ids = {r["id"] for r in train_rows}
        val_ids = {r["id"] for r in val_rows}
        inter = train_ids.intersection(val_ids)
        print(f"[fold{fold}] train={len(train_rows)} val={len(val_rows)} intersection={len(inter)}")
        if inter:
            raise RuntimeError(f"Split leakage detected for fold{fold}: {len(inter)} overlapping ids (example: {next(iter(inter))})")

        fold_dir = ensure_dir(oof_cache_dir / f"fold_{fold}")

        # Ensure checkpoints exist
        ckpt_map = {}
        for arch, backbone in experts:
            ex = expert_name(arch, backbone)
            ckpt = _ckpt_path(run_dir, fold, ex, args.which)
            if not ckpt.exists():
                raise FileNotFoundError(f"Missing layer1 checkpoint for fold{fold}: {ckpt}")
            ckpt_map[ex] = str(ckpt)

        ds = SegmentationDataset2D(val_rows, dataset_cfg, augs_cfg=None, is_train=False, limit=args.limit)
        dl = DataLoader(ds, batch_size=1, shuffle=False)

        for img_t, _, meta in tqdm(dl, desc=f"OOF layer1 fold{fold} -> {val_split}"):
            sample_id = meta["id"][0] if isinstance(meta["id"], list) else meta["id"]
            out_path = fold_dir / f"{sample_id}.npz"
            rel_prob_path = Path(f"fold_{fold}") / f"{sample_id}.npz"
            if args.skip_existing and out_path.exists():
                k = (int(fold), str(sample_id))
                rec = existing_map.get(k)
                if rec is None:
                    rec = {
                        "sample_id": str(sample_id),
                        "sample_fold": int(fold),
                        "predictor_fold": int(fold),
                        "split": val_split,
                        "prob_path": rel_prob_path.as_posix(),
                        # "shape": list(stacked.shape),
                        "num_classes": int(num_classes),
                        "num_experts": int(len(experts)),
                        "experts": list(ckpt_map.keys()),
                        "experts_limit": int(args.experts_limit) if args.experts_limit is not None else None,
                        "model_ckpt_paths": dict(ckpt_map),
                        "which": args.which,
                        "seed": int(exp_cfg.get("seed", 0)),
                        "timestamp": ts,
                        "exp_name": str(exp_cfg.get("exp_name")),
                        "dataset": str(dataset_cfg.get("name")),
                    }
                all_records.append(rec)
                continue

            probs_k = []
            for arch, backbone in experts:
                ex = expert_name(arch, backbone)
                ckpt = Path(ckpt_map[ex])
                model = build_smp_model(
                    arch,
                    backbone,
                    in_channels=in_channels,
                    classes=num_classes,
                    encoder_weights=encoder_weights,
                )
                state = torch.load(ckpt, map_location="cpu")
                model.load_state_dict(state["model"], strict=True)
                model.to(device)
                model.eval()

                with torch.no_grad():
                    logits = model(img_t.to(device))
                    probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]  # [M,H,W]
                probs_k.append(probs.astype(cache_dtype))

            stacked = np.stack(probs_k, axis=0)  # [K,M,H,W]
            np.savez_compressed(out_path, probs=stacked)

            rec: Dict[str, Any] = {
                "sample_id": str(sample_id),
                "sample_fold": int(fold),
                "predictor_fold": int(fold),
                "split": val_split,
                "prob_path": rel_prob_path.as_posix(),
                # "shape": list(stacked.shape),
                "num_classes": int(num_classes),
                "num_experts": int(len(experts)),
                "experts": list(ckpt_map.keys()),
                "experts_limit": int(args.experts_limit) if args.experts_limit is not None else None,
                "model_ckpt_paths": dict(ckpt_map),
                "which": args.which,
                "seed": int(exp_cfg.get("seed", 0)),
                "timestamp": ts,
                "exp_name": str(exp_cfg.get("exp_name")),
                "dataset": str(dataset_cfg.get("name")),
            }
            all_records.append(rec)
    
    merged = dict(existing_map)
    # Leakage checks
    for r in all_records:
        k = (int(r.get("sample_fold", -1)), str(r.get("sample_id")))
        merged[k] = r
        if int(r["predictor_fold"]) != int(r["sample_fold"]):
            raise AssertionError(f"Leakage check failed: predictor_fold!=sample_fold for {r.get('sample_id')}")

    ensure_dir(oof_manifest_path.parent)
    save_jsonl(oof_manifest_path, list(merged.values()))
    print(f"Saved OOF manifest: {oof_manifest_path} (rows={len(list(merged.values()))})")

if __name__ == "__main__":
    main()

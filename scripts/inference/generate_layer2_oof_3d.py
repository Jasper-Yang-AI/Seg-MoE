"""
Generate Layer2 3D out-of-fold logits.

Each fold loads the trained Layer2 checkpoints for its experts and predicts on
the corresponding validation split using sliding-window inference.
"""
from __future__ import annotations

import argparse
import datetime as _dt
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from tqdm import tqdm

from seg_moe.data.layer2_oof_dataset_3d import Layer2OOFDataset3D
from seg_moe.models.factory_3d import build_expert_3d, expert_name_3d, list_experts_3d
from seg_moe.utils.checkpoint import load_trusted_model_state_dict
from seg_moe.utils.config import load_config, resolve_run_dir
from seg_moe.utils.io import ensure_dir, load_jsonl, save_jsonl


def _load_splits(dataset_cfg: dict) -> list[dict]:
    path = Path(dataset_cfg["paths"]["splits_dir"]) / "splits_train5fold_testfixed.jsonl"
    return load_jsonl(path)


def _find_folds(rows: list[dict]) -> list[int]:
    folds = set()
    for row in rows:
        split = str(row.get("split", ""))
        if split.startswith("val_fold"):
            try:
                folds.add(int(split.replace("val_fold", "")))
            except ValueError:
                pass
    return sorted(folds)


def _l2_ckpt_path(run_dir: Path, fold: int, ex_name: str, which: str) -> Path:
    return run_dir / "checkpoints" / "layer2" / f"fold{fold}" / ex_name / f"{which}.pt"


def _sliding_window(model, x, roi_size, sw_batch, overlap, device):
    try:
        from monai.inferers import sliding_window_inference

        return sliding_window_inference(
            x, roi_size, sw_batch, model, overlap=overlap, mode="gaussian", device=device
        )
    except ImportError:
        return model(x)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate 3D Layer2 OOF predictions")
    ap.add_argument("--exp", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--which", choices=["best", "last"], default="best")
    ap.add_argument("--fold", type=int, default=None, help="Single fold (default: all)")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--no-uncertainty", action="store_true",
                    help="Disable uncertainty channels when reconstructing Layer2 inputs")
    args = ap.parse_args()

    exp_cfg = load_config(args.exp)
    dataset_cfg = load_config(exp_cfg["dataset"]["config"])
    models_cfg = load_config(args.models)
    run_dir = Path(resolve_run_dir(exp_cfg))

    layering = exp_cfg["layering"]
    exp_name = exp_cfg["exp_name"]

    def _resolve(path_like: str) -> str:
        return path_like.replace("${exp_name}", exp_name)

    l1_oof_manifest = Path(_resolve(str(layering["oof_manifest_path"])))
    l2_cache_dir = Path(_resolve(str(layering["l2_oof_cache_dir"])))
    l2_manifest = Path(_resolve(str(layering["l2_oof_manifest_path"])))
    ensure_dir(l2_cache_dir)

    rows = _load_splits(dataset_cfg)
    folds = [int(args.fold)] if args.fold is not None else _find_folds(rows)
    num_classes = int(dataset_cfg["task"]["num_classes"])
    expert_cfgs = list_experts_3d(models_cfg)
    num_experts = len(expert_cfgs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sw_cfg = dataset_cfg.get("input", {})
    roi_size = tuple(int(s) for s in sw_cfg.get("roi_size", [128, 128, 64]))
    sw_batch = 2

    existing_map: dict[tuple[int, str], dict] = {}
    if args.skip_existing and l2_manifest.exists():
        for record in load_jsonl(l2_manifest):
            existing_map[(int(record["sample_fold"]), str(record["sample_id"]))] = record

    all_records: List[Dict[str, Any]] = []
    ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for fold in folds:
        val_rows = [row for row in rows if row.get("split") == f"val_fold{fold}"]
        print(f"\n[Layer2 OOF 3D] fold={fold} val_volumes={len(val_rows)}")

        val_ds = Layer2OOFDataset3D(
            samples=val_rows,
            dataset_cfg=dataset_cfg,
            oof_manifest_path=str(l1_oof_manifest),
            expected_num_experts=num_experts,
            is_train=False,
            add_uncertainty=not args.no_uncertainty,
        )
        in_channels = val_ds.in_channels

        fold_models: List[torch.nn.Module] = []
        for expert_cfg in expert_cfgs:
            name = expert_name_3d(expert_cfg)
            ckpt_path = _l2_ckpt_path(run_dir, fold, name, args.which)
            if not ckpt_path.exists():
                raise FileNotFoundError(
                    f"Layer2 checkpoint not found for {name} fold{fold}: {ckpt_path}. "
                    "Run scripts/train/train_layer2_3d.py before generating Layer2 OOF."
                )

            model = build_expert_3d(expert_cfg, in_channels=in_channels, num_classes=num_classes)
            model.load_state_dict(load_trusted_model_state_dict(ckpt_path), strict=False)
            model.eval().to(device)
            fold_models.append(model)
            print(f"  Loaded L2 {name} from {ckpt_path}")

        for sample_idx, sample in enumerate(tqdm(val_rows, desc=f"fold{fold}")):
            sample_id = str(sample["id"])
            key = (fold, sample_id)
            if key in existing_map:
                all_records.append(existing_map[key])
                continue

            image_t, _, _ = val_ds[sample_idx]
            image_t = image_t.unsqueeze(0).to(device)

            vol_logits: List[np.ndarray] = []
            for model in fold_models:
                with torch.no_grad():
                    with torch.amp.autocast("cuda", enabled=device.type == "cuda", dtype=torch.bfloat16):
                        raw = _sliding_window(model, image_t, roi_size, sw_batch, 0.5, device)
                vol_logits.append(raw.squeeze(0).float().cpu().numpy().astype(np.float16))

            out_path = l2_cache_dir / f"fold_{fold}" / f"{sample_id}.npz"
            ensure_dir(out_path.parent)
            np.savez_compressed(str(out_path), logits=np.stack(vol_logits, axis=0))

            all_records.append(
                {
                    "sample_id": sample_id,
                    "sample_fold": fold,
                    "predictor_fold": fold,
                    "prob_path": str(out_path),
                    "has_logits": True,
                    "num_classes": num_classes,
                    "num_experts": len(fold_models),
                    "created_at": ts,
                }
            )

    save_jsonl(l2_manifest, all_records)
    print(f"\n[Layer2 OOF 3D] Saved {len(all_records)} records -> {l2_manifest}")


if __name__ == "__main__":
    main()

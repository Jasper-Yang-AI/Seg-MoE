"""
Generate Layer1 3D Out-of-Fold (OOF) logit predictions.

For each fold k, loads the Layer1 checkpoints trained on train_fold{k}
and predicts on val_fold{k} using MONAI sliding-window inference.
Output: [K, M, D, H, W] logits per volume, saved as .npz.

Usage:
    python scripts/inference/generate_layer1_oof_3d.py \\
        --exp    configs/3d/exp/exp_prostate_local_3d.yaml \\
        --models configs/3d/models_3d.yaml \\
        --which best
"""
from __future__ import annotations

import argparse
import datetime as _dt
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from tqdm import tqdm

from seg_moe.data.dataset_3d import SegmentationDataset3D
from seg_moe.models.factory_3d import build_expert_3d, expert_name_3d, list_experts_3d
from seg_moe.utils.config import load_config, resolve_run_dir
from seg_moe.utils.io import ensure_dir, load_jsonl, save_jsonl


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_splits(dataset_cfg: dict) -> list[dict]:
    path = Path(dataset_cfg["paths"]["splits_dir"]) / "splits_train5fold_testfixed.jsonl"
    return load_jsonl(path)


def _find_folds(rows: list[dict]) -> list[int]:
    folds = set()
    for r in rows:
        s = str(r.get("split", ""))
        if s.startswith("val_fold"):
            try:
                folds.add(int(s.replace("val_fold", "")))
            except ValueError:
                pass
    return sorted(folds)


def _ckpt_path(run_dir: Path, fold: int, ex_name: str, which: str) -> Path:
    return run_dir / "checkpoints" / "layer1" / f"fold{fold}" / ex_name / f"{which}.pt"


def _sliding_window(model, x, roi_size, sw_batch, overlap, device):
    try:
        from monai.inferers import sliding_window_inference
        return sliding_window_inference(x, roi_size, sw_batch, model,
                                        overlap=overlap, mode="gaussian", device=device)
    except ImportError:
        return model(x)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Generate 3D Layer1 OOF predictions")
    ap.add_argument("--exp",          required=True)
    ap.add_argument("--models",       required=True)
    ap.add_argument("--which",        choices=["best", "last"], default="best")
    ap.add_argument("--fold",         type=int, default=None, help="Single fold (default: all)")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    exp_cfg     = load_config(args.exp)
    dataset_cfg = load_config(exp_cfg["dataset"]["config"])
    models_cfg  = load_config(args.models)
    run_dir     = Path(resolve_run_dir(exp_cfg))

    cache_root = Path(
        exp_cfg["layering"]["cache_root"].replace("${exp_name}", exp_cfg["exp_name"])
    )
    oof_cache_dir = Path(
        str(exp_cfg["layering"]["oof_cache_dir"]).replace("${exp_name}", exp_cfg["exp_name"])
    )
    oof_manifest_path = Path(
        str(exp_cfg["layering"]["oof_manifest_path"]).replace("${exp_name}", exp_cfg["exp_name"])
    )
    ensure_dir(oof_cache_dir)

    rows        = _load_splits(dataset_cfg)
    folds       = [int(args.fold)] if args.fold is not None else _find_folds(rows)
    num_classes = int(dataset_cfg["task"]["num_classes"])
    in_channels = int(dataset_cfg["input"].get("image_channels", 1))
    expert_cfgs = list_experts_3d(models_cfg)
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sw_cfg      = dataset_cfg.get("input", {})
    roi_size    = tuple(int(s) for s in sw_cfg.get("roi_size", [128, 128, 64]))
    sw_batch    = 2

    # Load existing manifest
    existing_map: dict[str, dict] = {}
    if args.skip_existing and oof_manifest_path.exists():
        for r in load_jsonl(oof_manifest_path):
            existing_map[(int(r["sample_fold"]), str(r["sample_id"]))] = r

    all_records: List[Dict[str, Any]] = []
    ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for fold in folds:
        val_rows = [r for r in rows if r.get("split") == f"val_fold{fold}"]
        print(f"\n[Layer1 OOF 3D] fold={fold}  val_volumes={len(val_rows)}")

        val_ds = SegmentationDataset3D(val_rows, dataset_cfg, is_train=False)

        # Load all experts for this fold
        fold_models: List[torch.nn.Module] = []
        for ec in expert_cfgs:
            name = expert_name_3d(ec)
            ckpt = _ckpt_path(run_dir, fold, name, args.which)
            if not ckpt.exists():
                raise FileNotFoundError(f"Checkpoint not found: {ckpt}. Train layer1 first.")
            m = build_expert_3d(ec, in_channels=in_channels, num_classes=num_classes)
            state = torch.load(ckpt, map_location="cpu", weights_only=True)
            m.load_state_dict(
                {k.removeprefix("module."): v for k, v in state["model"].items()}, strict=False
            )
            m.eval().to(device)
            fold_models.append(m)
            print(f"  Loaded {name} from {ckpt}")

        for si, sample in enumerate(tqdm(val_rows, desc=f"fold{fold}")):
            sid = str(sample["id"])
            key = (fold, sid)
            if key in existing_map:
                all_records.append(existing_map[key])
                continue

            img_t, _, meta = val_ds[si]
            img_t = img_t.unsqueeze(0).to(device)   # [1, C, D, H, W]

            # Collect logits from all experts
            vol_logits: List[np.ndarray] = []
            for m in fold_models:
                with torch.no_grad():
                    with torch.amp.autocast("cuda", enabled=device.type == "cuda", dtype=torch.bfloat16):
                        logits = _sliding_window(m, img_t, roi_size, sw_batch, 0.5, device)
                raw = logits.squeeze(0).float().cpu().numpy().astype(np.float16)
                vol_logits.append(raw)   # [M, D, H, W]

            stacked = np.stack(vol_logits, axis=0)  # [K, M, D, H, W]
            out_path = oof_cache_dir / f"fold_{fold}" / f"{sid}.npz"
            ensure_dir(out_path.parent)
            np.savez_compressed(str(out_path), logits=stacked)

            rec = {
                "sample_id":       sid,
                "sample_fold":     fold,
                "predictor_fold":  fold,
                "prob_path":       str(out_path),
                "num_classes":     num_classes,
                "num_experts":     len(fold_models),
                "created_at":      ts,
            }
            all_records.append(rec)

    save_jsonl(oof_manifest_path, all_records)
    print(f"\n[Layer1 OOF 3D] Saved {len(all_records)} records → {oof_manifest_path}")


if __name__ == "__main__":
    main()

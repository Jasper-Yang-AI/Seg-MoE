"""
Generate Layer2 3D Out-of-Fold (OOF) logit predictions.

For each fold k, loads the Layer2 checkpoints trained on train_fold{k}
and predicts on val_fold{k} using MONAI sliding-window inference.
Input per volume: original MRI channels + concatenated Layer1 OOF probs.
Output: [K, M, D, H, W] logits per volume, saved as .npz.

Usage:
    python scripts/inference/generate_layer2_oof_3d.py \\
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

from seg_moe.data.layer2_oof_dataset_3d import Layer2OOFDataset3D
from seg_moe.models.factory_3d import (
    build_expert_3d,
    expert_name_3d,
    list_experts_3d,
    transfer_layer1_to_layer2_3d,
)
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


def _l1_ckpt_path(run_dir: Path, fold: int, ex_name: str, which: str) -> Path:
    return run_dir / "checkpoints" / "layer1" / f"fold{fold}" / ex_name / f"{which}.pt"


def _l2_ckpt_path(run_dir: Path, fold: int, ex_name: str, which: str) -> Path:
    return run_dir / "checkpoints" / "layer2" / f"fold{fold}" / ex_name / f"{which}.pt"


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
    ap = argparse.ArgumentParser(description="Generate 3D Layer2 OOF predictions")
    ap.add_argument("--exp",           required=True)
    ap.add_argument("--models",        required=True)
    ap.add_argument("--which",         choices=["best", "last"], default="best")
    ap.add_argument("--fold",          type=int, default=None, help="Single fold (default: all)")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    exp_cfg     = load_config(args.exp)
    dataset_cfg = load_config(exp_cfg["dataset"]["config"])
    models_cfg  = load_config(args.models)
    run_dir     = Path(resolve_run_dir(exp_cfg))

    layering    = exp_cfg["layering"]
    exp_name    = exp_cfg["exp_name"]

    def _resolve(s: str) -> str:
        return s.replace("${exp_name}", exp_name)

    l1_oof_manifest = Path(_resolve(str(layering["oof_manifest_path"])))
    l2_cache_dir    = Path(_resolve(str(layering["l2_oof_cache_dir"])))
    l2_manifest     = Path(_resolve(str(layering["l2_oof_manifest_path"])))
    ensure_dir(l2_cache_dir)

    rows        = _load_splits(dataset_cfg)
    folds       = [int(args.fold)] if args.fold is not None else _find_folds(rows)
    num_classes = int(dataset_cfg["task"]["num_classes"])
    in_channels = int(dataset_cfg["input"].get("image_channels", 1))
    expert_cfgs = list_experts_3d(models_cfg)
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sw_cfg    = dataset_cfg.get("input", {})
    roi_size  = tuple(int(s) for s in sw_cfg.get("roi_size", [128, 128, 64]))
    sw_batch  = 2

    # Compute layer2 in_channels (mirrors Layer2OOFDataset3D.in_channels)
    K = len(expert_cfgs)
    M = num_classes
    uncertainty_channels = 1 + M  # entropy + disagreement
    l2_in_channels = in_channels + K * M + uncertainty_channels

    # Load existing OOF manifest (L2)
    existing_map: dict[tuple, dict] = {}
    if args.skip_existing and l2_manifest.exists():
        for r in load_jsonl(l2_manifest):
            existing_map[(int(r["sample_fold"]), str(r["sample_id"]))] = r

    all_records: List[Dict[str, Any]] = []
    ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for fold in folds:
        val_rows = [r for r in rows if r.get("split") == f"val_fold{fold}"]
        print(f"\n[Layer2 OOF 3D] fold={fold}  val_volumes={len(val_rows)}")

        val_ds = Layer2OOFDataset3D(
            samples=val_rows,
            dataset_cfg=dataset_cfg,
            oof_manifest_path=str(l1_oof_manifest),
            is_train=False,
        )
        in_ch_actual = val_ds.in_channels

        # Load Layer2 models for this fold
        fold_models: List[torch.nn.Module] = []
        for ec in expert_cfgs:
            name  = expert_name_3d(ec)
            l2_ck = _l2_ckpt_path(run_dir, fold, name, args.which)

            if l2_ck.exists():
                # Directly load Layer2 weights if available
                m = build_expert_3d(ec, in_channels=in_ch_actual, num_classes=num_classes)
                state = torch.load(l2_ck, map_location="cpu", weights_only=True)
                m.load_state_dict(
                    {k.removeprefix("module."): v for k, v in state["model"].items()},
                    strict=False,
                )
                print(f"  Loaded L2 {name} from {l2_ck}")
            else:
                # Fall back: load Layer1 and transfer weights
                l1_ck = _l1_ckpt_path(run_dir, fold, name, args.which)
                if not l1_ck.exists():
                    raise FileNotFoundError(
                        f"Neither L2 ({l2_ck}) nor L1 ({l1_ck}) checkpoint found for {name} fold{fold}. "
                        f"Run train_layer1_3d.py and train_layer2_3d.py first."
                    )
                m_l1 = build_expert_3d(ec, in_channels=in_channels, num_classes=num_classes)
                state_l1 = torch.load(l1_ck, map_location="cpu", weights_only=True)
                m_l1.load_state_dict(
                    {k.removeprefix("module."): v for k, v in state_l1["model"].items()},
                    strict=False,
                )
                m = build_expert_3d(ec, in_channels=in_ch_actual, num_classes=num_classes)
                transfer_layer1_to_layer2_3d(
                    m_l1, m,
                    base_in_channels=in_channels,
                    extra_in_channels=in_ch_actual - in_channels,
                )
                print(f"  Transferred L1→L2 {name} (no L2 ckpt found)")

            m.eval().to(device)
            fold_models.append(m)

        for si, sample in enumerate(tqdm(val_rows, desc=f"fold{fold}")):
            sid = str(sample["id"])
            key = (fold, sid)
            if key in existing_map:
                all_records.append(existing_map[key])
                continue

            img_t, _, meta = val_ds[si]     # [C2, D, H, W] with OOF channels
            img_t = img_t.unsqueeze(0).to(device)  # [1, C2, D, H, W]

            vol_logits: List[np.ndarray] = []

            for m in fold_models:
                with torch.no_grad():
                    with torch.amp.autocast("cuda", enabled=device.type == "cuda", dtype=torch.bfloat16):
                        raw = _sliding_window(m, img_t, roi_size, sw_batch, 0.5, device)
                # raw: [1, M, D, H, W]
                raw_np   = raw.squeeze(0).float().cpu().numpy()
                vol_logits.append(raw_np.astype(np.float16))

            out_path = l2_cache_dir / f"fold_{fold}" / f"{sid}.npz"
            ensure_dir(out_path.parent)

            stacked_logits = np.stack(vol_logits, axis=0)  # [K, M, D, H, W]
            np.savez_compressed(str(out_path), logits=stacked_logits)

            rec = {
                "sample_id":       sid,
                "sample_fold":     fold,
                "predictor_fold":  fold,
                "prob_path":       str(out_path),
                "has_logits":      True,
                "num_classes":     num_classes,
                "num_experts":     len(fold_models),
                "created_at":      ts,
            }
            all_records.append(rec)

    save_jsonl(l2_manifest, all_records)
    print(f"\n[Layer2 OOF 3D] Saved {len(all_records)} records → {l2_manifest}")


if __name__ == "__main__":
    main()

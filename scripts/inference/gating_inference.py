"""
Gating dynamic fusion inference for 2D experiments.

Expected pipeline:
    Layer1 train -> Layer1 OOF -> Layer2 train -> Layer2 OOF
    -> train_gating.py -> gating_inference.py -> eval_methods.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from seg_moe.data.indexing import infer_num_classes
from seg_moe.data.oof import get_oof_prob_path, load_oof_manifest
from seg_moe.evaluation.metrics_2d import compute_segmentation_metrics_batch
from seg_moe.gating.patch_gating_2d import PatchConvGate2D, PatchGatingConfig
from seg_moe.models.factory_2d import list_experts
from seg_moe.utils.config import load_config, resolve_run_dir
from seg_moe.utils.io import ensure_dir, load_jsonl
from seg_moe.utils.patches import compute_patch_positions, merge_patches_2d


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


def _load_gating_model(ckpt_path: Path, device: torch.device) -> PatchConvGate2D:
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    gc = state["gate_cfg"]
    cfg = PatchGatingConfig(
        num_experts=gc["num_experts"],
        num_classes=gc["num_classes"],
        patch_size=gc["patch_size"],
        stride=gc["stride"],
        hidden_dim=gc["hidden_dim"],
        score_hidden_dim=gc.get("score_hidden_dim", gc["hidden_dim"]),
        dropout=gc["dropout"],
        per_class=gc["per_class"],
        use_residual_head=gc.get("use_residual_head", True),
        use_entropy=gc.get("use_entropy", True),
        use_consensus_features=gc.get("use_consensus_features", True),
        use_disagreement_features=gc.get("use_disagreement_features", True),
        use_confidence_features=gc.get("use_confidence_features", True),
        blend_mode=gc.get("blend_mode", "gaussian"),
        temperature_start=float(state.get("temperature", gc.get("temperature_start", 2.0))),
        temperature_end=float(gc.get("temperature_end", 0.5)),
    )
    model = PatchConvGate2D(cfg)
    model.load_state_dict(state["model"])
    model.to(device)
    model.eval()
    return model


def infer_single_image(
    logits: np.ndarray,
    model: PatchConvGate2D,
    device: torch.device,
    temperature: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fuse one sample of Layer2 expert logits.

    Args:
        logits: [K, M, H, W] Layer2 expert logits.

    Returns:
        fused_logits: [M, H, W]
        weight_map: [K, H, W]
        weight_map_per_class: [K, M, H, W]
    """
    logits = np.asarray(logits, dtype=np.float32)
    if logits.ndim != 4:
        raise ValueError(f"logits must have shape [K, M, H, W], got {tuple(logits.shape)}")

    K, M, H, W = logits.shape
    cfg = model.cfg
    ps = cfg.patch_size
    stride = cfg.stride
    positions = compute_patch_positions(H, W, ps, stride)

    patches_logits = [
        logits[:, :, y : y + ps, x : x + ps]
        for y, x in positions
    ]
    logits_batch = torch.from_numpy(np.stack(patches_logits, axis=0)).float().to(device)

    with torch.no_grad():
        weights = model(logits_batch, temperature=temperature)
        fused_patches = model.fuse_logits(logits_batch, weights)

    fused_list = [fused_patches[i].cpu().numpy() for i in range(len(positions))]
    fused_map = merge_patches_2d(fused_list, positions, (H, W), ps, blend_mode=cfg.blend_mode)

    weights_np = weights.cpu().numpy()
    if weights_np.ndim == 3:
        weights_expert = weights_np.mean(axis=2)
        flat_patches = [
            np.broadcast_to(weights_np[i][:, :, None, None], (K, M, ps, ps)).copy().reshape(K * M, ps, ps)
            for i in range(len(positions))
        ]
        w_km = merge_patches_2d(flat_patches, positions, (H, W), ps, blend_mode=cfg.blend_mode)
        weight_map_per_class = w_km.reshape(K, M, H, W)
    else:
        weights_expert = weights_np
        weight_map_per_class = None

    expert_weight_patches = [
        np.broadcast_to(weights_expert[i][:, None, None], (K, ps, ps)).copy()
        for i in range(len(positions))
    ]
    weight_map = merge_patches_2d(expert_weight_patches, positions, (H, W), ps, blend_mode=cfg.blend_mode)

    if weight_map_per_class is None:
        weight_map_per_class = np.repeat(weight_map[:, None, :, :], M, axis=1)

    return fused_map, weight_map, weight_map_per_class


def main() -> None:
    ap = argparse.ArgumentParser(description="Gating dynamic fusion inference")
    ap.add_argument("--exp", required=True)
    ap.add_argument("--gating-config", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--split", type=str, default=None, help="Split to evaluate (default: val_fold{fold})")
    ap.add_argument("--temperature", type=float, default=None, help="Override gating temperature")
    ap.add_argument("--save-weights", action="store_true", help="Save per-expert weight maps")
    ap.add_argument("--save-weight-png", action="store_true", help="Save weight map PNGs")
    ap.add_argument("--gpus", type=str, default=None)
    args = ap.parse_args()

    exp_cfg = load_config(args.exp)
    models_cfg = load_config(args.models)
    dataset_cfg = load_config(exp_cfg["dataset"]["config"])

    num_classes = infer_num_classes(dataset_cfg)
    num_experts = len(list_experts(models_cfg))
    fold = int(args.fold)
    split = args.split or f"val_fold{fold}"

    if torch.cuda.is_available():
        gpu_id = int(args.gpus.split(",")[0]) if args.gpus else 0
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")

    run_dir = Path(resolve_run_dir(exp_cfg))
    ckpt_path = run_dir / "checkpoints" / "gating" / f"fold{fold}" / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Gating checkpoint not found: {ckpt_path}. Run scripts/train/train_gating.py first."
        )
    model = _load_gating_model(ckpt_path, device)
    if model.cfg.num_experts != num_experts:
        raise ValueError(
            f"Gating checkpoint expects {model.cfg.num_experts} experts, but models config defines {num_experts}."
        )
    if model.cfg.num_classes != num_classes:
        raise ValueError(
            f"Gating checkpoint expects {model.cfg.num_classes} classes, but dataset config defines {num_classes}."
        )

    if args.temperature is None:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        temperature = float(state.get("temperature", model.cfg.temperature_end))
    else:
        temperature = float(args.temperature)
    print(f"Gating inference | fold={fold} split={split} tau={temperature:.2f} domain=logits")

    cache_root = Path(exp_cfg["layering"]["cache_root"].replace("${exp_name}", exp_cfg["exp_name"]))
    oof_manifest_path = Path(
        str(
            exp_cfg.get("layering", {}).get(
                "l2_oof_manifest_path",
                cache_root / "oof" / "layer2" / "oof_manifest_layer2.jsonl",
            )
        ).replace("${exp_name}", exp_cfg["exp_name"])
    )
    if not oof_manifest_path.exists():
        raise FileNotFoundError(
            f"Layer2 OOF manifest not found: {oof_manifest_path}. "
            "Run scripts/inference/generate_layer2_oof.py first."
        )
    oof_map = load_oof_manifest(oof_manifest_path)

    rows = _load_splits(dataset_cfg)
    eval_rows = [r for r in rows if r.get("split") == split]
    print(f"Evaluating {len(eval_rows)} samples")

    out_dir = run_dir / "results" / "gating" / f"fold{fold}" / split
    ensure_dir(out_dir)
    if args.save_weights:
        ensure_dir(out_dir / "weight_maps")
        if args.save_weight_png:
            ensure_dir(out_dir / "weight_maps" / "png")

    label_map = {int(k): int(v) for k, v in dataset_cfg["task"].get("label_map", {}).items()}
    all_metrics: list[dict] = []

    for sample in tqdm(eval_rows, desc=f"gating inference {split}"):
        sample_id = str(sample["id"])
        if sample_id not in oof_map:
            continue

        cache = np.load(get_oof_prob_path(oof_map, sample_id))
        if "logits" in cache:
            logits_arr = cache["logits"].astype(np.float32)
        elif "probs" in cache:
            probs = cache["probs"].astype(np.float32)
            logits_arr = np.log(np.clip(probs, 1e-6, 1.0))
        else:
            raise KeyError(f"Layer2 OOF cache for {sample_id} is missing 'logits'/'probs'")

        fused, weight_map, weight_map_per_class = infer_single_image(logits_arr, model, device, temperature)

        mask = np.array(Image.open(sample["mask_path"]).convert("L"), dtype=np.uint8)
        if label_map:
            mapped = mask.copy()
            for src, dst in label_map.items():
                mapped[mask == src] = dst
            mask = mapped

        pred = np.argmax(fused, axis=0).astype(np.int64)
        fused_probs = torch.from_numpy(fused).unsqueeze(0).float().softmax(dim=1)
        mask_t = torch.from_numpy(mask.astype(np.int64)).unsqueeze(0)
        metrics = compute_segmentation_metrics_batch(fused_probs, mask_t, num_classes=num_classes)
        metrics["sample_id"] = sample_id
        all_metrics.append(metrics)

        np.savez_compressed(out_dir / f"{sample_id}.npz", fused=fused, pred=pred)

        if args.save_weights:
            np.savez_compressed(
                out_dir / "weight_maps" / f"{sample_id}.npz",
                weight_map=weight_map,
                weight_map_per_class=weight_map_per_class,
            )
            if args.save_weight_png:
                png_dir = out_dir / "weight_maps" / "png"
                for expert_idx in range(weight_map.shape[0]):
                    weight_u8 = (weight_map[expert_idx] * 255).clip(0, 255).astype(np.uint8)
                    Image.fromarray(weight_u8).save(png_dir / f"{sample_id}_expert{expert_idx}.png")

    if all_metrics:
        dice_values = [m.get("dice_mean", 0.0) for m in all_metrics]
        mean_dice = float(np.mean(dice_values))
        print(f"\nGating fusion | {split} | mean Dice = {mean_dice:.4f}")
        with open(out_dir / "metrics.json", "w") as f:
            json.dump(
                {
                    "split": split,
                    "fold": fold,
                    "mean_dice": mean_dice,
                    "n_samples": len(all_metrics),
                    "temperature": temperature,
                    "per_sample": all_metrics,
                },
                f,
                indent=2,
                default=str,
            )
        print(f"Results saved to {out_dir}")


if __name__ == "__main__":
    main()

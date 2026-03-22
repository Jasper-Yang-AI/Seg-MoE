"""Patch-gating inference with optional Layer1 priors and anatomy context."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from seg_moe.data.gating_patch_dataset import (
    build_layer1_semantic_maps,
    build_position_channels,
    extract_slice_index,
)
from seg_moe.data.oof import get_oof_prob_path, load_oof_manifest
from seg_moe.data.indexing import infer_num_classes
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
    """Load trained gating model from checkpoint."""
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    gc = state["gate_cfg"]
    cfg = PatchGatingConfig(
        num_experts=gc["num_experts"],
        num_classes=gc["num_classes"],
        image_channels=gc.get("image_channels", 3),
        patch_size=gc["patch_size"],
        stride=gc["stride"],
        hidden_dim=gc["hidden_dim"],
        dropout=gc["dropout"],
        per_class=gc["per_class"],
        use_residual_head=gc.get("use_residual_head", True),
        use_entropy=gc.get("use_entropy", True),
        use_consensus_features=gc.get("use_consensus_features", True),
        use_disagreement_features=gc.get("use_disagreement_features", True),
        use_confidence_features=gc.get("use_confidence_features", True),
        use_prior_agreement_features=gc.get("use_prior_agreement_features", False),
        use_layer1_semantics=gc.get("use_layer1_semantics", False),
        use_image_context=gc.get("use_image_context", False),
        use_position_channels=gc.get("use_position_channels", False),
        use_slice_position=gc.get("use_slice_position", False),
        use_context_film=gc.get("use_context_film", True),
        blend_mode=gc.get("blend_mode", "gaussian"),
        temperature_start=float(state.get("temperature", gc.get("temperature_start", 2.0))),
        temperature_end=float(gc.get("temperature_end", 0.5)),
    )
    model = PatchConvGate2D(cfg)
    model.load_state_dict(state["model"])
    model.to(device)
    model.eval()
    temperature = float(state.get("temperature", cfg.temperature_end))
    return model, temperature


def _read_image(path: str, image_channels: int) -> np.ndarray:
    img = Image.open(path)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    if image_channels == 1:
        img = img.convert("L")
        arr = np.array(img, dtype=np.uint8)[:, :, None]
    else:
        img = img.convert("RGB")
        arr = np.array(img, dtype=np.uint8)
    arr = imagenet_normalize(arr)
    return np.transpose(arr.astype(np.float32), (2, 0, 1))


def _load_probs_or_logits(npz_path: Path) -> np.ndarray:
    data = np.load(npz_path)
    if "logits" in data:
        logits = data["logits"].astype(np.float32)
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        return exp_logits / (exp_logits.sum(axis=1, keepdims=True) + 1e-8)
    if "probs" in data:
        return data["probs"].astype(np.float32)
    raise KeyError(f"OOF cache missing 'logits'/'probs': {npz_path}")


def _build_slice_pos_map(samples: list[dict]) -> dict[str, float]:
    patient_to_z: dict[str, list[int]] = {}
    for sample in samples:
        sid = str(sample["id"])
        pid = str(sample.get("patient_id") or sid)
        z = extract_slice_index(sid)
        if z is not None:
            patient_to_z.setdefault(pid, []).append(z)

    slice_pos: dict[str, float] = {}
    for sample in samples:
        sid = str(sample["id"])
        pid = str(sample.get("patient_id") or sid)
        z = extract_slice_index(sid)
        if z is None or pid not in patient_to_z:
            slice_pos[sid] = 0.5
            continue
        z_vals = patient_to_z[pid]
        z_min = min(z_vals)
        z_max = max(z_vals)
        slice_pos[sid] = 0.5 if z_max <= z_min else float((z - z_min) / (z_max - z_min))
    return slice_pos


def _build_extra_maps(
    sample: dict,
    model_cfg: PatchGatingConfig,
    spatial_shape: tuple[int, int],
    l1_oof_map: dict | None,
    slice_pos_map: dict[str, float],
) -> dict[str, np.ndarray | float]:
    height, width = spatial_shape
    extra: dict[str, np.ndarray | float] = {}

    if model_cfg.use_image_context:
        extra["image"] = _read_image(sample["image_path"], model_cfg.image_channels)

    if model_cfg.use_layer1_semantics:
        if l1_oof_map is None:
            raise ValueError("Gate requires Layer1 semantic priors, but Layer1 OOF manifest is unavailable")
        probs = _load_probs_or_logits(get_oof_prob_path(l1_oof_map, str(sample["id"])))
        semantic = build_layer1_semantic_maps(probs)
        m = model_cfg.num_classes
        extra["layer1_mean"] = semantic[:m]
        extra["layer1_entropy"] = semantic[m : m + 1]
        extra["layer1_disagreement"] = semantic[m + 1 :]

    if model_cfg.use_position_channels:
        extra["coords"] = build_position_channels(height, width)

    if model_cfg.use_slice_position:
        extra["slice_pos"] = float(slice_pos_map.get(str(sample["id"]), 0.5))

    return extra


def infer_single_image(
    logits: np.ndarray,
    model: PatchConvGate2D,
    device: torch.device,
    *,
    temperature: float = 0.5,
    extra_maps: dict[str, np.ndarray | float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run gating inference on a single image.

    Args:
        logits: [K, M, H, W] Layer2 expert logits.
        model: trained gating model.
        device: torch device.
        temperature: softmax temperature.
        extra_maps: optional full-image context maps.

    Returns:
        fused_logits: [M, H, W]
        weight_map: [K, H, W]
        weight_map_per_class: [K, M, H, W]
    """
    k, m, height, width = logits.shape
    cfg = model.cfg
    patch_size = cfg.patch_size
    stride = cfg.stride

    positions = compute_patch_positions(H, W, ps, stride)

    # Build patches
    logits_flat = logits.reshape(K * M, H, W)   # gate input [K*M, H, W]
    patches_input = []
    patches_logits = []
    for y, x in positions:
        patches_input.append(logits_flat[:, y : y + ps, x : x + ps])
        patches_logits.append(logits[:, :, y : y + ps, x : x + ps])

    input_batch = torch.from_numpy(np.stack(patches_input)).float().to(device)
    logits_batch = torch.from_numpy(np.stack(patches_logits)).float().to(device)

    with torch.no_grad():
        weights = model(input_batch, temperature=temperature)       # [N, K] or [N, K, M]
        fused_patches = model.fuse_logits(logits_batch, weights)    # [N, M, pH, pW]

    fused_list = [fused_patches[i].cpu().numpy() for i in range(len(positions))]
    fused_map = merge_patches_2d(fused_list, positions, (height, width), patch_size, blend_mode=cfg.blend_mode)

    weights_np = weights.cpu().numpy()
    if weights_np.ndim == 3:
        weights_expert = weights_np.mean(axis=2)
        flat_patches = [
            np.broadcast_to(weights_np[i][:, :, None, None], (K, M, ps, ps)).copy().reshape(C_km, ps, ps)
            for i in range(len(positions))
        ]
        w_km = merge_patches_2d(flat_patches, positions, (height, width), patch_size, blend_mode=cfg.blend_mode)
        weight_map_per_class = w_km.reshape(k, m, height, width)
    else:
        weights_expert = weights_np
        weight_map_per_class = None

    w_expert_patches = [
        np.broadcast_to(weights_expert[i][:, None, None], (K, ps, ps)).copy()
        for i in range(len(positions))
    ]
    weight_map = merge_patches_2d(w_expert_patches, positions, (H, W), ps, blend_mode=cfg.blend_mode)

    if weight_map_per_class is None:
        weight_map_per_class = np.repeat(weight_map[:, None, :, :], M, axis=1)  # [K,M,H,W]

    return fused_map, weight_map, weight_map_per_class
    """Run gating inference on a single image.

    Args:
        probs:  [K, M, H, W] Layer2 expert probability maps (float32)
        model:  trained gating model
        device: torch device
        temperature: gating softmax temperature
        logits: [K, M, H, W] raw logits (required when model.cfg.fusion_domain == "logits")

    Returns:
        fused_map:   [M, H, W]  fused prediction map (probs or logits, depending on fusion_domain)
        weight_map:  [K, H, W]  per-expert spatial weight map (blended, averaged over classes)
        weight_map_per_class: [K, M, H, W]  per-expert per-class spatial weight map
    """
    K, M, H, W = probs.shape
    cfg = model.cfg
    ps = cfg.patch_size
    stride = cfg.stride
    fusion_domain = cfg.fusion_domain
    input_domain = cfg.input_domain

    positions = compute_patch_positions(H, W, ps, stride)

    # Prepare flattened input (probs, logits, or concat)
    probs_flat = probs.reshape(K * M, H, W)
    if input_domain == "logits":
        if logits is None:
            raise ValueError("input_domain='logits' but logits=None. Pass logits array.")
        logits_flat = logits.reshape(K * M, H, W)
        gate_input = logits_flat
    elif input_domain == "probs+logits":
        if logits is None:
            raise ValueError("input_domain='probs+logits' but logits=None. Pass logits array.")
        logits_flat = logits.reshape(K * M, H, W)
        gate_input = np.concatenate([probs_flat, logits_flat], axis=0)  # [K*2M, H, W]
    else:
        gate_input = probs_flat  # [K*M, H, W]

    # Collect patches
    patches_input = []
    patches_probs = []
    patches_logits = []
    for y, x in positions:
        patches_input.append(gate_input[:, y : y + ps, x : x + ps])
        patches_probs.append(probs[:, :, y : y + ps, x : x + ps])
        if logits is not None:
            patches_logits.append(logits[:, :, y : y + ps, x : x + ps])

    input_batch = torch.from_numpy(np.stack(patches_input, axis=0)).float().to(device)
    probs_batch = torch.from_numpy(np.stack(patches_probs, axis=0)).float().to(device)
    if patches_logits:
        logits_batch = torch.from_numpy(np.stack(patches_logits, axis=0)).float().to(device)

    with torch.no_grad():
        weights = model(input_batch, temperature=temperature)  # [N, K] or [N, K, M]
        if fusion_domain == "logits" and patches_logits:
            fused_patches = model.fuse_logits(logits_batch, weights)  # [N, M, pH, pW]
        else:
            fused_patches = model.fuse_probs(probs_batch, weights)    # [N, M, pH, pW]

    # Merge fused patches
    fused_list = [fused_patches[i].cpu().numpy() for i in range(len(positions))]
    fused_map = merge_patches_2d(fused_list, positions, (H, W), ps, blend_mode=cfg.blend_mode)

    # Build per-expert weight map — PRESERVE per-class resolution
    weights_np = weights.cpu().numpy()  # [N, K] or [N, K, M]
    if weights_np.ndim == 3:
        # per_class=True: [N, K, M]
        # Per-expert collapsed (mean over classes) → [N, K]
        weights_expert = weights_np.mean(axis=2)      # [N, K]
        # Per-class per-expert maps → [K, M, H, W]
        per_class_patches = []
        for i in range(len(positions)):
            w_km = weights_np[i]           # [K, M]
            w_spatial = np.broadcast_to(
                w_km[:, :, None, None], (K, M, ps, ps)
            ).copy()                        # [K, M, pH, pW]
            per_class_patches.append(w_spatial)
        # Merge per-class weight patches: treat [K*M] as channels
        C_km = K * M
        flat_patches = [p.reshape(C_km, ps, ps) for p in per_class_patches]
        w_km_spatial = merge_patches_2d(flat_patches, positions, (H, W), ps, blend_mode=cfg.blend_mode)
        weight_map_per_class = w_km_spatial.reshape(K, M, H, W)
    else:
        weights_expert = weights_np    # [N, K]
        # Expand [K] → [K, M] by replicating for all classes
        weight_map_per_class = None    # will construct below from weight_map

    # Merge per-expert collapsed weight map
    w_expert_patches = []
    for i in range(len(positions)):
        w = weights_expert[i]   # [K]
        w_spatial = np.broadcast_to(w[:, None, None], (K, ps, ps)).copy()
        w_expert_patches.append(w_spatial)
    weight_map = merge_patches_2d(w_expert_patches, positions, (H, W), ps, blend_mode=cfg.blend_mode)

    if weight_map_per_class is None:
        # Replicate weight_map [K,H,W] over M classes
        weight_map_per_class = np.repeat(weight_map[:, None, :, :], M, axis=1)  # [K,M,H,W]

    return fused_map, weight_map, weight_map_per_class


def main() -> None:
    ap = argparse.ArgumentParser(description="Gating dynamic fusion inference")
    ap.add_argument("--exp", required=True)
    ap.add_argument("--gating-config", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--split", type=str, default=None,
                    help="Split to evaluate (default: val_fold{fold})")
    ap.add_argument("--temperature", type=float, default=None,
                    help="Override gating temperature (default: from checkpoint)")
    ap.add_argument("--save-weights", action="store_true",
                    help="Save per-expert gating weight maps (.npz + PNG heatmaps)")
    ap.add_argument("--save-weight-png", action="store_true",
                    help="Also save per-expert weight maps as PNG heatmaps (requires --save-weights)")
    ap.add_argument("--gpus", type=str, default=None)
    args = ap.parse_args()

    exp_cfg = load_config(args.exp)
    gating_cfg_raw = load_config(args.gating_config)
    models_cfg = load_config(args.models)
    dataset_cfg = load_config(exp_cfg["dataset"]["config"])

    num_classes = infer_num_classes(dataset_cfg)
    K = len(list_experts(models_cfg))
    fold = args.fold
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

    temperature = args.temperature
    if temperature is None:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        temperature = float(state.get("temperature", model.cfg.temperature_end))
    print(f"Gating inference | fold={fold} split={split} τ={temperature:.2f} domain=logits")

    # OOF probs (来源: Layer2 OOF, 非 Layer1)
    cache_root = Path(
        exp_cfg["layering"]["cache_root"].replace("${exp_name}", exp_cfg["exp_name"])
    )
    oof_manifest_path = Path(
        str(
            exp_cfg.get("layering", {}).get(
                "l2_oof_manifest_path",
                cache_root / "oof" / "layer2" / "oof_manifest_layer2.jsonl",
            )
        ).replace("${exp_name}", exp_cfg["exp_name"])
    )
    if not l2_oof_manifest_path.exists():
        raise FileNotFoundError(
            f"Layer2 OOF manifest not found: {l2_oof_manifest_path}. "
            "Run scripts/inference/generate_layer2_oof.py first."
        )
    l2_oof_map = load_oof_manifest(l2_oof_manifest_path)

    l1_oof_map = None
    if model.cfg.use_layer1_semantics:
        l1_oof_manifest_path = Path(
            str(
                exp_cfg.get("layering", {}).get(
                    "oof_manifest_path",
                    cache_root / "oof" / "layer1" / "oof_manifest.jsonl",
                )
            ).replace("${exp_name}", exp_cfg["exp_name"])
        )
        if not l1_oof_manifest_path.exists():
            raise FileNotFoundError(
                f"Layer1 OOF manifest not found: {l1_oof_manifest_path}. "
                "Gate config requires Layer1 semantic priors."
            )
        l1_oof_map = load_oof_manifest(l1_oof_manifest_path)

    rows = _load_splits(dataset_cfg)
    eval_rows = [r for r in rows if r.get("split") == split]
    slice_pos_map = _build_slice_pos_map(eval_rows)
    print(f"Evaluating {len(eval_rows)} samples")

    out_dir = run_dir / "results" / "gating" / f"fold{fold}" / split
    ensure_dir(out_dir)
    if args.save_weights:
        ensure_dir(out_dir / "weight_maps")
        if args.save_weight_png:
            ensure_dir(out_dir / "weight_maps" / "png")

    label_map = {int(k): int(v) for k, v in dataset_cfg["task"].get("label_map", {}).items()}
    all_metrics: list[dict] = []

    for s in tqdm(eval_rows, desc=f"gating inference {split}"):
        sid = str(s["id"])
        if sid not in oof_map:
            continue

        probs_data = np.load(get_oof_prob_path(oof_map, sid))
        if "logits" in probs_data:
            logits_arr = probs_data["logits"].astype(np.float32)
        else:
            # Legacy fallback: log-odds from probs
            p = probs_data["probs"].astype(np.float32).clip(1e-6, 1 - 1e-6)
            logits_arr = np.log(p / (1 - p))

        fused, weight_map, weight_map_per_class = infer_single_image(
            logits_arr, model, device, temperature,
        )

        mask = np.array(Image.open(sample["mask_path"]).convert("L"), dtype=np.uint8)
        if label_map:
            mapped = mask.copy()
            for src, dst in label_map.items():
                mapped[mask == src] = dst
            mask = mapped

        pred = np.argmax(fused, axis=0).astype(np.int64)
        fused_probs = torch.from_numpy(fused).unsqueeze(0).float().softmax(dim=1)
        mask_t = torch.from_numpy(mask.astype(np.int64)).unsqueeze(0)
        metrics = compute_segmentation_metrics_batch(
            fused_probs, mask_t, num_classes=num_classes,
        )
        metrics["sample_id"] = sid
        all_metrics.append(metrics)

        # Save predictions
        np.savez_compressed(out_dir / f"{sid}.npz", fused=fused, pred=pred)

        # Save weight maps
        if args.save_weights:
            np.savez_compressed(
                out_dir / "weight_maps" / f"{sid}.npz",
                weight_map=weight_map,              # [K, H, W]  averaged over classes
                weight_map_per_class=weight_map_per_class,  # [K, M, H, W]  class-resolved
            )
            if args.save_weight_png:
                png_dir = out_dir / "weight_maps" / "png"
                for k in range(weight_map.shape[0]):
                    w_k = weight_map[k]             # [H, W]  in [0, 1]
                    w_uint8 = (w_k * 255).clip(0, 255).astype(np.uint8)
                    Image.fromarray(w_uint8).save(png_dir / f"{sid}_expert{k}.png")

    # Aggregate metrics
    if all_metrics:
        mean_dice = float(np.mean([m.get("dice_mean", 0.0) for m in all_metrics]))
        print(f"\nGating fusion | {split} | mean Dice = {mean_dice:.4f}")

        # Save metrics
        with open(out_dir / "metrics.json", "w") as f:
            json.dump(
                {"split": split, "fold": fold, "mean_dice": mean_dice,
                 "n_samples": len(all_metrics), "temperature": temperature,
                 "per_sample": all_metrics},
                f, indent=2, default=str,
            )
        print(f"Results saved to {out_dir}")


if __name__ == "__main__":
    main()

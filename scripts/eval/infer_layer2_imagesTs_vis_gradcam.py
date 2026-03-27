from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from seg_moe.data.indexing import infer_image_channels, infer_num_classes
from seg_moe.data.transforms import normalize_image
from seg_moe.models.factory_2d import build_expert, expert_name, list_experts
from seg_moe.utils.checkpoint import (
    extract_model_state_dict,
    load_trusted_torch_checkpoint,
    normalize_state_dict_keys,
)
from seg_moe.utils.config import load_config, resolve_run_dir
from seg_moe.utils.io import ensure_dir


def _read_nii(path: Path) -> np.ndarray:
    import SimpleITK as sitk

    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)  # [Z, Y, X]
    if arr.ndim == 2:
        arr = arr[None]
    return arr


def _dice_binary(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> float:
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    tp = float(np.logical_and(pred_b, gt_b).sum())
    fp = float(np.logical_and(pred_b, ~gt_b).sum())
    fn = float(np.logical_and(~pred_b, gt_b).sum())
    return (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)


def _dice_mean_fg(pred_mask: np.ndarray, gt_mask: np.ndarray, num_classes: int) -> float:
    vals: List[float] = []
    for cid in range(1, num_classes):
        p = pred_mask == cid
        g = gt_mask == cid
        if not np.any(p) and not np.any(g):
            continue
        vals.append(_dice_binary(p, g))
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def _dice_per_class(pred_mask: np.ndarray, gt_mask: np.ndarray, num_classes: int) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for cid in range(num_classes):
        p = pred_mask == cid
        g = gt_mask == cid
        out[f"dice_c{cid}_selected_slice"] = float(_dice_binary(p, g))
    return out


def _resolve_labels_ts_dir(args_labels_ts_dir: str | None, dataset_cfg: dict) -> Optional[Path]:
    if args_labels_ts_dir:
        p = Path(args_labels_ts_dir)
        return p if p.exists() else None

    raw_dir = Path(str(dataset_cfg.get("paths", {}).get("raw_dir", "")))
    labels_ts_rel = str(dataset_cfg.get("raw_structure", {}).get("labels_ts_dir", "labelsTs"))
    cand = raw_dir / labels_ts_rel
    if cand.exists():
        return cand
    return None


def _read_label_slice(
    labels_ts_dir: Path,
    case_id: str,
    sel_z: int,
    out_hw: Tuple[int, int],
    label_map: Dict[int, int],
) -> Optional[np.ndarray]:
    p = labels_ts_dir / f"{case_id}.nii.gz"
    if not p.exists():
        return None
    lab = _read_nii(p).astype(np.int64)  # [Z,H,W]
    if sel_z < 0 or sel_z >= lab.shape[0]:
        return None
    m = lab[sel_z]
    if label_map:
        mapped = m.copy()
        for k, v in label_map.items():
            mapped[m == int(k)] = int(v)
        m = mapped
    m = _resize2d(m.astype(np.uint8), out_hw, is_mask=True).astype(np.int64)
    return m


def _percentile_normalize_uint8(vol: np.ndarray, plow: float = 0.5, phigh: float = 99.5) -> np.ndarray:
    """Match training data preparation in scripts/data/prepare_prostate.py."""
    v = vol.astype(np.float32)
    fg = v[v > 0]
    if fg.size > 100:
        lo = float(np.percentile(fg, plow))
        hi = float(np.percentile(fg, phigh))
    else:
        lo = float(np.percentile(v, plow))
        hi = float(np.percentile(v, phigh))

    if hi - lo < 1e-6:
        hi = lo + 1.0

    v = np.clip(v, lo, hi)
    v = (v - lo) / (hi - lo)
    return (v * 255.0).clip(0, 255).astype(np.uint8)


def _resize2d(arr: np.ndarray, size_hw: Tuple[int, int], is_mask: bool) -> np.ndarray:
    h, w = size_hw
    if arr.shape[0] == h and arr.shape[1] == w:
        return arr
    pil = Image.fromarray(arr)
    pil = pil.resize((w, h), resample=Image.NEAREST if is_mask else Image.BILINEAR)
    return np.array(pil)


def _parse_device(gpus: str | None) -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    if gpus:
        gpu_id = int(gpus.split(",")[0])
    else:
        gpu_id = 0
    return torch.device(f"cuda:{gpu_id}")


def _discover_case_ids(images_ts_dir: Path) -> List[str]:
    ids: set[str] = set()
    for p in images_ts_dir.glob("*_0000.nii.gz"):
        ids.add(p.name.replace("_0000.nii.gz", ""))
    return sorted(ids)


def _build_modal_stack(case_id: str, images_ts_dir: Path, out_hw: Tuple[int, int]) -> np.ndarray:
    """Return [Z,H,W,3], where channel dim is modality m0/m1/m2 after train-consistent prep."""
    mods = []
    for m in range(3):
        path = images_ts_dir / f"{case_id}_{m:04d}.nii.gz"
        if not path.exists():
            raise FileNotFoundError(f"Missing modality file: {path}")
        vol = _read_nii(path)
        mods.append(_percentile_normalize_uint8(vol, 0.5, 99.5))

    shapes = [x.shape for x in mods]
    if len(set(shapes)) != 1:
        raise ValueError(f"Modality shape mismatch for {case_id}: {shapes}")

    z = mods[0].shape[0]
    out = []
    for zi in range(z):
        c0 = _resize2d(mods[0][zi], out_hw, is_mask=False)
        c1 = _resize2d(mods[1][zi], out_hw, is_mask=False)
        c2 = _resize2d(mods[2][zi], out_hw, is_mask=False)
        out.append(np.stack([c0, c1, c2], axis=-1))
    return np.stack(out, axis=0)


def _palette() -> list[tuple[int, int, int]]:
    return [
        (0, 0, 0),
        (255, 0, 0),
        (0, 255, 0),
        (0, 128, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (255, 128, 0),
    ]


def _mask_to_rgb(mask: np.ndarray, num_classes: int) -> np.ndarray:
    pal = _palette()
    out = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for cid in range(num_classes):
        out[mask == cid] = np.array(pal[cid % len(pal)], dtype=np.uint8)
    return out


def _gray_to_rgb(gray: np.ndarray) -> np.ndarray:
    return np.stack([gray, gray, gray], axis=-1)


def _overlay_mask(image_rgb: np.ndarray, mask: np.ndarray, num_classes: int, alpha: float = 0.35) -> np.ndarray:
    out = image_rgb.astype(np.float32).copy()
    pal = _palette()
    for cid in range(1, num_classes):
        m = mask == cid
        if not np.any(m):
            continue
        color = np.array(pal[cid % len(pal)], dtype=np.float32)
        out[m] = (1.0 - alpha) * out[m] + alpha * color
    return np.clip(out, 0, 255).astype(np.uint8)


def _jet_colormap(v01: np.ndarray) -> np.ndarray:
    x = np.clip(v01, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255.0).astype(np.uint8)


def _overlay_cam(image_rgb: np.ndarray, cam01: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    base = image_rgb.astype(np.float32)
    heat = _jet_colormap(cam01).astype(np.float32)
    mix = (1.0 - alpha) * base + alpha * heat
    return np.clip(mix, 0, 255).astype(np.uint8)


def _find_last_conv2d(model: nn.Module) -> nn.Conv2d:
    last: Optional[nn.Conv2d] = None
    for _, mod in model.named_modules():
        if isinstance(mod, nn.Conv2d):
            last = mod
    if last is None:
        raise RuntimeError("No Conv2d layer found for Grad-CAM")
    return last


def _resolve_module_by_name(model: nn.Module, path: str) -> nn.Module:
    mod: nn.Module = model
    for part in path.split("."):
        if hasattr(mod, part):
            mod = getattr(mod, part)
            continue
        if part.isdigit() and isinstance(mod, (nn.Sequential, nn.ModuleList)):
            mod = mod[int(part)]
            continue
        raise KeyError(f"Module path not found: {path}")
    return mod


def _select_gradcam_layer(model: nn.Module, target_layer: str) -> tuple[str, nn.Conv2d]:
    if target_layer and target_layer.lower() != "auto":
        mod = _resolve_module_by_name(model, target_layer)
        if not isinstance(mod, nn.Conv2d):
            raise TypeError(f"Target layer '{target_layer}' is not nn.Conv2d")
        return target_layer, mod

    candidates: list[tuple[str, nn.Conv2d]] = []
    spatial_candidates: list[tuple[str, nn.Conv2d]] = []
    clean_spatial_candidates: list[tuple[str, nn.Conv2d]] = []
    skip_keywords = ("seg", "out", "final", "classifier", "head", "logit")

    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Conv2d):
            continue
        candidates.append((name, mod))
        if any(int(k) > 1 for k in mod.kernel_size):
            spatial_candidates.append((name, mod))
            if not any(tok in name.lower() for tok in skip_keywords):
                clean_spatial_candidates.append((name, mod))

    if clean_spatial_candidates:
        return clean_spatial_candidates[-1]
    if spatial_candidates:
        return spatial_candidates[-1]
    if candidates:
        return candidates[-1]
    raise RuntimeError("No Conv2d layer found for Grad-CAM")


def _resolve_target_class(
    pred: np.ndarray,
    logits: torch.Tensor,
    target_class_arg: str,
    num_classes: int,
) -> int:
    raw = str(target_class_arg).strip().lower()
    if raw != "auto":
        target_class = int(raw)
        if target_class < 0 or target_class >= num_classes:
            raise ValueError(f"target_class must be in [0, {num_classes - 1}], got {target_class}")
        return target_class

    if num_classes <= 1:
        return 0

    fg_ids = list(range(1, num_classes))
    fg_counts = np.asarray([(pred == cid).sum() for cid in fg_ids], dtype=np.int64)
    if fg_counts.size > 0 and int(fg_counts.max()) > 0:
        return int(fg_ids[int(fg_counts.argmax())])

    fg_logits = logits[:, 1:, :, :].mean(dim=(0, 2, 3))
    return int(torch.argmax(fg_logits).item() + 1)


def _select_slice_for_class(pred: np.ndarray, logits: torch.Tensor, target_class: int) -> int:
    z = pred.shape[0]
    target_pixels = (pred == target_class).reshape(z, -1).sum(axis=1)
    if int(target_pixels.max()) > 0:
        return int(target_pixels.argmax())
    target_scores = logits[:, target_class, :, :].mean(dim=(1, 2)).detach().cpu().numpy()
    return int(target_scores.argmax())


def _compute_gradcam(
    model: nn.Module,
    x: torch.Tensor,
    target_class: int,
    target_layer: str = "auto",
    topk_ratio: float = 0.2,
) -> tuple[np.ndarray, str]:
    acts: Dict[str, torch.Tensor] = {}
    grads: Dict[str, torch.Tensor] = {}

    layer_name, target_layer_mod = _select_gradcam_layer(model, target_layer)

    def fwd_hook(_m: nn.Module, _inp: tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
        acts["v"] = out

    def bwd_hook(_m: nn.Module, _gin: tuple[torch.Tensor, ...], gout: tuple[torch.Tensor, ...]) -> None:
        grads["v"] = gout[0]

    h1 = target_layer_mod.register_forward_hook(fwd_hook)
    h2 = target_layer_mod.register_full_backward_hook(bwd_hook)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(x)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]

        if target_class < 0 or target_class >= int(logits.shape[1]):
            raise ValueError(f"target_class {target_class} out of range for logits with {logits.shape[1]} channels")

        score_map = logits[:, target_class, :, :]
        pred_mask = (logits.argmax(dim=1) == target_class).float()
        if float(pred_mask.sum().item()) > 0.0:
            score = (score_map * pred_mask).sum() / pred_mask.sum().clamp_min(1.0)
        else:
            flat = score_map.flatten(start_dim=1)
            k = max(1, int(round(float(flat.shape[1]) * float(topk_ratio))))
            score = flat.topk(k, dim=1).values.mean()

        score.backward(retain_graph=False)

        a = acts["v"]
        g = grads["v"]
        weights = g.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * a).sum(dim=1))
        cam = F.interpolate(cam[:, None], size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]

        cam_np = cam.detach().cpu().numpy()[0]
        cam_np = cam_np - cam_np.min()
        den = cam_np.max()
        if den > 1e-8:
            cam_np = cam_np / den
        else:
            cam_np = np.zeros_like(cam_np, dtype=np.float32)
        return cam_np.astype(np.float32), layer_name
    finally:
        h1.remove()
        h2.remove()


def main() -> None:
    ap = argparse.ArgumentParser(description="Layer2 inference visualization with train-inference consistency")
    ap.add_argument("--exp", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--images-ts-dir", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--which", choices=["best", "last"], default="best")
    ap.add_argument("--cases", default="", help="Comma-separated case IDs")
    ap.add_argument("--num-cases", type=int, default=3)
    ap.add_argument("--gpus", type=str, default=None)
    ap.add_argument("--labels-ts-dir", type=str, default=None,
                    help="Optional labelsTs directory for metric computation")
    ap.add_argument("--no-uncertainty", action="store_true")
    ap.add_argument("--target-class", default="auto",
                    help="Class id for Grad-CAM. Use 'auto' to pick the dominant predicted foreground class.")
    ap.add_argument("--target-layer", default="auto",
                    help="Conv layer name for Grad-CAM. Default auto picks the last spatial Conv2d.")
    ap.add_argument("--cam-topk-ratio", type=float, default=0.2,
                    help="Top-k ratio fallback when the target class is absent in the predicted mask.")
    ap.add_argument("--alpha", type=float, default=0.35)
    ap.add_argument("--cam-alpha", type=float, default=0.45)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    exp_cfg = load_config(args.exp)
    dataset_cfg = load_config(exp_cfg["dataset"]["config"])
    models_cfg = load_config(args.models)
    run_dir = Path(resolve_run_dir(exp_cfg))
    device = _parse_device(args.gpus)

    num_classes = infer_num_classes(dataset_cfg)
    base_in = infer_image_channels(dataset_cfg)
    image_size = tuple(int(v) for v in dataset_cfg["input"].get("image_size", [256, 256]))
    normalize_cfg = dict(dataset_cfg["input"].get("normalize", {}) or {})

    expert_cfgs = list_experts(models_cfg)
    num_experts = len(expert_cfgs)

    add_uncertainty = not args.no_uncertainty
    extra_uncertainty_ch = (1 + num_classes) if add_uncertainty else 0
    l2_in_channels = base_in + num_experts * num_classes + extra_uncertainty_ch

    images_ts_dir = Path(args.images_ts_dir)
    if not images_ts_dir.exists():
        raise FileNotFoundError(f"imagesTs dir not found: {images_ts_dir}")

    discovered = _discover_case_ids(images_ts_dir)
    if not discovered:
        raise RuntimeError(f"No *_0000.nii.gz found in {images_ts_dir}")

    if args.cases.strip():
        case_ids = [c.strip() for c in args.cases.split(",") if c.strip()]
    else:
        case_ids = discovered[: int(args.num_cases)]

    label_map = {int(k): int(v) for k, v in (dataset_cfg.get("task", {}).get("label_map") or {}).items()}
    labels_ts_dir = _resolve_labels_ts_dir(args.labels_ts_dir, dataset_cfg)

    out_root = Path(args.out_dir) if args.out_dir else (run_dir / "results" / "layer2" / "imagesTs_cases_vis_gradcam")
    ensure_dir(out_root)

    print(
        f"[L2 consistent infer] fold={args.fold} device={device} "
        f"cases={case_ids} add_uncertainty={add_uncertainty} labels_ts={labels_ts_dir}"
    )

    # Load all expert checkpoints once.
    layer1_models: Dict[str, nn.Module] = {}
    layer2_models: Dict[str, nn.Module] = {}

    for ec in expert_cfgs:
        name = expert_name(ec)

        l1_ckpt = run_dir / "checkpoints" / "layer1" / f"fold{int(args.fold)}" / name / f"{args.which}.pt"
        if not l1_ckpt.exists():
            raise FileNotFoundError(f"Missing layer1 checkpoint: {l1_ckpt}")

        l1 = build_expert(ec, in_channels=base_in, num_classes=num_classes)
        s1 = load_trusted_torch_checkpoint(l1_ckpt, map_location="cpu")
        l1.load_state_dict(normalize_state_dict_keys(extract_model_state_dict(s1)), strict=True)
        l1.to(device).eval()
        layer1_models[name] = l1

        l2_ckpt = run_dir / "checkpoints" / "layer2" / f"fold{int(args.fold)}" / name / f"{args.which}.pt"
        if not l2_ckpt.exists():
            raise FileNotFoundError(f"Missing layer2 checkpoint: {l2_ckpt}")

        l2 = build_expert(ec, in_channels=l2_in_channels, num_classes=num_classes)
        s2 = load_trusted_torch_checkpoint(l2_ckpt, map_location="cpu")
        l2.load_state_dict(normalize_state_dict_keys(extract_model_state_dict(s2)), strict=True)
        l2.to(device).eval()
        layer2_models[name] = l2

    summary_rows: List[Dict[str, Any]] = []

    for case_id in case_ids:
        modal_stack = _build_modal_stack(case_id, images_ts_dir, image_size)  # [Z,H,W,3]
        z, h, w, _ = modal_stack.shape

        # Training-consistent image normalization path.
        x_img_np = []
        for zi in range(z):
            img_n = normalize_image(modal_stack[zi], normalize_cfg)
            x_img_np.append(np.transpose(img_n.astype(np.float32), (2, 0, 1)))
        x_img_t = torch.from_numpy(np.stack(x_img_np, axis=0)).to(device)  # [Z,3,H,W]

        # Layer1 probabilities used as Layer2 input channels.
        probs_by_expert = []
        for ec in expert_cfgs:
            name = expert_name(ec)
            with torch.no_grad():
                logits = layer1_models[name](x_img_t)
                if isinstance(logits, (list, tuple)):
                    logits = logits[0]
                probs_by_expert.append(torch.softmax(logits, dim=1))  # [Z,M,H,W]

        probs_k = torch.stack(probs_by_expert, dim=0)  # [K,Z,M,H,W]
        probs_flat = probs_k.permute(1, 0, 2, 3, 4).reshape(z, num_experts * num_classes, h, w)

        parts = [x_img_t, probs_flat]
        if add_uncertainty:
            eps = 1e-8
            mean_probs = probs_k.mean(dim=0)  # [Z,M,H,W]
            entropy = -(mean_probs * torch.log(mean_probs + eps)).sum(dim=1, keepdim=True)
            entropy = entropy / (np.log(num_classes) + eps)
            disagreement = probs_k.std(dim=0)  # [Z,M,H,W]
            parts.extend([entropy, disagreement])

        x_l2 = torch.cat(parts, dim=1)

        for ec in expert_cfgs:
            model_name = expert_name(ec)
            model = layer2_models[model_name]

            with torch.no_grad():
                logits2 = model(x_l2)
                if isinstance(logits2, (list, tuple)):
                    logits2 = logits2[0]
                pred = logits2.argmax(dim=1).detach().cpu().numpy().astype(np.uint8)  # [Z,H,W]

            target_class = _resolve_target_class(pred, logits2, args.target_class, num_classes)
            sel_z = _select_slice_for_class(pred, logits2, target_class)

            # Class-specific Grad-CAM for selected slice.
            x_sel = x_l2[sel_z:sel_z + 1].clone().detach().requires_grad_(True)
            cam, gradcam_layer = _compute_gradcam(
                model,
                x_sel,
                target_class=target_class,
                target_layer=args.target_layer,
                topk_ratio=float(args.cam_topk_ratio),
            )

            pred_mask = pred[sel_z]
            rgb_view = modal_stack[sel_z]  # modality-as-RGB view for compatibility

            out_dir = ensure_dir(out_root / model_name / case_id)

            # Legacy composite outputs.
            raw_path = out_dir / f"z{sel_z:03d}.raw.png"
            seg_path = out_dir / f"z{sel_z:03d}.seg.png"
            overlay_path = out_dir / f"z{sel_z:03d}.overlay.png"
            gradcam_path = out_dir / f"z{sel_z:03d}.gradcam.png"

            Image.fromarray(rgb_view, mode="RGB").save(raw_path)
            Image.fromarray(_mask_to_rgb(pred_mask, num_classes), mode="RGB").save(seg_path)
            Image.fromarray(_overlay_mask(rgb_view, pred_mask, num_classes, alpha=float(args.alpha)), mode="RGB").save(overlay_path)
            Image.fromarray(_overlay_cam(rgb_view, cam, alpha=float(args.cam_alpha)), mode="RGB").save(gradcam_path)

            # Per-modality raw / overlay / gradcam outputs requested by user.
            mod_paths: Dict[str, str] = {}
            for mid in range(3):
                gray = modal_stack[sel_z, :, :, mid]
                gray_rgb = _gray_to_rgb(gray)

                m_raw = out_dir / f"z{sel_z:03d}.m{mid}.raw.png"
                m_overlay = out_dir / f"z{sel_z:03d}.m{mid}.overlay.png"
                m_cam = out_dir / f"z{sel_z:03d}.m{mid}.gradcam.png"

                Image.fromarray(gray, mode="L").save(m_raw)
                Image.fromarray(_overlay_mask(gray_rgb, pred_mask, num_classes, alpha=float(args.alpha)), mode="RGB").save(m_overlay)
                Image.fromarray(_overlay_cam(gray_rgb, cam, alpha=float(args.cam_alpha)), mode="RGB").save(m_cam)

                mod_paths[f"m{mid}_raw_path"] = str(m_raw)
                mod_paths[f"m{mid}_overlay_path"] = str(m_overlay)
                mod_paths[f"m{mid}_gradcam_path"] = str(m_cam)

            # Per-class binary masks.
            per_class_paths: Dict[str, str] = {}
            for cid in range(num_classes):
                ch = (pred_mask == cid).astype(np.uint8) * 255
                ch_path = out_dir / f"z{sel_z:03d}.seg_c{cid}.png"
                Image.fromarray(ch, mode="L").save(ch_path)
                per_class_paths[f"seg_c{cid}_path"] = str(ch_path)

            # Added output metric: mean foreground Dice on selected slice (if labels available).
            dice_mean_selected_slice = float("nan")
            per_label_dice: Dict[str, float] = {
                f"dice_c{cid}_selected_slice": float("nan") for cid in range(num_classes)
            }
            if labels_ts_dir is not None:
                gt_mask = _read_label_slice(
                    labels_ts_dir=labels_ts_dir,
                    case_id=case_id,
                    sel_z=sel_z,
                    out_hw=(h, w),
                    label_map=label_map,
                )
                if gt_mask is not None:
                    dice_mean_selected_slice = _dice_mean_fg(pred_mask, gt_mask, num_classes)
                    per_label_dice = _dice_per_class(pred_mask, gt_mask, num_classes)

            summary_rows.append(
                {
                    "model": model_name,
                    "case_id": case_id,
                    "selected_slice": sel_z,
                    "target_class": target_class,
                    "gradcam_layer": gradcam_layer,
                    "dice_mean_selected_slice": dice_mean_selected_slice,
                    **per_label_dice,
                    "raw_path": str(raw_path),
                    "seg_path": str(seg_path),
                    "overlay_path": str(overlay_path),
                    "gradcam_path": str(gradcam_path),
                    **mod_paths,
                    **per_class_paths,
                }
            )

            print(f"  {model_name} | {case_id} | z={sel_z} | c={target_class} | layer={gradcam_layer}")

        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary_csv = out_root / "imagesTs_layer2_vis_summary.csv"
    if summary_rows:
        keys = list(summary_rows[0].keys())
        with open(summary_csv, "w", encoding="utf-8", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=keys)
            wr.writeheader()
            wr.writerows(summary_rows)

    print(f"[L2 consistent infer] outputs: {out_root}")
    print(f"[L2 consistent infer] summary: {summary_csv}")


if __name__ == "__main__":
    main()

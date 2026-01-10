from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from seg_moe.combiners.decision_template import DecisionTemplateCombiner
from seg_moe.combiners.ole import OLECombiner
from seg_moe.combiners.we_clpso import WECLPSOCombiner
from seg_moe.data.dataset_2d import SegmentationDataset2D
from seg_moe.data.indexing import infer_image_channels, infer_num_classes
from seg_moe.evaluation.metrics_2d import compute_segmentation_metrics_batch
from seg_moe.models.factory_2d import build_smp_model, expert_name, list_experts
from seg_moe.utils.config import load_config, resolve_run_dir
from seg_moe.utils.io import ensure_dir, load_jsonl, save_json
from seg_moe.data.layer2_dataset import Layer2Dataset


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


def _ckpt(run_dir: Path, layer: str, fold: int, ex: str, which: str = "best") -> Path:
    return run_dir / "checkpoints" / layer / f"fold{fold}" / ex / f"{which}.pt"


def _ckpt_unetpp(run_dir: Path, fold: int, backbone: str, which: str = "best") -> Path:
    return run_dir / "checkpoints" / "unetpp" / f"fold{fold}" / f"unetpp-{backbone}" / f"{which}.pt"


def _eval_single(model: torch.nn.Module, dl: DataLoader, num_classes: int, device: torch.device) -> dict:
    model.eval()
    metrics = []
    with torch.no_grad():
        for x, y, meta in tqdm(dl, desc="eval"):
            logits = model(x.to(device))
            probs = torch.softmax(logits, dim=1)
            spacing = meta.get("spacing_yx")
            spacing_yx = None
            if spacing is not None:
                # dataloader collates to list; take first
                s0 = spacing[0]
                if isinstance(s0, (list, tuple)) and len(s0) == 2:
                    spacing_yx = (float(s0[0]), float(s0[1]))
            metrics.append(compute_segmentation_metrics_batch(probs, y.to(device), num_classes=num_classes, spacing_yx=spacing_yx))
    # mean
    out = {k: float(np.mean([m.get(k, 0.0) for m in metrics])) for k in metrics[0].keys()} if metrics else {}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--training", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--which", choices=["best", "last"], default="best")
    args = ap.parse_args()

    exp_cfg = load_config(args.exp)
    dataset_cfg = load_config(exp_cfg["dataset"]["config"])
    models_cfg = load_config(args.models)
    run_dir = resolve_run_dir(exp_cfg)
    results_dir = ensure_dir(Path(exp_cfg["output"]["results_dir"].replace("${exp_name}", exp_cfg["exp_name"])))

    rows = _load_splits(dataset_cfg)
    fold = int(args.fold)

    # evaluate on test if exists else on val_fold
    split_candidates = ["test", f"val_fold{fold}"]
    split = next((s for s in split_candidates if any(r.get("split") == s for r in rows)), f"val_fold{fold}")
    eval_rows = [r for r in rows if r.get("split") == split]

    num_classes = infer_num_classes(dataset_cfg)
    in_channels = infer_image_channels(dataset_cfg)

    ds = SegmentationDataset2D(eval_rows, dataset_cfg, augs_cfg=None, is_train=False)
    dl = DataLoader(ds, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    experts = list_experts(models_cfg)
    encoder_weights = models_cfg.get("smp", {}).get("encoder_weights", "imagenet")

    records = []

    # single experts (layer1)
    for arch, backbone in experts:
        ex = expert_name(arch, backbone)
        ckpt = _ckpt(run_dir, "layer1", fold, ex, which=args.which)
        if not ckpt.exists():
            continue
        model = build_smp_model(arch, backbone, in_channels=in_channels, classes=num_classes, encoder_weights=encoder_weights)
        state = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(state["model"], strict=True)
        model.to(device)
        m = _eval_single(model, dl, num_classes=num_classes, device=device)
        records.append({"dataset": dataset_cfg["name"], "split": split, "method": f"single_{ex}", **m})

    # UNet++ baseline (single strong model)
    unetpp_cfg = models_cfg.get("unetpp", {})
    if unetpp_cfg.get("enabled", True):
        backbone = str(unetpp_cfg.get("backbone", "resnet34")).lower()
        ckpt = _ckpt_unetpp(run_dir, fold, backbone, which=args.which)
        if ckpt.exists():
            model = build_smp_model(
                "unetplusplus",
                backbone,
                in_channels=in_channels,
                classes=num_classes,
                encoder_weights=encoder_weights,
            )
            state = torch.load(ckpt, map_location="cpu")
            model.load_state_dict(state["model"], strict=True)
            model.to(device)
            m = _eval_single(model, dl, num_classes=num_classes, device=device)
            records.append({"dataset": dataset_cfg["name"], "split": split, "method": "unetpp", **m})

    # combiner methods require cached probs. If not available, skip.
    cache_root = Path(exp_cfg["layering"]["cache_root"].replace("${exp_name}", exp_cfg["exp_name"]))
    l1_cache = cache_root / "layer1_probs" / dataset_cfg["name"] / split

    weights_out = {
        "dataset": dataset_cfg["name"],
        "fold": fold,
        "experts": [expert_name(a, b) for a, b in experts],
        "num_classes": num_classes,
        "layer1": {},
        "layer2": {},
    }

    if l1_cache.exists():
        # build flattened training data from val split for fitting combiners
        fit_split = f"val_fold{fold}"
        fit_rows = [r for r in rows if r.get("split") == fit_split]
        fit_ds = SegmentationDataset2D(fit_rows, dataset_cfg, augs_cfg=None, is_train=False)
        fit_dl = DataLoader(fit_ds, batch_size=1, shuffle=False)

        X_list = []
        y_list = []
        for _, mask, meta in tqdm(fit_dl, desc="collect combiner fit"):
            sid = meta["id"][0]
            npz = np.load(cache_root / "layer1_probs" / dataset_cfg["name"] / fit_split / f"{sid}.npz")
            probs = npz["probs"].astype(np.float32)  # [K,M,H,W]
            H, W = probs.shape[-2], probs.shape[-1]
            probs_flat = probs.transpose(2, 3, 0, 1).reshape(H * W, probs.shape[0], probs.shape[1])
            X_list.append(probs_flat)
            y_list.append(mask.numpy().reshape(-1))
        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)

        # OLE-9
        ole = OLECombiner(mode="lsq_bounded")
        ole.fit(X, y, num_classes=num_classes)
        # DT-9
        dt = DecisionTemplateCombiner()
        dt.fit(X, y, num_classes=num_classes)
        # WE-CLPSO
        we = WECLPSOCombiner(n_particles=10, iters=30)
        we.fit(X, y, num_classes=num_classes)

        # evaluate on split using cached probs
        def eval_cached_combiner(name: str, pred_fn):
            metrics = []
            for _, mask, meta in tqdm(dl, desc=f"eval {name}"):
                sid = meta["id"][0]
                npz = np.load(l1_cache / f"{sid}.npz")
                probs = npz["probs"].astype(np.float32)  # [K,M,H,W]
                H, W = probs.shape[-2], probs.shape[-1]
                probs_flat = probs.transpose(2, 3, 0, 1).reshape(H * W, probs.shape[0], probs.shape[1])
                out = pred_fn(probs_flat)
                if out.ndim == 1:
                    pred = out.reshape(H, W)
                else:
                    pred = np.argmax(out, axis=-1).reshape(H, W)
                # convert to torch probs (one-hot) for metric wrapper
                pred_1h = np.eye(num_classes, dtype=np.float32)[pred].transpose(2, 0, 1)[None]
                metrics.append(compute_segmentation_metrics_batch(torch.from_numpy(pred_1h), mask, num_classes=num_classes))
            return {k: float(np.mean([m.get(k, 0.0) for m in metrics])) for k in metrics[0].keys()} if metrics else {}

        m_ole = eval_cached_combiner("ole9", lambda pf: ole.predict(pf))
        records.append({"dataset": dataset_cfg["name"], "split": split, "method": "ole9", **m_ole})

        m_dt = eval_cached_combiner("dt9", lambda pf: dt.predict(pf))
        records.append({"dataset": dataset_cfg["name"], "split": split, "method": "dt9", **m_dt})

        m_we = eval_cached_combiner("we_clpso", lambda pf: we.predict(pf))
        records.append({"dataset": dataset_cfg["name"], "split": split, "method": "we_clpso", **m_we})

        # save combiner weights (layer1)
        weights_out["layer1"]["ole9"] = ole.weights.w.tolist() if ole.weights else None
        weights_out["layer1"]["we_clpso"] = we.state.w.tolist() if we.state else None

    # two-layer: evaluate layer2 experts and final fusion based on layer2 cached probs
    l2_cache = cache_root / "layer2_probs" / dataset_cfg["name"] / split
    if l2_cache.exists():
        # evaluate layer2 single experts (need I* input, built from layer1 probs)
        layer1_for_eval = cache_root / "layer1_probs" / dataset_cfg["name"] / split
        if layer1_for_eval.exists():
            base_ds = SegmentationDataset2D(eval_rows, dataset_cfg, augs_cfg=None, is_train=False)
            base_in = in_channels
            K = len(experts)
            in_ch_l2 = base_in + K * num_classes
            ds_l2 = Layer2Dataset(base_ds, layer1_for_eval, num_experts=K, num_classes=num_classes)
            dl_l2 = DataLoader(ds_l2, batch_size=1, shuffle=False)

            for arch, backbone in experts:
                ex = expert_name(arch, backbone)
                ckpt = _ckpt(run_dir, "layer2", fold, ex, which=args.which)
                if not ckpt.exists():
                    continue
                model = build_smp_model(arch, backbone, in_channels=in_ch_l2, classes=num_classes, encoder_weights=encoder_weights)
                state = torch.load(ckpt, map_location="cpu")
                model.load_state_dict(state["model"], strict=True)
                model.to(device)
                m = _eval_single(model, dl_l2, num_classes=num_classes, device=device)
                records.append({"dataset": dataset_cfg["name"], "split": split, "method": f"layer2_single_{ex}", **m})

        # fit combiners on layer2 cached probs (val split), evaluate on split
        fit_split = f"val_fold{fold}"
        fit_rows = [r for r in rows if r.get("split") == fit_split]
        fit_cache = cache_root / "layer2_probs" / dataset_cfg["name"] / fit_split
        if fit_cache.exists() and fit_rows:
            fit_ds = SegmentationDataset2D(fit_rows, dataset_cfg, augs_cfg=None, is_train=False)
            fit_dl = DataLoader(fit_ds, batch_size=1, shuffle=False)

            X_list = []
            y_list = []
            for _, mask, meta in tqdm(fit_dl, desc="collect layer2 combiner fit"):
                sid = meta["id"][0]
                npz = np.load(fit_cache / f"{sid}.npz")
                probs = npz["probs"].astype(np.float32)  # [K,M,H,W]
                H, W = probs.shape[-2], probs.shape[-1]
                probs_flat = probs.transpose(2, 3, 0, 1).reshape(H * W, probs.shape[0], probs.shape[1])
                X_list.append(probs_flat)
                y_list.append(mask.numpy().reshape(-1))
            X2 = np.concatenate(X_list, axis=0)
            y2 = np.concatenate(y_list, axis=0)

            ole2 = OLECombiner(mode="lsq_bounded")
            ole2.fit(X2, y2, num_classes=num_classes)
            dt2 = DecisionTemplateCombiner()
            dt2.fit(X2, y2, num_classes=num_classes)
            we2 = WECLPSOCombiner(n_particles=10, iters=30)
            we2.fit(X2, y2, num_classes=num_classes)

            def eval_cached_layer2(name: str, pred_fn):
                metrics = []
                for _, mask, meta in tqdm(dl, desc=f"eval {name}"):
                    sid = meta["id"][0]
                    npz = np.load(l2_cache / f"{sid}.npz")
                    probs = npz["probs"].astype(np.float32)
                    H, W = probs.shape[-2], probs.shape[-1]
                    pf = probs.transpose(2, 3, 0, 1).reshape(H * W, probs.shape[0], probs.shape[1])
                    out = pred_fn(pf)
                    if out.ndim == 1:
                        pred = out.reshape(H, W)
                    else:
                        pred = np.argmax(out, axis=-1).reshape(H, W)
                    pred_1h = np.eye(num_classes, dtype=np.float32)[pred].transpose(2, 0, 1)[None]
                    metrics.append(compute_segmentation_metrics_batch(torch.from_numpy(pred_1h), mask, num_classes=num_classes))
                return {k: float(np.mean([m.get(k, 0.0) for m in metrics])) for k in metrics[0].keys()} if metrics else {}

            m_p = eval_cached_layer2("proposed_2layer_ole9", lambda pf: ole2.predict(pf))
            records.append({"dataset": dataset_cfg["name"], "split": split, "method": "proposed_2layer_ole9", **m_p})

            m_p = eval_cached_layer2("proposed_2layer_dt9", lambda pf: dt2.predict(pf))
            records.append({"dataset": dataset_cfg["name"], "split": split, "method": "proposed_2layer_dt9", **m_p})

            m_p = eval_cached_layer2("proposed_2layer_we_clpso", lambda pf: we2.predict(pf))
            records.append({"dataset": dataset_cfg["name"], "split": split, "method": "proposed_2layer_we_clpso", **m_p})

            weights_out["layer2"]["ole9"] = ole2.weights.w.tolist() if ole2.weights else None
            weights_out["layer2"]["we_clpso"] = we2.state.w.tolist() if we2.state else None

    save_json(results_dir / f"{dataset_cfg['name']}_fold{fold}_weights.json", weights_out)

    df = pd.DataFrame.from_records(records)
    out_csv = results_dir / f"metrics_{dataset_cfg['name']}_fold{fold}_{split}.csv"
    df.to_csv(out_csv, index=False)
    print(f"Wrote: {out_csv}")


if __name__ == "__main__":
    main()

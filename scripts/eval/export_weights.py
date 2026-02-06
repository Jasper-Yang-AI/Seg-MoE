"""
Export combiner weights (OLE / WE-CLPSO) as readable JSON.

Usage:
    python scripts/eval/export_weights.py \
        --exp configs/2d/exp/exp_msd_task03_liver.yaml \
        --models configs/2d/models.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from seg_moe.models.factory_2d import expert_name, list_experts
from seg_moe.utils.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--folds", type=int, nargs="+", default=[0])
    args = ap.parse_args()

    exp_cfg = load_config(args.exp)
    models_cfg = load_config(args.models)
    dataset_name = exp_cfg["dataset"]["name"]
    results_dir = Path(
        exp_cfg["output"]["results_dir"].replace("${exp_name}", exp_cfg["exp_name"])
    )

    expert_cfgs = list_experts(models_cfg)
    names = [expert_name(ec) for ec in expert_cfgs]

    collected = {}
    for fold in args.folds:
        w_path = results_dir / f"{dataset_name}_fold{fold}_weights.json"
        if not w_path.exists():
            print(f"  skip {w_path} (not found)")
            continue
        with open(w_path, "r") as f:
            data = json.load(f)
        collected[f"fold{fold}"] = data

    if not collected:
        print("No weight files found. Run eval_methods.py first.")
        return

    tables_dir = Path(
        exp_cfg.get("output", {})
        .get("tables_dir", f"runs/{exp_cfg['exp_name']}/tables")
        .replace("${exp_name}", exp_cfg["exp_name"])
    )
    tables_dir.mkdir(parents=True, exist_ok=True)

    out = tables_dir / "expert_weights.json"
    summary = {"experts": names, "folds": collected}
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {out}")

    # Print readable summary
    for fold_key, data in collected.items():
        print(f"\n--- {fold_key} ---")
        layer1 = data.get("layer1", {})
        for method, weights in layer1.items():
            if weights is not None:
                w_str = ", ".join(f"{n}={w:.4f}" for n, w in zip(names, weights))
                print(f"  {method}: [{w_str}]")


if __name__ == "__main__":
    main()

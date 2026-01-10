from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from seg_moe.models.factory_2d import expert_name, list_experts
from seg_moe.utils.config import load_config
from seg_moe.utils.io import ensure_dir, load_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--models", default="configs/2d/models.yaml")
    args = ap.parse_args()

    exp_cfg = load_config(args.exp)
    models_cfg = load_config(args.models)
    exp_name = exp_cfg["exp_name"]

    results_dir = Path(exp_cfg["output"]["results_dir"].replace("${exp_name}", exp_name))
    tables_dir = ensure_dir(Path(exp_cfg["output"]["tables_dir"].replace("${exp_name}", exp_name)))

    # weights stored by eval_methods
    dataset_cfg = load_config(exp_cfg["dataset"]["config"])
    wjson = results_dir / f"{dataset_cfg['name']}_fold{int(args.fold)}_weights.json"
    if not wjson.exists():
        raise FileNotFoundError(f"Missing weights json: {wjson}. Run scripts/eval_methods.py first.")

    data = load_json(wjson)
    experts = [expert_name(a, b) for a, b in list_experts(models_cfg)]
    class_names = [f"class_{i}" for i in range(int(dataset_cfg["task"]["num_classes"]))]

    # New structure: {layer1:{...}, layer2:{...}}
    for layer in ["layer1", "layer2"]:
        block = data.get(layer, {}) if isinstance(data, dict) else {}
        if not isinstance(block, dict):
            continue

        if block.get("ole9") is not None:
            w = pd.DataFrame(block["ole9"], index=experts, columns=class_names)
            out = tables_dir / f"table6_weights_{layer}_ole9_{dataset_cfg['name']}.csv"
            w.to_csv(out)
            print(f"Wrote {out}")

        if block.get("we_clpso") is not None:
            w = pd.DataFrame(block["we_clpso"], index=experts, columns=class_names)
            out = tables_dir / f"table6_weights_{layer}_weclpso_{dataset_cfg['name']}.csv"
            w.to_csv(out)
            print(f"Wrote {out}")


if __name__ == "__main__":
    main()

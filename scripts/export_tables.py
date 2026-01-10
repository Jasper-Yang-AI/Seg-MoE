from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from seg_moe.utils.config import load_config
from seg_moe.utils.io import ensure_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    args = ap.parse_args()

    exp_cfg = load_config(args.exp)
    exp_name = exp_cfg["exp_name"]
    results_dir = Path(exp_cfg["output"]["results_dir"].replace("${exp_name}", exp_name))
    tables_dir = ensure_dir(Path(exp_cfg["output"]["tables_dir"].replace("${exp_name}", exp_name)))

    csvs = sorted(results_dir.glob("metrics_*_fold*_*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No metrics CSVs found under {results_dir}")

    df = pd.concat([pd.read_csv(p) for p in csvs], ignore_index=True)

    # Table style: method x dataset for each metric
    for metric in ["dice_mean", "iou_mean", "hd_mean", "mad_mean"]:
        if metric not in df.columns:
            continue
        pivot = df.pivot_table(index="method", columns="dataset", values=metric, aggfunc="mean")
        out = tables_dir / f"table_{metric}.csv"
        pivot.to_csv(out)
        print(f"Wrote {out}")

    # Paper-like grouped tables (coarse):
    # - table2_single_models: 9 single experts
    # - table3_ensemble_methods: OLE/DT/WE-CLPSO (layer1)
    # - table4_proposed_vs_baselines: UNet++ + proposed_2layer_* + best ensemble baselines
    for metric in ["dice_mean", "iou_mean", "hd_mean", "mad_mean"]:
        if metric not in df.columns:
            continue

        def _write(name: str, sub: pd.DataFrame):
            if sub.empty:
                return
            pivot = sub.pivot_table(index="method", columns="dataset", values=metric, aggfunc="mean")
            out = tables_dir / f"{name}_{metric}.csv"
            pivot.to_csv(out)
            print(f"Wrote {out}")

        _write("table2_single_models", df[df["method"].str.startswith("single_")])
        _write("table3_ensemble_methods", df[df["method"].isin(["ole9", "dt9", "we_clpso"])])
        _write(
            "table4_proposed_vs_baselines",
            df[df["method"].isin(["unetpp", "proposed_2layer_ole9", "proposed_2layer_dt9", "proposed_2layer_we_clpso"])],
        )


if __name__ == "__main__":
    main()

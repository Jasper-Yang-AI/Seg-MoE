"""
Export evaluation metrics as summary tables (CSV / LaTeX).

Usage:
    python scripts/eval/export_tables.py \
        --exp configs/2d/exp/exp_msd_task03_liver.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from seg_moe.utils.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--folds", type=int, nargs="+", default=[0])
    ap.add_argument("--latex", action="store_true", help="Also emit LaTeX tables")
    args = ap.parse_args()

    exp_cfg = load_config(args.exp)
    dataset_name = exp_cfg["dataset"]["name"]
    tables_dir = Path(
        exp_cfg.get("output", {})
        .get("tables_dir", f"runs/{exp_cfg['exp_name']}/tables")
        .replace("${exp_name}", exp_cfg["exp_name"])
    )
    results_dir = Path(
        exp_cfg["output"]["results_dir"].replace("${exp_name}", exp_cfg["exp_name"])
    )

    # Collect per-fold CSVs
    frames = []
    for fold in args.folds:
        pattern = f"metrics_{dataset_name}_fold{fold}_*.csv"
        for csv in sorted(results_dir.glob(pattern)):
            df = pd.read_csv(csv)
            df["fold"] = fold
            frames.append(df)

    if not frames:
        print(f"No result CSVs found in {results_dir}")
        return

    all_df = pd.concat(frames, ignore_index=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # ------ Table 1: Single expert results ------
    single = all_df[all_df["method"].str.startswith("single_")]
    if not single.empty:
        out = tables_dir / "table1_single_experts.csv"
        single.to_csv(out, index=False)
        print(f"Wrote {out}")
        if args.latex:
            print(single.to_latex(index=False))

    # ------ Table 2: Ensemble / combiner results ------
    ensemble_methods = ["mean_ensemble", "ole", "dt", "we_clpso"]
    ens = all_df[all_df["method"].isin(ensemble_methods)]
    if not ens.empty:
        out = tables_dir / "table2_ensemble_methods.csv"
        ens.to_csv(out, index=False)
        print(f"Wrote {out}")
        if args.latex:
            print(ens.to_latex(index=False))

    # ------ Table 3: All methods combined ------
    out = tables_dir / "table3_all_methods.csv"
    all_df.to_csv(out, index=False)
    print(f"Wrote {out}")

    # ------ Summary (mean ± std across folds) ------
    if len(args.folds) > 1:
        metric_cols = [c for c in all_df.columns if c not in ("dataset", "split", "method", "fold")]
        summary = all_df.groupby("method")[metric_cols].agg(["mean", "std"])
        out = tables_dir / "table_summary.csv"
        summary.to_csv(out)
        print(f"Wrote {out}")

    print("Done.")


if __name__ == "__main__":
    main()

"""
Export evaluation metrics as summary tables (CSV / LaTeX).

Handles eval_methods.py output with:
  - Layer1/Layer2/Gating methods (L1_*, L2_*, gating)
  - Per-class metrics (dice_c1, dice_c2, ...) and aggregated (dice_mean)
  - Statistical significance results
  - Multi-fold mean ± std

Usage:
    python scripts/eval/export_tables.py \\
        --exp configs/2d/exp/exp_msd_task03_liver.yaml \\
        --folds 0 1 2 3 4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from seg_moe.utils.config import load_config


# ── Metric column ordering (Maier-Hein et al. 2024 recommended) ──
_PRIMARY = ["dice_mean", "hd95_mean", "nsd_mean"]
_SECONDARY = ["iou_mean", "asd_mean", "sens_mean", "prec_mean"]
_ALIASES = {"hd_mean", "mad_mean"}  # backward compat, skip in output


def _order_metric_cols(cols: list[str]) -> list[str]:
    """Order metric columns: primary → secondary → per-class → rest."""
    ordered = []
    for c in _PRIMARY:
        if c in cols:
            ordered.append(c)
    for c in _SECONDARY:
        if c in cols:
            ordered.append(c)
    # Per-class columns sorted
    per_class = sorted(c for c in cols if "_c" in c and c not in ordered)
    ordered.extend(per_class)
    # Remaining
    remaining = [c for c in cols if c not in ordered and c not in _ALIASES]
    ordered.extend(remaining)
    return ordered


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--folds", type=int, nargs="+", default=[0])
    ap.add_argument("--latex", action="store_true", help="Also emit LaTeX tables")
    ap.add_argument("--per-class", action="store_true", help="Include per-class columns")
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

    # ── Collect per-fold summary CSVs ──
    frames = []
    for fold in args.folds:
        pattern = f"metrics_{dataset_name}_fold{fold}_*.csv"
        for csv in sorted(results_dir.glob(pattern)):
            # Skip per-sample files
            if "per_sample" in csv.name:
                continue
            df = pd.read_csv(csv)
            df["fold"] = fold
            frames.append(df)

    if not frames:
        print(f"No result CSVs found in {results_dir}")
        return

    all_df = pd.concat(frames, ignore_index=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Determine metric columns
    meta_cols = {"dataset", "split", "method", "fold", "sample_id"}
    all_metric_cols = [c for c in all_df.columns if c not in meta_cols and c not in _ALIASES]
    if not args.per_class:
        all_metric_cols = [c for c in all_metric_cols if "_c" not in c]
    ordered_cols = _order_metric_cols(all_metric_cols)

    # ── Table 1: Layer1 single experts ──
    l1_single = all_df[all_df["method"].str.startswith("L1_")]
    if not l1_single.empty:
        disp_cols = ["method", "fold"] + [c for c in ordered_cols if c in l1_single.columns]
        out = tables_dir / "table1_L1_experts.csv"
        l1_single[disp_cols].to_csv(out, index=False)
        print(f"Wrote {out}")
        if args.latex:
            print(l1_single[disp_cols].to_latex(index=False, float_format="%.4f"))

    # ── Table 2: Layer2 single experts ──
    l2_single = all_df[all_df["method"].str.startswith("L2_")]
    if not l2_single.empty:
        disp_cols = ["method", "fold"] + [c for c in ordered_cols if c in l2_single.columns]
        out = tables_dir / "table2_L2_experts.csv"
        l2_single[disp_cols].to_csv(out, index=False)
        print(f"Wrote {out}")

    # ── Table 3: Ensemble / combiner results ──
    ensemble_tags = ["_mean", "_majority", "_ole", "_dt", "_we_clpso", "gating"]
    ens = all_df[all_df["method"].apply(
        lambda m: any(tag in m for tag in ensemble_tags)
    )]
    if not ens.empty:
        disp_cols = ["method", "fold"] + [c for c in ordered_cols if c in ens.columns]
        out = tables_dir / "table3_ensemble_methods.csv"
        ens[disp_cols].to_csv(out, index=False)
        print(f"Wrote {out}")
        if args.latex:
            print(ens[disp_cols].to_latex(index=False, float_format="%.4f"))

    # ── Table 4: All methods combined ──
    disp_cols = ["method", "fold"] + [c for c in ordered_cols if c in all_df.columns]
    out = tables_dir / "table4_all_methods.csv"
    all_df[disp_cols].to_csv(out, index=False)
    print(f"Wrote {out}")

    # ── Summary (mean ± std across folds) ──
    if len(args.folds) > 1:
        agg_cols = [c for c in ordered_cols if c in all_df.columns]
        summary = all_df.groupby("method")[agg_cols].agg(["mean", "std"])
        out = tables_dir / "table_summary_mean_std.csv"
        summary.to_csv(out)
        print(f"Wrote {out}")

        # Pretty-print summary with ± notation
        summary_pretty = pd.DataFrame(index=summary.index)
        for col in agg_cols:
            if (col, "mean") in summary.columns:
                summary_pretty[col] = summary.apply(
                    lambda r: f"{r[(col, 'mean')]:.4f}±{r[(col, 'std')]:.4f}"
                    if pd.notna(r[(col, "mean")]) else "—",
                    axis=1,
                )
        out = tables_dir / "table_summary_pretty.csv"
        summary_pretty.to_csv(out)
        print(f"Wrote {out}")

    # ── Statistical significance ──
    sig_frames = []
    for fold in args.folds:
        pattern = f"significance_{dataset_name}_fold{fold}_*.csv"
        for csv in sorted(results_dir.glob(pattern)):
            df = pd.read_csv(csv)
            df["fold"] = fold
            sig_frames.append(df)

    if sig_frames:
        sig_df = pd.concat(sig_frames, ignore_index=True)
        out = tables_dir / "table_significance.csv"
        sig_df.to_csv(out, index=False)
        print(f"Wrote {out}")

    print("Done.")


if __name__ == "__main__":
    main()

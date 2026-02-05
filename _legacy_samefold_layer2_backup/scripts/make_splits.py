from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from seg_moe.utils.io import ensure_dir, load_jsonl, save_json, save_jsonl
from seg_moe.utils.config import load_config


def _kfold_indices(n: int, k: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.array_split(idx, k)
    out = []
    for i in range(k):
        val = folds[i]
        train = np.concatenate([folds[j] for j in range(k) if j != i])
        out.append((train, val))
    return out


def _kfold_groups(group_ids: list[str], k: int, seed: int) -> list[tuple[set[str], set[str]]]:
    """Return (train_group_set, val_group_set) for each fold."""
    uniq = sorted(set(group_ids))
    rng = np.random.default_rng(seed)
    idx = np.arange(len(uniq))
    rng.shuffle(idx)
    folds = np.array_split(idx, k)
    out: list[tuple[set[str], set[str]]] = []
    for i in range(k):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        val_groups = {uniq[int(j)] for j in val_idx}
        train_groups = {uniq[int(j)] for j in train_idx}
        out.append((train_groups, val_groups))
    return out


def _get_group_key_value(row: dict, group_key: str) -> str:
    v = row.get(group_key)
    if v is None:
        # fallback: try parse from id using common convention "<group>_..."
        rid = str(row.get("id", ""))
        if "_" in rid:
            return rid.split("_", 1)[0]
        return rid
    return str(v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-config", required=True)
    ap.add_argument("--index-jsonl", default=None, help="Prepared master index jsonl (if prepare created it)")
    args = ap.parse_args()

    dcfg = load_config(args.dataset_config)
    splits_dir = Path(dcfg["paths"]["splits_dir"])
    ensure_dir(splits_dir)

    master_index = Path(args.index_jsonl) if args.index_jsonl else (splits_dir / "index_all.jsonl")
    if not master_index.exists():
        raise FileNotFoundError(f"Master index not found: {master_index}")

    rows = load_jsonl(master_index)

    split_cfg = dcfg["split"]
    stype = split_cfg["type"]
    folds = int(split_cfg.get("folds", 5))
    seed = int(split_cfg.get("seed", 42))
    group_key = split_cfg.get("group_key", None)

    # Expect rows have split=raw_train/raw_test or just 'all'
    if stype == "5fold_from_all":
        all_rows = [r | {"split": "all"} for r in rows]
        out_rows: List[Dict[str, Any]] = []

        if group_key:
            gids = [_get_group_key_value(r, str(group_key)) for r in all_rows]
            folds_groups = _kfold_groups(gids, folds, seed)
            for fi, (tr_g, va_g) in enumerate(folds_groups):
                for r in all_rows:
                    g = _get_group_key_value(r, str(group_key))
                    if g in va_g:
                        out_rows.append(r | {"split": f"val_fold{fi}"})
                    elif g in tr_g:
                        out_rows.append(r | {"split": f"train_fold{fi}"})
        else:
            kfold = _kfold_indices(len(all_rows), folds, seed)
            for fi, (tr, va) in enumerate(kfold):
                for j in tr:
                    out_rows.append(all_rows[int(j)] | {"split": f"train_fold{fi}"})
                for j in va:
                    out_rows.append(all_rows[int(j)] | {"split": f"val_fold{fi}"})
        save_jsonl(splits_dir / "splits_5fold.jsonl", out_rows)
        save_json(splits_dir / "meta.json", {"type": stype, "folds": folds, "seed": seed, "group_key": group_key})
        print(f"Wrote {splits_dir / 'splits_5fold.jsonl'}")
        return

    if stype == "holdout20_then_5fold":
        test_ratio = float(split_cfg.get("test_ratio", 0.2))
        out_rows: List[Dict[str, Any]] = []

        if group_key:
            gids = [_get_group_key_value(r, str(group_key)) for r in rows]
            uniq = sorted(set(gids))
            rng = np.random.default_rng(seed)
            rng.shuffle(uniq)
            n_test_g = int(round(len(uniq) * test_ratio))
            test_g = set(uniq[:n_test_g])
            train_g = set(uniq[n_test_g:])
            test_rows = [r | {"split": "test"} for r in rows if _get_group_key_value(r, str(group_key)) in test_g]
            train_pool = [r for r in rows if _get_group_key_value(r, str(group_key)) in train_g]

            train_gids = [_get_group_key_value(r, str(group_key)) for r in train_pool]
            folds_groups = _kfold_groups(train_gids, folds, seed)
            for fi, (tr_g, va_g) in enumerate(folds_groups):
                for r in train_pool:
                    g = _get_group_key_value(r, str(group_key))
                    if g in va_g:
                        out_rows.append(r | {"split": f"val_fold{fi}"})
                    elif g in tr_g:
                        out_rows.append(r | {"split": f"train_fold{fi}"})
        else:
            rng = np.random.default_rng(seed)
            idx = np.arange(len(rows))
            rng.shuffle(idx)
            n_test = int(round(len(rows) * test_ratio))
            test_idx = idx[:n_test]
            train_pool_idx = idx[n_test:]

            train_pool = [rows[int(i)] for i in train_pool_idx]
            test_rows = [rows[int(i)] | {"split": "test"} for i in test_idx]

            kfold = _kfold_indices(len(train_pool), folds, seed)
            for fi, (tr, va) in enumerate(kfold):
                for j in tr:
                    out_rows.append(train_pool[int(j)] | {"split": f"train_fold{fi}"})
                for j in va:
                    out_rows.append(train_pool[int(j)] | {"split": f"val_fold{fi}"})

        out_rows.extend(test_rows)
        save_jsonl(splits_dir / "splits_holdout20_5fold.jsonl", out_rows)
        save_json(
            splits_dir / "meta.json",
            {"type": stype, "folds": folds, "seed": seed, "test_ratio": test_ratio, "group_key": group_key},
        )
        print(f"Wrote {splits_dir / 'splits_holdout20_5fold.jsonl'}")
        return

    if stype == "train_5fold_test_fixed":
        # rows should include split=raw_train/raw_test
        train_rows = [r for r in rows if r.get("split") in ("raw_train", "train")]
        test_rows = [r | {"split": "test"} for r in rows if r.get("split") in ("raw_test", "test")]
        if not train_rows or not test_rows:
            raise ValueError("Expected raw_train and raw_test in master index for train_5fold_test_fixed")

        out_rows: List[Dict[str, Any]] = []

        if group_key:
            gids = [_get_group_key_value(r, str(group_key)) for r in train_rows]
            folds_groups = _kfold_groups(gids, folds, seed)
            for fi, (tr_g, va_g) in enumerate(folds_groups):
                for r in train_rows:
                    g = _get_group_key_value(r, str(group_key))
                    if g in va_g:
                        out_rows.append(r | {"split": f"val_fold{fi}"})
                    elif g in tr_g:
                        out_rows.append(r | {"split": f"train_fold{fi}"})
        else:
            kfold = _kfold_indices(len(train_rows), folds, seed)
            for fi, (tr, va) in enumerate(kfold):
                for j in tr:
                    out_rows.append(train_rows[int(j)] | {"split": f"train_fold{fi}"})
                for j in va:
                    out_rows.append(train_rows[int(j)] | {"split": f"val_fold{fi}"})

        out_rows.extend(test_rows)
        save_jsonl(splits_dir / "splits_train5fold_testfixed.jsonl", out_rows)
        save_json(splits_dir / "meta.json", {"type": stype, "folds": folds, "seed": seed, "group_key": group_key})
        print(f"Wrote {splits_dir / 'splits_train5fold_testfixed.jsonl'}")
        return

    raise ValueError(f"Unknown split.type: {stype}")


if __name__ == "__main__":
    main()

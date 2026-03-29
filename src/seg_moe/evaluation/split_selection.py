from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class EvalSelection:
    target_folds: list[int]
    eval_split: str | None
    run_all_folds: bool


def resolve_eval_selection(
    fold_arg: str,
    available_val_folds: Sequence[int],
    *,
    has_test_split: bool,
    predictor_fold: int = 0,
) -> EvalSelection:
    """Resolve evaluation mode from the CLI fold argument.

    Semantics:
    - ``k``: evaluate ``val_fold{k}``
    - ``all``: evaluate every validation fold
    - ``test``: evaluate the test split using checkpoints from ``predictor_fold``
    """

    fold_arg = str(fold_arg).strip().lower()
    available = sorted(int(f) for f in available_val_folds)
    if not available:
        raise ValueError("available_val_folds must not be empty")

    if fold_arg == "all":
        return EvalSelection(target_folds=available, eval_split=None, run_all_folds=True)

    if fold_arg == "test":
        predictor_fold = int(predictor_fold)
        if not has_test_split:
            raise ValueError("Requested test evaluation, but dataset splits do not contain 'test'")
        if predictor_fold not in available:
            raise ValueError(
                f"predictor_fold {predictor_fold} not found. Available val folds: {available}"
            )
        return EvalSelection(target_folds=[predictor_fold], eval_split="test", run_all_folds=False)

    fold = int(fold_arg)
    if fold not in available:
        raise ValueError(f"Requested fold {fold} not found. Available val folds: {available}")
    return EvalSelection(target_folds=[fold], eval_split=f"val_fold{fold}", run_all_folds=False)

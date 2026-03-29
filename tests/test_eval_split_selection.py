from __future__ import annotations

import pytest

from seg_moe.data.oof import parse_val_fold, resolve_prediction_cache_paths
from seg_moe.evaluation.split_selection import resolve_eval_selection


def test_parse_val_fold():
    assert parse_val_fold("val_fold0") == 0
    assert parse_val_fold("val_fold12") == 12
    assert parse_val_fold("test") is None


def test_resolve_eval_selection_numeric_fold_stays_on_validation():
    sel = resolve_eval_selection("0", [0, 1, 2], has_test_split=True, predictor_fold=2)
    assert sel.target_folds == [0]
    assert sel.eval_split == "val_fold0"
    assert sel.run_all_folds is False


def test_resolve_eval_selection_test_uses_predictor_fold():
    sel = resolve_eval_selection("test", [0, 1, 2], has_test_split=True, predictor_fold=2)
    assert sel.target_folds == [2]
    assert sel.eval_split == "test"
    assert sel.run_all_folds is False


def test_resolve_eval_selection_requires_existing_test_split():
    with pytest.raises(ValueError):
        resolve_eval_selection("test", [0, 1, 2], has_test_split=False, predictor_fold=0)


def test_resolve_prediction_cache_paths_for_validation_and_test():
    exp_cfg = {
        "exp_name": "demo_exp",
        "layering": {
            "cache_root": "runs/${exp_name}/cache",
            "oof_cache_dir": "runs/${exp_name}/cache/oof/layer1",
            "oof_manifest_path": "runs/${exp_name}/cache/oof/layer1/oof_manifest.jsonl",
            "l2_oof_cache_dir": "runs/${exp_name}/cache/oof/layer2",
            "l2_oof_manifest_path": "runs/${exp_name}/cache/oof/layer2/oof_manifest_layer2.jsonl",
        },
    }

    val_cache_dir, val_manifest = resolve_prediction_cache_paths(
        exp_cfg, "layer1", predictor_fold=0, split="val_fold0"
    )
    assert str(val_cache_dir).endswith(r"runs\demo_exp\cache\oof\layer1")
    assert str(val_manifest).endswith(r"runs\demo_exp\cache\oof\layer1\oof_manifest.jsonl")

    test_cache_dir, test_manifest = resolve_prediction_cache_paths(
        exp_cfg, "layer2", predictor_fold=3, split="test"
    )
    assert str(test_cache_dir).endswith(r"runs\demo_exp\cache\inference\layer2\fold_3\test")
    assert str(test_manifest).endswith(r"runs\demo_exp\cache\inference\layer2\fold_3\test\manifest.jsonl")

"""
Test 2: Cache logits write/read consistency.

Verifies that cached logits can be written and read back with
correct shape and case_id.
"""
import os
import tempfile

import numpy as np
import pytest
import torch

from seg_moe.utils.io import ensure_dir


def _save_logits(path, logits_np, case_id, shape):
    np.savez_compressed(
        str(path),
        logits=logits_np,
        meta={"case_id": case_id, "shape": list(shape)},
    )


def test_cache_roundtrip():
    """Write logits as npz, read back, verify shape and values."""
    M, D, H, W = 3, 16, 16, 16
    logits = np.random.randn(M, D, H, W).astype(np.float16)
    case_id = "liver_001"

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, f"{case_id}.npz")
        _save_logits(out_path, logits, case_id, logits.shape)

        with np.load(out_path, allow_pickle=True) as loaded:
            loaded_logits = loaded["logits"].copy()
            loaded_meta = loaded["meta"].item()

        assert loaded_logits.shape == logits.shape
        assert loaded_logits.dtype == np.float16
        np.testing.assert_array_equal(loaded_logits, logits)
        assert loaded_meta["case_id"] == case_id
        assert loaded_meta["shape"] == list(logits.shape)


def test_multi_expert_cache_shapes():
    """Cache 3 experts for same case, all shapes should match."""
    M, D, H, W = 3, 16, 16, 16
    experts = ["swin-unetr-base", "nnunet-v2", "segresnet-base"]

    with tempfile.TemporaryDirectory() as tmpdir:
        for ename in experts:
            edir = ensure_dir(os.path.join(tmpdir, ename))
            logits = np.random.randn(M, D, H, W).astype(np.float16)
            _save_logits(os.path.join(edir, "case_0001.npz"), logits, "case_0001", logits.shape)

        shapes = []
        for ename in experts:
            with np.load(os.path.join(tmpdir, ename, "case_0001.npz"), allow_pickle=True) as data:
                shapes.append(data["logits"].shape)

        assert all(s == shapes[0] for s in shapes), f"Shape mismatch across experts: {shapes}"


def test_cache_many_cases():
    """Cache and read back multiple cases."""
    M, D, H, W = 3, 8, 8, 8

    with tempfile.TemporaryDirectory() as tmpdir:
        n_cases = 10
        for i in range(n_cases):
            cid = f"case_{i:04d}"
            logits = np.random.randn(M, D, H, W).astype(np.float16)
            _save_logits(os.path.join(tmpdir, f"{cid}.npz"), logits, cid, logits.shape)

        files = sorted(f for f in os.listdir(tmpdir) if f.endswith(".npz"))
        assert len(files) == n_cases

        for f in files:
            with np.load(os.path.join(tmpdir, f), allow_pickle=True) as data:
                assert data["logits"].shape == (M, D, H, W)

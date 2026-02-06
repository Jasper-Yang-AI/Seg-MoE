"""
Test 6: End-to-end smoke test.

Uses tiny mock data to verify the full pipeline:
  build experts → forward pass → cache logits → load & fuse → eval metrics
"""
import os
import tempfile

import numpy as np
import pytest
import torch

from seg_moe.models.experts.factory import ExpertFactory
from seg_moe.utils.config import load_config


@pytest.fixture(scope="module")
def experts():
    cfg = load_config("configs/3d/experts.yaml")
    factory = ExpertFactory(cfg)
    return factory.build_all(in_channels=1, classes=3)


def test_end2end_smoke(experts):
    """
    Full smoke pipeline:
    1) Build 3 experts
    2) Forward pass on random [1, 1, 32, 32, 32]
    3) Cache logits to temp dir
    4) Load cached logits
    5) Mean-fuse
    6) Compute argmax → Dice
    """
    B, C, D, H, W = 1, 1, 64, 64, 64  # SwinUNETR needs >=64
    M = 3  # num classes
    K = len(experts)
    x = torch.randn(B, C, D, H, W)

    # 1-2: forward pass
    all_logits = []
    for exp in experts:
        exp.eval()
        with torch.no_grad():
            out = exp.predict_logits(x)
        assert out.shape == (B, M, D, H, W), f"{exp.name}: shape {out.shape}"
        all_logits.append(out)

    # 3: cache to disk
    with tempfile.TemporaryDirectory() as tmpdir:
        for exp, logits in zip(experts, all_logits):
            edir = os.path.join(tmpdir, exp.name)
            os.makedirs(edir, exist_ok=True)
            logits_np = logits.numpy()[0].astype(np.float16)
            np.savez_compressed(
                os.path.join(edir, "case_0000.npz"),
                logits=logits_np,
                meta={"case_id": "case_0000", "shape": list(logits_np.shape)},
            )

        # 4: load
        loaded_logits = []
        for exp in experts:
            with np.load(os.path.join(tmpdir, exp.name, "case_0000.npz"), allow_pickle=True) as data:
                loaded = torch.from_numpy(data["logits"].astype(np.float32)).unsqueeze(0)
                loaded_logits.append(loaded)
                assert loaded.shape == (1, M, D, H, W)

    # 5: mean fuse
    stacked = torch.stack(loaded_logits, dim=0)
    fused = stacked.mean(dim=0)
    assert fused.shape == (B, M, D, H, W)

    # 6: eval (Dice against random label)
    pred = fused.argmax(dim=1)  # [B, D, H, W]
    label = torch.randint(0, M, (B, D, H, W))

    dices = []
    for c in range(1, M):
        p = (pred == c).float()
        t = (label == c).float()
        inter = (p * t).sum()
        union = p.sum() + t.sum()
        dice = (2 * inter / (union + 1e-7)).item()
        dices.append(dice)

    mean_dice = float(np.mean(dices))
    # With random data, dice is low — just check it's computable
    assert 0.0 <= mean_dice <= 1.0, f"Invalid dice: {mean_dice}"
    print(f"\n  End-to-end smoke: K={K}, mean_dice={mean_dice:.4f} (expected low on random data)")


def test_train_smoke():
    """Minimal 1-step training smoke test."""
    cfg = load_config("configs/3d/experts.yaml")
    factory = ExpertFactory(cfg)
    # Just train segresnet (fastest)
    expert = factory.build_one("segresnet", in_channels=1, classes=3)
    expert.train()

    x = torch.randn(1, 1, 64, 64, 64)
    y = torch.randint(0, 3, (1, 64, 64, 64))

    from seg_moe.training.losses import ce_plus_dice
    logits = expert(x)
    loss = ce_plus_dice(logits, y, num_classes=3)
    loss.backward()

    assert loss.item() > 0
    assert loss.isfinite()
    print(f"\n  Train smoke: loss={loss.item():.4f}")

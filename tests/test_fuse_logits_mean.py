"""
Test 3: Mean logits fusion forward shape.

Verifies that averaging logits from K experts produces correct shape.
"""
import numpy as np
import pytest
import torch


def mean_fuse_logits(logits_list):
    """Mean fusion of K expert logits.

    Args:
        logits_list: list of K tensors, each [B, M, D, H, W]

    Returns:
        Mean logits [B, M, D, H, W]
    """
    stacked = torch.stack(logits_list, dim=0)  # [K, B, M, D, H, W]
    return stacked.mean(dim=0)  # [B, M, D, H, W]


def test_mean_fuse_shape():
    B, M, D, H, W = 2, 3, 16, 16, 16
    K = 3
    logits = [torch.randn(B, M, D, H, W) for _ in range(K)]
    fused = mean_fuse_logits(logits)
    assert fused.shape == (B, M, D, H, W)


def test_mean_fuse_correct_values():
    B, M, D, H, W = 1, 3, 4, 4, 4
    a = torch.ones(B, M, D, H, W) * 2.0
    b = torch.ones(B, M, D, H, W) * 4.0
    c = torch.ones(B, M, D, H, W) * 6.0
    fused = mean_fuse_logits([a, b, c])
    assert torch.allclose(fused, torch.ones_like(fused) * 4.0)


def test_mean_fuse_single_expert():
    """With 1 expert, mean fusion == identity."""
    B, M, D, H, W = 1, 3, 8, 8, 8
    logits = torch.randn(B, M, D, H, W)
    fused = mean_fuse_logits([logits])
    assert torch.allclose(fused, logits)


def test_fused_logits_not_softmax():
    """Fused logits should NOT sum to 1 along class dim."""
    B, M, D, H, W = 1, 3, 8, 8, 8
    logits = [torch.randn(B, M, D, H, W) for _ in range(3)]
    fused = mean_fuse_logits(logits)
    class_sum = fused.sum(dim=1)
    all_ones = torch.allclose(class_sum, torch.ones_like(class_sum), atol=0.01)
    assert not all_ones, "Fused output looks like softmax"

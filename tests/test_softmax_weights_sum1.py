"""
Test 5: Softmax weights sum to 1.

Tests that gating/fusion weight schemes produce valid probability
distributions (sum_k = 1 for each spatial location / class).
"""
import pytest
import torch
import torch.nn.functional as F


def compute_softmax_weights(raw_weights):
    """Given raw gating logits [B, K, ...], return softmax weights summing to 1 over K."""
    return F.softmax(raw_weights, dim=1)


def test_softmax_weights_sum_to_1_global():
    """Global weights: [B, K] → sum over K = 1."""
    B, K = 2, 3
    raw = torch.randn(B, K)
    w = compute_softmax_weights(raw)
    assert w.shape == (B, K)
    sums = w.sum(dim=1)
    assert torch.allclose(sums, torch.ones(B), atol=1e-5)


def test_softmax_weights_sum_to_1_spatial():
    """Spatial weights: [B, K, D, H, W] → sum over K = 1 at every voxel."""
    B, K, D, H, W = 1, 3, 8, 8, 8
    raw = torch.randn(B, K, D, H, W)
    w = compute_softmax_weights(raw)
    sums = w.sum(dim=1)  # [B, D, H, W]
    expected = torch.ones(B, D, H, W)
    assert torch.allclose(sums, expected, atol=1e-5)


def test_softmax_weights_sum_to_1_per_class():
    """Per-class weights: [B, K, M, D, H, W] → sum over K = 1."""
    B, K, M, D, H, W = 1, 3, 3, 4, 4, 4
    raw = torch.randn(B, K, M, D, H, W)
    w = compute_softmax_weights(raw)
    sums = w.sum(dim=1)  # [B, M, D, H, W]
    expected = torch.ones(B, M, D, H, W)
    assert torch.allclose(sums, expected, atol=1e-5)


def test_softmax_weights_positive():
    """All softmax weights should be >= 0."""
    raw = torch.randn(2, 3, 4, 4, 4)
    w = compute_softmax_weights(raw)
    assert (w >= 0).all()


def test_weighted_fusion():
    """Weighted fusion: sum_k( w_k * logits_k ) preserves shape."""
    B, K, M, D, H, W = 1, 3, 3, 8, 8, 8
    logits = torch.randn(B, K, M, D, H, W)
    raw_w = torch.randn(B, K, 1, 1, 1, 1)  # broadcast over M, D, H, W
    w = F.softmax(raw_w, dim=1)
    fused = (w * logits).sum(dim=1)  # [B, M, D, H, W]
    assert fused.shape == (B, M, D, H, W)

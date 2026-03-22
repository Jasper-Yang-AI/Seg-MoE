"""Unit tests for the 2D patch-level gating pipeline."""
from __future__ import annotations

import numpy as np
import torch

from seg_moe.gating.patch_gating_2d import (
    PatchConvGate2D,
    PatchGatingConfig,
    compute_load_balance_loss,
    compute_temperature,
)
from seg_moe.utils.patches import (
    compute_patch_positions,
    merge_patches_2d,
    split_into_patches_2d,
)

K, M = 3, 3
PH, PW = 64, 64


def _default_cfg(**kw) -> PatchGatingConfig:
    return PatchGatingConfig(
        num_experts=K,
        num_classes=M,
        patch_size=PH,
        stride=32,
        hidden_dim=32,
        dropout=0.0,
        **kw,
    )


class TestPatchConvGate2D:
    def test_output_shape_structured_logits(self):
        cfg = _default_cfg(per_class=False)
        model = PatchConvGate2D(cfg)
        x = torch.randn(4, K, M, PH, PW)
        w = model(x)
        assert w.shape == (4, K)

    def test_output_shape_per_expert(self):
        model = PatchConvGate2D(_default_cfg(per_class=False))
        x = torch.randn(4, K, M, PH, PW)
        w = model(x)
        assert w.shape == (4, K)

    def test_context_forward_shape(self):
        cfg = _default_cfg(
            per_class=True,
            use_layer1_semantics=True,
            use_image_context=True,
            use_position_channels=True,
            use_slice_position=True,
            use_prior_agreement_features=True,
        )
        model = PatchConvGate2D(cfg)
        x = torch.randn(2, K, M, PH, PW)
        extra = {
            "image": torch.randn(2, 3, PH, PW),
            "layer1_mean": torch.randn(2, M, PH, PW).softmax(dim=1),
            "layer1_entropy": torch.rand(2, 1, PH, PW),
            "layer1_disagreement": torch.rand(2, M, PH, PW),
            "coords": torch.rand(2, 2, PH, PW),
            "slice_pos": torch.rand(2, 1),
        }
        w = model(x, extra=extra)
        assert w.shape == (2, K, M)

    def test_output_shape_per_class(self):
        model = PatchConvGate2D(_default_cfg(per_class=True))
        x = torch.randn(4, K, M, PH, PW)
        w = model(x)
        assert w.shape == (4, K, M)

    def test_legacy_flattened_input_is_supported(self):
        model = PatchConvGate2D(_default_cfg(per_class=False))
        x = torch.randn(4, K * M, PH, PW)
        w = model(x)
        assert w.shape == (4, K)

    def test_weights_sum_to_one(self):
        model = PatchConvGate2D(_default_cfg(per_class=False))
        x = torch.randn(8, K, M, PH, PW)
        w = model(x)
        assert torch.allclose(w.sum(dim=1), torch.ones(8), atol=1e-5)

    def test_weights_sum_to_one_per_class(self):
        model = PatchConvGate2D(_default_cfg(per_class=True))
        x = torch.randn(8, K, M, PH, PW)
        w = model(x)
        assert torch.allclose(w.sum(dim=1), torch.ones(8, M), atol=1e-5)

    def test_fuse_probs_shape(self):
        model = PatchConvGate2D(_default_cfg(per_class=False))
        probs = torch.randn(4, K, M, PH, PW).softmax(dim=2)
        w = torch.randn(4, K).softmax(dim=1)
        fused = model.fuse_probs(probs, w)
        assert fused.shape == (4, M, PH, PW)

    def test_fuse_probs_per_class_shape(self):
        model = PatchConvGate2D(_default_cfg(per_class=True))
        probs = torch.randn(4, K, M, PH, PW).softmax(dim=2)
        w = torch.randn(4, K, M).softmax(dim=1)
        fused = model.fuse_probs(probs, w)
        assert fused.shape == (4, M, PH, PW)

    def test_temperature_effect(self):
        model = PatchConvGate2D(_default_cfg())
        x = torch.randn(16, K, M, PH, PW)
        w_hot = model(x, temperature=0.1)
        w_cold = model(x, temperature=10.0)
        ent_hot = -(w_hot * (w_hot + 1e-8).log()).sum(dim=1).mean()
        ent_cold = -(w_cold * (w_cold + 1e-8).log()).sum(dim=1).mean()
        assert ent_cold > ent_hot


class TestLoadBalance:
    def test_uniform_minimum(self):
        w = torch.ones(100, K) / K
        loss = compute_load_balance_loss(w)
        assert abs(float(loss) - 1.0) < 1e-5

    def test_collapsed_maximum(self):
        w = torch.zeros(100, K)
        w[:, 0] = 1.0
        loss = compute_load_balance_loss(w)
        assert float(loss) > K - 0.1


class TestTemperature:
    def test_endpoints(self):
        assert abs(compute_temperature(0, 50, 2.0, 0.5) - 2.0) < 1e-5
        assert abs(compute_temperature(49, 50, 2.0, 0.5) - 0.5) < 1e-5

    def test_monotonic_decrease(self):
        temps = [compute_temperature(e, 50, 2.0, 0.5) for e in range(50)]
        for i in range(len(temps) - 1):
            assert temps[i] >= temps[i + 1]


class TestPatches:
    def test_positions_non_overlap(self):
        pos = compute_patch_positions(256, 256, 64, 64)
        assert len(pos) == 16

    def test_positions_overlap(self):
        pos = compute_patch_positions(256, 256, 64, 32)
        assert len(pos) == 49

    def test_split_count(self):
        x = np.random.randn(3, 256, 256).astype(np.float32)
        patches, positions = split_into_patches_2d(x, 64, 32)
        assert len(patches) == 49
        assert len(positions) == 49
        assert patches[0].shape == (3, 64, 64)

    def test_split_merge_identity_non_overlap(self):
        x = np.random.randn(3, 256, 256).astype(np.float32)
        patches, positions = split_into_patches_2d(x, 64, 64)
        merged = merge_patches_2d(patches, positions, (256, 256), 64, blend_mode="average")
        assert np.allclose(merged, x, atol=1e-5)

    def test_split_merge_overlap_smooth(self):
        x = np.ones((3, 256, 256), dtype=np.float32) * 0.5
        patches, positions = split_into_patches_2d(x, 64, 32)
        merged = merge_patches_2d(patches, positions, (256, 256), 64, blend_mode="gaussian")
        assert np.allclose(merged, 0.5, atol=1e-4)

    def test_torch_tensor_support(self):
        x = torch.randn(3, 128, 128)
        patches, positions = split_into_patches_2d(x, 64, 64)
        assert len(patches) == 4
        assert patches[0].shape == (3, 64, 64)

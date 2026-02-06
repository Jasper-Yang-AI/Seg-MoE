"""
Test 1: Three experts output shape/dtype alignment.

Verifies that all 3 experts (nnunet, swin_unetr, segresnet) produce
logits with identical shape [B, M, D, H, W] and dtype.
"""
import pytest
import torch

from seg_moe.models.experts.factory import ExpertFactory
from seg_moe.utils.config import load_config


@pytest.fixture(scope="module")
def experts():
    cfg = load_config("configs/3d/experts.yaml")
    factory = ExpertFactory(cfg)
    return factory.build_all(in_channels=1, classes=3)


@pytest.fixture(scope="module")
def dummy_input():
    return torch.randn(1, 1, 64, 64, 64)  # SwinUNETR needs >=64 per dim


def test_three_experts_built(experts):
    assert len(experts) == 3, f"Expected 3 experts, got {len(experts)}"


def test_all_have_name(experts):
    names = [e.name for e in experts]
    assert all(isinstance(n, str) and len(n) > 0 for n in names)
    assert len(set(names)) == 3, "Expert names must be unique"


def test_all_have_num_classes(experts):
    for e in experts:
        assert e.num_classes == 3


def test_output_shapes_match(experts, dummy_input):
    shapes = []
    for e in experts:
        e.eval()
        with torch.no_grad():
            out = e.predict_logits(dummy_input)
        shapes.append(out.shape)
        assert out.ndim == 5, f"{e.name}: expected 5D, got {out.ndim}D"
        assert out.shape[0] == 1, f"{e.name}: batch dim mismatch"
        assert out.shape[1] == 3, f"{e.name}: class dim mismatch"

    # All shapes identical
    assert all(s == shapes[0] for s in shapes), f"Shape mismatch: {shapes}"


def test_output_dtypes_match(experts, dummy_input):
    dtypes = []
    for e in experts:
        e.eval()
        with torch.no_grad():
            out = e.predict_logits(dummy_input)
        dtypes.append(out.dtype)
    assert all(d == dtypes[0] for d in dtypes), f"Dtype mismatch: {dtypes}"


def test_output_is_logits_not_softmax(experts, dummy_input):
    """Logits should NOT sum to 1 along class dim (that would be softmax)."""
    for e in experts:
        e.eval()
        with torch.no_grad():
            out = e.predict_logits(dummy_input)
        class_sum = out.sum(dim=1)
        # If it were softmax, every spatial location would sum to ~1.0
        # With raw logits, this is very unlikely
        all_ones = torch.allclose(class_sum, torch.ones_like(class_sum), atol=0.01)
        assert not all_ones, f"{e.name}: output looks like softmax, not logits"

"""
Test 4: Non-overlapping patch split/merge identity.

Verifies that splitting a volume into non-overlapping patches and
reassembling gives back the original volume.
"""
import pytest
import torch


def split_patches(volume, patch_size):
    """Split [B, C, D, H, W] into non-overlapping patches.

    Returns: list of (patch_tensor, (d_start, h_start, w_start))
    """
    B, C, D, H, W = volume.shape
    pd, ph, pw = patch_size
    patches = []
    for d in range(0, D, pd):
        for h in range(0, H, ph):
            for w in range(0, W, pw):
                patch = volume[:, :, d:d+pd, h:h+ph, w:w+pw]
                patches.append((patch, (d, h, w)))
    return patches


def merge_patches(patches, full_shape, patch_size):
    """Merge patches back into full volume."""
    out = torch.zeros(full_shape)
    pd, ph, pw = patch_size
    for patch, (d, h, w) in patches:
        actual_pd = patch.shape[2]
        actual_ph = patch.shape[3]
        actual_pw = patch.shape[4]
        out[:, :, d:d+actual_pd, h:h+actual_ph, w:w+actual_pw] = patch
    return out


def test_split_merge_identity_exact():
    """Volume shape is exact multiple of patch size."""
    B, C, D, H, W = 1, 3, 32, 32, 32
    ps = (16, 16, 16)
    vol = torch.randn(B, C, D, H, W)
    patches = split_patches(vol, ps)
    assert len(patches) == 8  # 2*2*2
    restored = merge_patches(patches, vol.shape, ps)
    assert torch.allclose(vol, restored)


def test_split_merge_non_exact():
    """Volume not exact multiple — last patches are smaller."""
    B, C, D, H, W = 1, 3, 30, 30, 30
    ps = (16, 16, 16)
    vol = torch.randn(B, C, D, H, W)
    patches = split_patches(vol, ps)
    restored = merge_patches(patches, vol.shape, ps)
    assert torch.allclose(vol, restored)


def test_split_merge_single_patch():
    B, C, D, H, W = 1, 3, 16, 16, 16
    ps = (16, 16, 16)
    vol = torch.randn(B, C, D, H, W)
    patches = split_patches(vol, ps)
    assert len(patches) == 1
    restored = merge_patches(patches, vol.shape, ps)
    assert torch.allclose(vol, restored)

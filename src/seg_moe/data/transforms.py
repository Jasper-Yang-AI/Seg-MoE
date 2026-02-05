from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np


def imagenet_normalize(image: np.ndarray) -> np.ndarray:
    # image: HWC, uint8 or float32
    img = image.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    if img.shape[2] == 1:
        # replicate grayscale to 3 channels before normalize
        img = np.repeat(img, 3, axis=2)
    img = (img - mean) / std
    return img


def build_albu(augs_cfg: Dict[str, Any], is_train: bool) -> Optional[Any]:
    import albumentations as A

    cfg_list = augs_cfg["train"] if is_train else augs_cfg["val"]
    transforms = []
    for t in cfg_list:
        name = t["name"]
        if name == "NoOp":
            continue
        cls = getattr(A, name)
        kwargs = {k: v for k, v in t.items() if k != "name"}
        transforms.append(cls(**kwargs))
    if not transforms:
        return None
    return A.Compose(transforms)


def build_albu_for_layer2_with_probs(augs_cfg: Dict[str, Any], is_train: bool) -> Tuple[Optional[Any], Optional[Any]]:
    """Build augmentation pipelines for layer2 concatenated inputs.

    We need spatial transforms (flip/rotate/etc.) to be applied consistently to:
      - image
      - mask
      - probs (OOF probability maps)

    But image-only transforms (brightness/contrast, noise) must NOT be applied to probs.

    Returns:
      (spatial_aug, image_only_aug)
    """

    import albumentations as A

    try:
        from albumentations.core.transforms_interface import ImageOnlyTransform
    except Exception:  # pragma: no cover
        ImageOnlyTransform = ()  # type: ignore

    cfg_list = augs_cfg["train"] if is_train else augs_cfg["val"]
    spatial = []
    image_only = []

    for t in cfg_list:
        name = t["name"]
        if name == "NoOp":
            continue
        cls = getattr(A, name)
        kwargs = {k: v for k, v in t.items() if k != "name"}
        tr = cls(**kwargs)
        if ImageOnlyTransform and isinstance(tr, ImageOnlyTransform):
            image_only.append(tr)
        else:
            spatial.append(tr)

    spatial_aug = A.Compose(spatial, additional_targets={"probs": "image"}) if spatial else None
    image_aug = A.Compose(image_only) if image_only else None
    return spatial_aug, image_aug

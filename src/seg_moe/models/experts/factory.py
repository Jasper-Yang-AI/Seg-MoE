"""
ExpertFactory — config-driven construction of heterogeneous 3D experts.

Usage:
    from seg_moe.models.experts.factory import ExpertFactory

    factory = ExpertFactory(models_cfg)      # from YAML
    experts = factory.build_all(in_ch=1, classes=3)
    for expert in experts:
        logits = expert.predict_logits(x)    # [B, M, D, H, W]
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from seg_moe.models.experts.base_expert import BaseExpert


# Registry: architecture key → builder function
_EXPERT_BUILDERS = {}


def _register_builders() -> None:
    """Lazy registration so imports only happen when needed."""
    if _EXPERT_BUILDERS:
        return

    from seg_moe.models.experts.segresnet_expert import build_segresnet_expert
    from seg_moe.models.experts.swinunetr_expert import build_swinunetr_expert
    from seg_moe.models.experts.nnunet_expert import build_nnunet_expert

    _EXPERT_BUILDERS.update({
        "segresnet": build_segresnet_expert,
        "seg_resnet": build_segresnet_expert,
        "swin_unetr": build_swinunetr_expert,
        "swinunetr": build_swinunetr_expert,
        "nnunet": build_nnunet_expert,
        "nnunet_v2": build_nnunet_expert,
    })


class ExpertFactory:
    """Build experts from a ``models_sota.yaml`` config.

    Config schema (``sota_experts`` list):
      - architecture: "segresnet"
        name: "segresnet-base"
        enabled: true
        config:
          init_filters: 32
          ...
    """

    def __init__(self, models_cfg: Dict[str, Any]) -> None:
        self.raw_cfg = models_cfg
        self.expert_cfgs: List[Dict[str, Any]] = models_cfg.get("sota_experts", [])

    def build_all(
        self,
        in_channels: int = 1,
        classes: int = 3,
        *,
        only_enabled: bool = True,
    ) -> List[BaseExpert]:
        """Construct all (enabled) experts.

        Returns a list of ``BaseExpert`` instances in config order.
        """
        _register_builders()
        experts: List[BaseExpert] = []
        for ecfg in self.expert_cfgs:
            if only_enabled and not ecfg.get("enabled", True):
                continue
            arch = ecfg["architecture"].lower().replace("-", "_")
            builder = _EXPERT_BUILDERS.get(arch)
            if builder is None:
                raise ValueError(
                    f"Unknown expert architecture '{arch}'. "
                    f"Available: {sorted(set(_EXPERT_BUILDERS.values()), key=lambda f: f.__name__)}"
                )
            inner_cfg = dict(ecfg.get("config", {}))
            inner_cfg.setdefault("name", ecfg.get("name", arch))
            expert = builder(in_channels=in_channels, out_channels=classes, config=inner_cfg)
            experts.append(expert)
        return experts

    def build_one(
        self,
        arch: str,
        in_channels: int = 1,
        classes: int = 3,
        config_overrides: Optional[Dict[str, Any]] = None,
    ) -> BaseExpert:
        """Build a single expert by architecture key."""
        _register_builders()
        arch_key = arch.lower().replace("-", "_")
        builder = _EXPERT_BUILDERS.get(arch_key)
        if builder is None:
            raise ValueError(f"Unknown expert architecture '{arch}'")

        # find matching cfg in yaml, else use empty
        matched = [e for e in self.expert_cfgs if e["architecture"].lower().replace("-", "_") == arch_key]
        base = dict((matched[0].get("config", {}) if matched else {}))
        if config_overrides:
            base.update(config_overrides)
        if matched:
            base.setdefault("name", matched[0].get("name", arch))
        return builder(in_channels=in_channels, out_channels=classes, config=base)

    @property
    def expert_names(self) -> List[str]:
        return [e.get("name", e["architecture"]) for e in self.expert_cfgs if e.get("enabled", True)]

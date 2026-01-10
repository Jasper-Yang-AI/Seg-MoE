from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from omegaconf import OmegaConf


def load_config(path: str | Path) -> Dict[str, Any]:
    cfg = OmegaConf.load(str(path))
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]


def merge_configs(*cfgs: Dict[str, Any]) -> Dict[str, Any]:
    base = OmegaConf.create({})
    for c in cfgs:
        base = OmegaConf.merge(base, OmegaConf.create(c))
    return OmegaConf.to_container(base, resolve=True)  # type: ignore[return-value]


def apply_debug_overrides(cfg: Dict[str, Any], debug_cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not debug_cfg:
        return cfg
    overrides = debug_cfg.get("overrides", {})
    if not overrides:
        return cfg
    return merge_configs(cfg, overrides)


def resolve_run_dir(exp_cfg: Dict[str, Any]) -> Path:
    exp_name = exp_cfg["exp_name"]
    run_dir = Path(exp_cfg.get("output", {}).get("run_dir", f"runs/{exp_name}"))
    # resolve ${exp_name} placeholders (simple)
    run_dir_str = str(run_dir).replace("${exp_name}", exp_name)
    return Path(run_dir_str)

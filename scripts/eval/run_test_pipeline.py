"""Run the full 2D fixed-test evaluation pipeline in one command.

This wrapper sequentially executes:
1. Layer1 test cache generation
2. Layer2 test cache generation
3. Gating inference on the test split
4. Comprehensive method evaluation on the test split
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _extend_opt(cmd: list[str], flag: str, value: str | int | None) -> None:
    if value is not None:
        cmd.extend([flag, str(value)])


def _extend_flag(cmd: list[str], flag: str, enabled: bool) -> None:
    if enabled:
        cmd.append(flag)


def build_commands(args: argparse.Namespace) -> list[list[str]]:
    predictor_fold = int(args.predictor_fold)

    l1_cmd = [
        "scripts/inference/generate_layer1_oof.py",
        "--exp", args.exp,
        "--models", args.models,
        "--which", args.which,
        "--fold", str(predictor_fold),
        "--split", "test",
    ]
    _extend_opt(l1_cmd, "--limit", args.limit)
    _extend_opt(l1_cmd, "--batch-size", args.batch_size)
    _extend_flag(l1_cmd, "--tta", bool(args.tta))
    _extend_flag(l1_cmd, "--skip-existing", bool(args.skip_existing))

    l2_cmd = [
        "scripts/inference/generate_layer2_oof.py",
        "--exp", args.exp,
        "--models", args.models,
        "--which", args.which,
        "--fold", str(predictor_fold),
        "--split", "test",
    ]
    _extend_opt(l2_cmd, "--limit", args.limit)
    _extend_opt(l2_cmd, "--batch-size", args.batch_size)
    _extend_flag(l2_cmd, "--tta", bool(args.tta))
    _extend_flag(l2_cmd, "--no-uncertainty", bool(args.no_uncertainty))
    _extend_flag(l2_cmd, "--skip-existing", bool(args.skip_existing))

    gating_cmd = [
        "scripts/inference/gating_inference.py",
        "--exp", args.exp,
        "--gating-config", args.gating_config,
        "--models", args.models,
        "--fold", str(predictor_fold),
        "--split", "test",
    ]
    _extend_opt(gating_cmd, "--gpus", args.gpus)

    eval_cmd = [
        "scripts/eval/eval_methods.py",
        "--exp", args.exp,
        "--training", args.training,
        "--models", args.models,
        "--fold", "test",
        "--predictor-fold", str(predictor_fold),
        "--which", args.which,
    ]
    _extend_opt(eval_cmd, "--gpus", args.gpus)
    _extend_flag(eval_cmd, "--skip-live-inference", bool(args.skip_live_inference))
    _extend_flag(eval_cmd, "--no-uncertainty", bool(args.no_uncertainty))
    _extend_flag(
        eval_cmd,
        "--allow-missing-gating-cache",
        bool(args.allow_missing_gating_cache),
    )

    return [l1_cmd, l2_cmd, gating_cmd, eval_cmd]


def _with_pythonpath(env: dict[str, str]) -> dict[str, str]:
    out = dict(env)
    src = str(ROOT / "src")
    existing = out.get("PYTHONPATH", "")
    out["PYTHONPATH"] = src if not existing else src + os.pathsep + existing
    return out


def _run_command(step_idx: int, total_steps: int, args: list[str], env: dict[str, str]) -> None:
    cmd = [sys.executable, *args]
    rendered = " ".join(shlex.quote(part) for part in cmd)
    print("\n" + "=" * 88)
    print(f"[Step {step_idx}/{total_steps}] {rendered}")
    print("=" * 88)
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the full fixed-test evaluation pipeline")
    ap.add_argument("--exp", required=True)
    ap.add_argument("--training", required=True)
    ap.add_argument("--gating-config", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--predictor-fold", type=int, default=0,
                    help="Checkpoint fold used to run the fixed test split")
    ap.add_argument("--which", choices=["best", "last"], default="best")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None,
                    help="Optional sample limit for smoke runs")
    ap.add_argument("--gpus", type=str, default=None)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--skip-live-inference", action="store_true")
    ap.add_argument("--no-uncertainty", action="store_true")
    ap.add_argument("--allow-missing-gating-cache", action="store_true")
    args = ap.parse_args()

    commands = build_commands(args)
    env = _with_pythonpath(os.environ)
    total = len(commands)
    for idx, cmd in enumerate(commands, start=1):
        _run_command(idx, total, cmd, env)

    print("\nTest pipeline finished.")


if __name__ == "__main__":
    main()

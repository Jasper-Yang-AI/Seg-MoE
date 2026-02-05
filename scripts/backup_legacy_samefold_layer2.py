from __future__ import annotations

import argparse
import datetime as _dt
import shutil
import subprocess
from pathlib import Path
from typing import Iterable


KEYWORDS = [
    "layer1",
    "layer2",
    "stack",
    "oof",
    "fold",
    "kfold",
    "prob",
    "logits",
    "concat",
    "feature",
    "pseudo",
    "ensemble",
    "combiner",
    "stage2",
    "second_stage",
]

ALWAYS_INCLUDE = [
    "scripts/train_2d_experts.py",
    "scripts/train_layer2.py",
    "scripts/cache_probs.py",
    "scripts/make_splits.py",
    "scripts/eval_methods.py",
    "src/seg_moe/data/layer2_dataset.py",
    "src/seg_moe/data/layer2_oof_dataset.py",
    "src/seg_moe/data/oof.py",
    "src/seg_moe/data/dataset_2d.py",
    "src/seg_moe/data/transforms.py",
    "src/seg_moe/data/indexing.py",
    "configs/2d/augs.yaml",
    "configs/2d/training.yaml",
    "configs/2d/models.yaml",
    "configs/2d/experiment.yaml",
]


def _try_git_hash(repo_root: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.STDOUT,
            text=True,
        ).strip()
        return out
    except Exception:
        return "N/A"


def _iter_repo_files(repo_root: Path) -> Iterable[Path]:
    exts = {".py", ".yaml", ".yml", ".md"}
    for path in repo_root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in exts:
            continue
        rel = path.relative_to(repo_root).as_posix()
        if rel.startswith("_legacy_samefold_layer2_backup/"):
            continue
        if rel.startswith(".git/"):
            continue
        yield path


def _matches_keywords(rel_posix: str) -> bool:
    s = rel_posix.lower()
    return any(k in s for k in KEYWORDS)


def _is_entry_script(rel_posix: str) -> bool:
    name = Path(rel_posix).name.lower()
    return name in {"train.py", "main.py", "runner.py"}


def _collect_files(repo_root: Path) -> list[Path]:
    selected: dict[str, Path] = {}

    for rel in ALWAYS_INCLUDE:
        p = repo_root / rel
        if p.exists() and p.is_file():
            selected[rel] = p

    for p in _iter_repo_files(repo_root):
        rel = p.relative_to(repo_root).as_posix()
        if _matches_keywords(rel) or _is_entry_script(rel):
            selected[rel] = p

    return [selected[k] for k in sorted(selected.keys())]


def _resolve_backup_root(repo_root: Path) -> Path:
    base = repo_root / "_legacy_samefold_layer2_backup"
    if not base.exists():
        return base
    date = _dt.date.today().isoformat()
    return base / date


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".", help="Repository root (default: .)")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    backup_root = _resolve_backup_root(repo_root)

    files = _collect_files(repo_root)
    backup_root.mkdir(parents=True, exist_ok=True)

    for src in files:
        rel = src.relative_to(repo_root)
        dst = backup_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    git_hash = _try_git_hash(repo_root)
    ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    readme = backup_root / "README.md"
    readme.write_text(
        "# Legacy backup: same-fold layer2 stacking\n\n"
        "这是修改前的同fold概率图拼接版本，用于实验留档。\n\n"
        f"- 备份时间: {ts}\n"
        f"- Git commit: {git_hash}\n"
        f"- 备份文件数: {len(files)}\n",
        encoding="utf-8",
    )

    print(f"Backup created at: {backup_root}")
    print(f"Copied files: {len(files)}")


if __name__ == "__main__":
    main()

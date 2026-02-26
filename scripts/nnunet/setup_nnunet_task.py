#!/usr/bin/env python
"""
Setup nnUNet v2 official training environment.

支持两种数据源:
  1. MSD 数据集 → 自动转换为 nnUNet v2 格式
  2. 已有 nnUNet 格式数据 → 直接 symlink + 生成 dataset.json

该脚本完成以下步骤:
  1. 设置 nnUNet 环境变量 (nnUNet_raw, nnUNet_preprocessed, nnUNet_results)
  2. [MSD] 将 MSD 数据集转换为 nnUNet v2 原始格式
     [Direct] 创建 symlink/junction + 生成 dataset.json
  3. 运行 nnUNet 数据指纹分析 + 预处理 (plan_and_preprocess)
  4. 打印后续训练命令

Usage:
    # MSD 数据集 (原有方式)
    python scripts/nnunet/setup_nnunet_task.py \
        --msd-dir data/raw/Task03_Liver \
        --dataset-id 3

    # 已有 nnUNet 格式数据 (新增: 如前列腺数据)
    python scripts/nnunet/setup_nnunet_task.py \
        --data-dir D:/Dataset002_ProstateCrop_Seg \
        --dataset-id 2 --dataset-name ProstateCrop_Seg \
        --channel-names T2w ADC DWI \
        --labels background PZ TZ lesion \
        --exp configs/2d/exp/exp_prostate_local.yaml

    # 自定义 nnUNet 数据目录
    python scripts/nnunet/setup_nnunet_task.py \
        --data-dir D:/Dataset002_ProstateCrop_Seg \
        --dataset-id 2 --dataset-name ProstateCrop_Seg \
        --channel-names T2w ADC DWI \
        --labels background PZ TZ lesion \
        --nnunet-base nnunet_data

Prerequisites:
    pip install nnunetv2>=2.2
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _set_env(base_dir: Path) -> None:
    """Set nnUNet environment variables."""
    raw_dir = base_dir / "nnUNet_raw"
    preprocessed_dir = base_dir / "nnUNet_preprocessed"
    results_dir = base_dir / "nnUNet_results"

    raw_dir.mkdir(parents=True, exist_ok=True)
    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    os.environ["nnUNet_raw"] = str(raw_dir)
    os.environ["nnUNet_preprocessed"] = str(preprocessed_dir)
    os.environ["nnUNet_results"] = str(results_dir)

    print(f"nnUNet_raw          = {raw_dir}")
    print(f"nnUNet_preprocessed = {preprocessed_dir}")
    print(f"nnUNet_results      = {results_dir}")


def _check_nnunet_installed() -> bool:
    """Check if nnUNet v2 is installed."""
    try:
        import nnunetv2
        version = getattr(nnunetv2, "__version__", "unknown")
        print(f"nnUNet v2 version: {version}")
        return True
    except ImportError:
        return False


def _create_dataset_json(
    data_dir: Path,
    channel_names: list[str],
    label_names: list[str],
) -> dict:
    """Create nnUNet v2 dataset.json for existing nnUNet-format data."""
    import json

    # Count training cases (unique case IDs in imagesTr)
    images_tr = data_dir / "imagesTr"
    if not images_tr.exists():
        raise FileNotFoundError(f"imagesTr not found: {images_tr}")

    # Find unique case IDs by stripping _XXXX.nii.gz suffix
    case_ids = set()
    for f in images_tr.iterdir():
        name = f.name
        if name.endswith(".nii.gz"):
            # Remove _XXXX.nii.gz to get case ID
            stem = name[: -len(".nii.gz")]
            # Remove last _XXXX (4 digits)
            parts = stem.rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit() and len(parts[1]) == 4:
                case_ids.add(parts[0])

    dataset_json = {
        "channel_names": {str(i): name for i, name in enumerate(channel_names)},
        "labels": {name: i for i, name in enumerate(label_names)},
        "numTraining": len(case_ids),
        "file_ending": ".nii.gz",
    }

    json_path = data_dir / "dataset.json"
    with open(json_path, "w") as f:
        json.dump(dataset_json, f, indent=2, ensure_ascii=False)
    print(f"✅ Created dataset.json: {json_path}")
    print(f"   Channels: {channel_names}")
    print(f"   Labels:   {label_names}")
    print(f"   Training cases: {len(case_ids)}")
    return dataset_json


def main() -> None:
    ap = argparse.ArgumentParser(description="Setup nnUNet v2 for MSD or nnUNet-format data")

    # Data source (mutually exclusive)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--msd-dir", help="Path to MSD task directory (e.g., data/raw/Task03_Liver)")
    src.add_argument("--data-dir", help="Path to existing nnUNet-format data directory")

    ap.add_argument("--dataset-id", type=int, required=True, help="nnUNet dataset ID (e.g., 2)")
    ap.add_argument("--dataset-name", default=None, help="Dataset name (for --data-dir mode)")
    ap.add_argument("--nnunet-base", default="nnunet_data", help="Base directory for nnUNet data (default: nnunet_data)")
    ap.add_argument("--skip-convert", action="store_true", help="Skip MSD conversion (if already converted)")
    ap.add_argument("--overwrite", action="store_true", help="Remove existing dataset dirs before conversion")
    ap.add_argument("--skip-preprocess", action="store_true", help="Skip preprocessing")
    ap.add_argument("--verify", action="store_true", help="Verify dataset integrity")

    # For --data-dir mode: metadata
    ap.add_argument("--channel-names", nargs="+", default=None, help="Channel names (e.g., T2w ADC DWI)")
    ap.add_argument("--labels", nargs="+", default=None, help="Label names in order (e.g., background PZ TZ lesion)")
    ap.add_argument("--exp", default=None, help="Experiment YAML (for print commands)")
    args = ap.parse_args()

    base_dir = Path(args.nnunet_base).resolve()
    is_msd = args.msd_dir is not None

    # Step 0: Check nnUNet installation
    if not _check_nnunet_installed():
        print("\n❌ nnUNet v2 not installed. Install with:")
        print("   pip install nnunetv2>=2.2")
        sys.exit(1)

    if is_msd:
        msd_dir = Path(args.msd_dir).resolve()
        if not msd_dir.exists():
            print(f"\n❌ MSD directory not found: {msd_dir}")
            sys.exit(1)
    else:
        data_dir = Path(args.data_dir).resolve()
        if not data_dir.exists():
            print(f"\n❌ Data directory not found: {data_dir}")
            sys.exit(1)

    # Step 1: Set environment variables
    print("\n" + "=" * 60)
    print("Step 1: Setting nnUNet environment variables")
    print("=" * 60)
    _set_env(base_dir)

    # Step 2: Prepare data
    if is_msd:
        # ── MSD mode: convert MSD → nnUNet format ──
        if not args.skip_convert:
            print("\n" + "=" * 60)
            print("Step 2: Converting MSD dataset to nnUNet format")
            print("=" * 60)

            # Clean up existing dataset directories for this ID to avoid conflict
            import shutil
            ds_prefix = f"Dataset{args.dataset_id:03d}"
            for parent in ["nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"]:
                parent_dir = base_dir / parent
                if parent_dir.exists():
                    for d in parent_dir.iterdir():
                        if d.name.startswith(ds_prefix):
                            if args.overwrite:
                                print(f"  Removing existing: {d}")
                                shutil.rmtree(d, ignore_errors=True)
                            else:
                                print(f"⚠️  Found existing dataset: {d}")
                                print(f"   Use --overwrite to remove it, or --skip-convert to skip conversion.")
                                sys.exit(1)

            try:
                from nnunetv2.dataset_conversion.convert_MSD_dataset import convert_msd_dataset
                print(f"Converting: {msd_dir}  →  {ds_prefix}")
                convert_msd_dataset(str(msd_dir), overwrite_target_id=args.dataset_id)
            except Exception as e:
                print(f"❌ MSD conversion failed: {e}")
                sys.exit(1)

            # Clean up stale *_COMPUTING_* temp folders left by convert_msd_dataset
            raw_dir = base_dir / "nnUNet_raw"
            for d in raw_dir.iterdir():
                if "COMPUTING" in d.name and d.is_dir():
                    print(f"  Removing stale temp folder: {d}")
                    shutil.rmtree(d, ignore_errors=True)

            print("✅ MSD conversion complete")
        else:
            print("\n[Skip] MSD conversion")

    else:
        # ── Direct mode: symlink/junction + dataset.json ──
        print("\n" + "=" * 60)
        print("Step 2: Setting up nnUNet-format data (direct mode)")
        print("=" * 60)

        ds_name = args.dataset_name or data_dir.name
        target_name = f"Dataset{args.dataset_id:03d}_{ds_name}"
        raw_dir = base_dir / "nnUNet_raw"
        target_dir = raw_dir / target_name

        if target_dir.exists():
            if args.overwrite:
                import shutil
                print(f"  Removing existing: {target_dir}")
                shutil.rmtree(target_dir, ignore_errors=True)
            elif target_dir.is_symlink() or target_dir.is_junction():
                print(f"  Symlink already exists: {target_dir}")
            else:
                print(f"⚠️  Found existing dataset: {target_dir}")
                print(f"   Use --overwrite to recreate.")
                sys.exit(1)

        if not target_dir.exists():
            # Create junction (Windows) or symlink (Linux)
            try:
                if sys.platform == "win32":
                    # Use junction on Windows (doesn't require admin privileges)
                    subprocess.run(
                        ["cmd", "/c", "mklink", "/J", str(target_dir), str(data_dir)],
                        check=True, capture_output=True,
                    )
                else:
                    target_dir.symlink_to(data_dir)
                print(f"✅ Created junction: {target_dir} → {data_dir}")
            except Exception as e:
                print(f"❌ Failed to create junction: {e}")
                print(f"   Try copying data manually to: {target_dir}")
                sys.exit(1)

        # Create dataset.json if not exists
        dj_path = data_dir / "dataset.json"
        if not dj_path.exists():
            if not args.channel_names or not args.labels:
                print("❌ --channel-names and --labels required for first-time setup")
                print("   Example: --channel-names T2w ADC DWI --labels background PZ TZ lesion")
                sys.exit(1)
            _create_dataset_json(data_dir, args.channel_names, args.labels)
        else:
            import json
            with open(dj_path) as f:
                dj = json.load(f)
            print(f"  dataset.json already exists: {len(dj.get('channel_names', {}))} channels, "
                  f"{dj.get('numTraining', '?')} training cases")

    # Step 3: Verify dataset (optional)
    if args.verify:
        print("\n" + "=" * 60)
        print("Step 2.5: Verifying dataset integrity")
        print("=" * 60)

        # Clean up stale *_COMPUTING_* temp folders
        import shutil as _shutil
        raw_dir = base_dir / "nnUNet_raw"
        for d in raw_dir.iterdir():
            if "_COMPUTING_" in d.name and d.is_dir():
                print(f"  Removing stale temp folder: {d}")
                _shutil.rmtree(d, ignore_errors=True)

        cmd = [
            sys.executable, "-m", "nnunetv2.experiment_planning.verify_dataset_integrity",
            "-d", str(args.dataset_id),
        ]
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, capture_output=False)

    # Step 4: Plan and preprocess
    if not args.skip_preprocess:
        print("\n" + "=" * 60)
        print("Step 3: Planning and preprocessing (fingerprint → plans → preprocess)")
        print("=" * 60)

        cmd = [
            sys.executable, "-m", "nnunetv2.experiment_planning.plan_and_preprocess_entrypoints",
            "-d", str(args.dataset_id),
            "--verify_dataset_integrity",
        ]
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=False)
        if result.returncode != 0:
            print("❌ Plan and preprocess failed")
            sys.exit(1)
        print("✅ Planning and preprocessing complete")
    else:
        print("\n[Skip] Planning and preprocessing")

    # Print training commands
    dataset_name = f"Dataset{args.dataset_id:03d}"
    plans_dir = base_dir / "nnUNet_preprocessed" / dataset_name
    if plans_dir.exists():
        # Try to find the actual dataset name
        for d in (base_dir / "nnUNet_raw").iterdir():
            if d.name.startswith(f"Dataset{args.dataset_id:03d}"):
                dataset_name = d.name
                break

    exp_yaml = args.exp or f"configs/2d/exp/exp_prostate_local.yaml"

    print("\n" + "=" * 60)
    print("✅ Setup complete! Next steps:")
    print("=" * 60)
    print()
    print("Before running training, set environment variables:")
    print(f'  $env:nnUNet_raw = "{base_dir / "nnUNet_raw"}"')
    print(f'  $env:nnUNet_preprocessed = "{base_dir / "nnUNet_preprocessed"}"')
    print(f'  $env:nnUNet_results = "{base_dir / "nnUNet_results"}"')
    print()
    print("# ---- Train nnUNet 2D (all 5 folds) ----")
    print("# 每折 1000 epochs, SGD + PolyLR, 官方配置")
    for fold in range(5):
        print(f"  nnUNetv2_train {args.dataset_id} 2d {fold} --npz")
    print()
    print("# 或者单折测试:")
    print(f"  nnUNetv2_train {args.dataset_id} 2d 0 --npz")
    print()
    print("# ---- 训练完成后导入权重到 Seg-MoE ----")
    print(f"  python scripts/nnunet/import_nnunet_weights.py \\")
    print(f"    --nnunet-base {args.nnunet_base} \\")
    print(f"    --dataset-id {args.dataset_id} \\")
    print(f"    --exp {exp_yaml} \\")
    print(f"    --config 2d --folds 0 1 2 3 4")


if __name__ == "__main__":
    main()

"""
nnUNet 测试集独立评估脚本
==============================
功能:
  1. 调用 nnUNetPredictor API 对 imagesTs 做 5-fold 集成推理 (保存概率图)
  2. 对每个 case 计算:
       DSC, IoU, HD95, NSD, Sensitivity, Precision, AUC-ROC
  3. 输出 per-case CSV 和 summary CSV

用法:
    conda activate segmoe
    python scripts/eval/eval_nnunet_test.py [options]

典型调用 (默认路径已硬编码可直接运行):
    python scripts/eval/eval_nnunet_test.py

or 指定路径:
    python scripts/eval/eval_nnunet_test.py \\
        --images_dir   E:/nnunetv2_WebUI/nnUNet_raw/Dataset002_ProstateCrop_seg/imagesTs \\
        --labels_dir   E:/nnunetv2_WebUI/nnUNet_raw/Dataset002_ProstateCrop_seg/labelsTs \\
        --output_dir   D:/Seg-MoE/runs/nnunet_test_predictions \\
        --nnunet_results_dir  D:/Seg-MoE/nnunet_data/nnUNet_results \\
        --dataset_id   Dataset002_ProstateCrop_Seg \\
        --trainer      nnUNetTrainer \\
        --config       2d \\
        --plans        nnUNetPlans \\
        --folds        0 1 2 3 4 \\
        --skip_predict          # 若推理已完成跳过此步骤
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import binary_erosion
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────
# 默认路径 (可通过命令行覆盖)
# ─────────────────────────────────────────────────────────────────────
DEFAULT_IMAGES_DIR   = r"E:\nnunetv2_WebUI\nnUNet_raw\Dataset002_ProstateCrop_seg\imagesTs"
DEFAULT_LABELS_DIR   = r"E:\nnunetv2_WebUI\nnUNet_raw\Dataset002_ProstateCrop_seg\labelsTs"
DEFAULT_OUTPUT_DIR   = r"D:\Seg-MoE\runs\nnunet_test_predictions"
DEFAULT_RESULTS_DIR  = r"D:\Seg-MoE\nnunet_data\nnUNet_results"
DEFAULT_PREPROC_DIR  = r"D:\Seg-MoE\nnunet_data\nnUNet_preprocessed"
DEFAULT_RAW_DIR      = r"D:\Seg-MoE\nnunet_data\nnUNet_raw"
DEFAULT_DATASET_ID   = "Dataset002_ProstateCrop_Seg"
DEFAULT_TRAINER      = "nnUNetTrainer"
DEFAULT_CONFIG       = "2d"
DEFAULT_PLANS        = "nnUNetPlans"
DEFAULT_FOLDS        = [0, 1, 2, 3, 4]


# ─────────────────────────────────────────────────────────────────────
# 指标函数
# ─────────────────────────────────────────────────────────────────────

def dice_score(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> float:
    tp = np.sum(pred & gt)
    fp = np.sum(pred & ~gt)
    fn = np.sum(~pred & gt)
    return float((2 * tp + eps) / (2 * tp + fp + fn + eps))


def iou_score(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> float:
    tp = np.sum(pred & gt)
    fp = np.sum(pred & ~gt)
    fn = np.sum(~pred & gt)
    return float((tp + eps) / (tp + fp + fn + eps))


def sensitivity_precision(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7):
    tp = np.sum(pred & gt)
    fp = np.sum(pred & ~gt)
    fn = np.sum(~pred & gt)
    sens = float((tp + eps) / (tp + fn + eps))
    prec = float((tp + eps) / (tp + fp + eps))
    return sens, prec


def hausdorff_distance_95(pred: np.ndarray, gt: np.ndarray,
                           spacing: tuple | None = None) -> float:
    """HD95: 95th percentile of symmetric surface distance."""
    try:
        import cc3d
    except ImportError:
        cc3d = None

    if not pred.any() or not gt.any():
        return float("nan")

    pred_border = pred ^ binary_erosion(pred)
    gt_border   = gt   ^ binary_erosion(gt)

    if spacing is None:
        spacing = (1.0,) * pred.ndim

    # 生成坐标点 (voxel → physical)
    pred_pts = np.column_stack(
        [c * s for c, s in zip(np.where(pred_border), spacing)]
    )
    gt_pts   = np.column_stack(
        [c * s for c, s in zip(np.where(gt_border), spacing)]
    )

    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return float("nan")

    # 使用 scipy cdist 计算最近邻距离
    from scipy.spatial.distance import cdist
    d_p2g = cdist(pred_pts, gt_pts).min(axis=1)
    d_g2p = cdist(gt_pts, pred_pts).min(axis=1)
    all_d = np.concatenate([d_p2g, d_g2p])
    return float(np.percentile(all_d, 95))


def nsd_score(pred: np.ndarray, gt: np.ndarray,
              spacing: tuple | None = None, tau: float = 2.0) -> float:
    """NSD: Normalized Surface Distance at tolerance tau mm."""
    if not pred.any() or not gt.any():
        return float("nan")
    if spacing is None:
        spacing = (1.0,) * pred.ndim

    pred_border = pred ^ binary_erosion(pred)
    gt_border   = gt   ^ binary_erosion(gt)

    pred_pts = np.column_stack(
        [c * s for c, s in zip(np.where(pred_border), spacing)]
    )
    gt_pts   = np.column_stack(
        [c * s for c, s in zip(np.where(gt_border), spacing)]
    )

    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return float("nan")

    from scipy.spatial.distance import cdist
    d_p2g = cdist(pred_pts, gt_pts).min(axis=1)
    d_g2p = cdist(gt_pts, pred_pts).min(axis=1)

    nsd = (np.sum(d_p2g <= tau) + np.sum(d_g2p <= tau)) / \
          (len(pred_pts) + len(gt_pts))
    return float(nsd)


def auc_roc_score(prob_fg: np.ndarray, gt: np.ndarray) -> float:
    """体素级 AUC-ROC.
    prob_fg: foreground 概率图 (3D or 2D flat), gt: binary label
    跳过全0或全1的 case (AUC无意义).
    """
    y_score = prob_fg.ravel().astype(np.float32)
    y_true  = gt.ravel().astype(np.int32)
    unique = np.unique(y_true)
    if len(unique) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


# ─────────────────────────────────────────────────────────────────────
# Case ID 提取
# ─────────────────────────────────────────────────────────────────────

def get_case_ids_from_labels(labels_dir: Path) -> list[str]:
    """从 labelsTs 推断 case id 列表."""
    ids = []
    for f in sorted(labels_dir.glob("*.nii.gz")):
        ids.append(f.name.replace(".nii.gz", ""))
    return ids


# ─────────────────────────────────────────────────────────────────────
# 步骤 1: nnUNet 推理 (使用 Python API)
# ─────────────────────────────────────────────────────────────────────

def run_nnunet_predict(
    images_dir: Path,
    output_dir: Path,
    results_dir: Path,
    preproc_dir: Path,
    raw_dir: Path,
    dataset_id: str,
    trainer: str,
    config: str,
    plans: str,
    folds: list[int],
    device: str = "cuda",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # 设置 nnUNet 环境变量 (必须在 import 之前设置)
    os.environ["nnUNet_raw"]          = str(raw_dir)
    os.environ["nnUNet_preprocessed"] = str(preproc_dir)
    os.environ["nnUNet_results"]      = str(results_dir)

    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    # 模型目录: {results}/{dataset_id}/{trainer}__{plans}__{config}
    model_dir = results_dir / dataset_id / f"{trainer}__{plans}__{config}"
    print(f"\n模型目录: {model_dir}")
    assert model_dir.exists(), f"模型目录不存在: {model_dir}"

    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"推理设备: {device_obj}")

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,     # 关闭 TTA 加速推理
        perform_everything_on_device=True,
        device=device_obj,
        verbose=False,
        allow_tqdm=True,
    )

    predictor.initialize_from_trained_model_folder(
        model_training_output_dir=str(model_dir),
        use_folds=tuple(folds),
        checkpoint_name="checkpoint_final.pth",
    )

    print(f"\n开始推理 {images_dir} → {output_dir}")
    print(f"使用 folds: {folds}, 保存概率图: True")
    print("=" * 60)

    predictor.predict_from_files(
        list_of_lists_or_source_folder=str(images_dir),
        output_folder_or_list_of_truncated_output_files=str(output_dir),
        save_probabilities=True,
        overwrite=False,                  # 已存在则跳过
        num_processes_preprocessing=2,
        num_processes_segmentation_export=2,
    )
    print("[完成] nnUNet 推理结束")


# ─────────────────────────────────────────────────────────────────────
# 步骤 2: 计算指标
# ─────────────────────────────────────────────────────────────────────

def evaluate_predictions(
    pred_dir: Path,
    labels_dir: Path,
    num_classes: int = 2,
) -> pd.DataFrame:
    rows = []
    case_ids = get_case_ids_from_labels(labels_dir)
    print(f"\n共 {len(case_ids)} 个 case 需要评估")

    for cid in tqdm(case_ids, desc="评估"):
        pred_path  = pred_dir   / f"{cid}.nii.gz"
        label_path = labels_dir / f"{cid}.nii.gz"
        prob_path  = pred_dir   / f"{cid}.npz"

        if not pred_path.exists():
            print(f"  [跳过] 预测文件不存在: {pred_path}")
            continue
        if not label_path.exists():
            print(f"  [跳过] 标签文件不存在: {label_path}")
            continue

        # 读取预测和标签 (SimpleITK)
        pred_img  = sitk.ReadImage(str(pred_path))
        label_img = sitk.ReadImage(str(label_path))

        pred_arr  = sitk.GetArrayFromImage(pred_img).astype(np.int32)
        label_arr = sitk.GetArrayFromImage(label_img).astype(np.int32)

        # 获取 physical spacing (z, y, x) → 转换为 mm
        spacing = pred_img.GetSpacing()  # (x, y, z) in SimpleITK
        spacing_zyx = (spacing[2], spacing[1], spacing[0])

        # 概率图 (foreground class = class 1)
        prob_fg = None
        if prob_path.exists():
            npz = np.load(str(prob_path))
            # nnUNetv2 保存格式: 'probabilities' shape=(C, D, H, W) 或 (C, H, W)
            key = "probabilities" if "probabilities" in npz else list(npz.keys())[0]
            probs = npz[key]   # shape: (C, ...)
            # foreground class index = 1
            fg_idx = min(1, probs.shape[0] - 1)
            prob_fg = probs[fg_idx]

        row: dict = {"case_id": cid}

        # 对每个前景类计算指标 (二分类只有 class 1)
        for cls in range(1, num_classes):
            pred_bin  = (pred_arr  == cls)
            label_bin = (label_arr == cls)

            row[f"DSC_cls{cls}"]         = dice_score(pred_bin, label_bin)
            row[f"IoU_cls{cls}"]         = iou_score(pred_bin, label_bin)
            sens, prec                   = sensitivity_precision(pred_bin, label_bin)
            row[f"Sensitivity_cls{cls}"] = sens
            row[f"Precision_cls{cls}"]   = prec

            # HD95 & NSD (计算较慢, 使用 physical spacing)
            row[f"HD95_cls{cls}"]  = hausdorff_distance_95(pred_bin, label_bin, spacing_zyx)
            row[f"NSD_cls{cls}"]   = nsd_score(pred_bin, label_bin, spacing_zyx, tau=2.0)

            # AUC-ROC (需要概率图)
            if prob_fg is not None:
                row[f"AUC_ROC_cls{cls}"] = auc_roc_score(prob_fg, label_bin)
            else:
                row[f"AUC_ROC_cls{cls}"] = float("nan")

        rows.append(row)

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="nnUNet 测试集评估 (DSC/HD95/AUC-ROC)")
    p.add_argument("--images_dir",        default=DEFAULT_IMAGES_DIR)
    p.add_argument("--labels_dir",        default=DEFAULT_LABELS_DIR)
    p.add_argument("--output_dir",        default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--nnunet_results_dir",default=DEFAULT_RESULTS_DIR)
    p.add_argument("--nnunet_preproc_dir",default=DEFAULT_PREPROC_DIR)
    p.add_argument("--nnunet_raw_dir",    default=DEFAULT_RAW_DIR)
    p.add_argument("--dataset_id",        default=DEFAULT_DATASET_ID)
    p.add_argument("--trainer",           default=DEFAULT_TRAINER)
    p.add_argument("--config",            default=DEFAULT_CONFIG)
    p.add_argument("--plans",             default=DEFAULT_PLANS)
    p.add_argument("--folds",             nargs="+", type=int, default=DEFAULT_FOLDS)
    p.add_argument("--num_classes",       type=int, default=2,
                   help="包含背景的类别总数 (二分类 prostate → 2)")
    p.add_argument("--device",            default="cuda",
                   choices=["cuda", "cpu", "mps"])
    p.add_argument("--skip_predict",      action="store_true",
                   help="跳过推理步骤 (预测结果已存在)")
    p.add_argument("--report_dir",        default=None,
                   help="指标 CSV 保存目录 (默认与 output_dir 相同)")
    return p.parse_args()


def main():
    args = parse_args()

    images_dir   = Path(args.images_dir)
    labels_dir   = Path(args.labels_dir)
    output_dir   = Path(args.output_dir)
    results_dir  = Path(args.nnunet_results_dir)
    preproc_dir  = Path(args.nnunet_preproc_dir)
    raw_dir      = Path(args.nnunet_raw_dir)
    report_dir   = Path(args.report_dir) if args.report_dir else output_dir

    report_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("nnUNet 测试集评估")
    print(f"  images_dir  : {images_dir}")
    print(f"  labels_dir  : {labels_dir}")
    print(f"  output_dir  : {output_dir}")
    print(f"  folds       : {args.folds}")
    print(f"  skip_predict: {args.skip_predict}")
    print("=" * 60)

    # ── Step 1: 推理 ──────────────────────────────────────────────
    if not args.skip_predict:
        run_nnunet_predict(
            images_dir  = images_dir,
            output_dir  = output_dir,
            results_dir = results_dir,
            preproc_dir = preproc_dir,
            raw_dir     = raw_dir,
            dataset_id  = args.dataset_id,
            trainer     = args.trainer,
            config      = args.config,
            plans       = args.plans,
            folds       = args.folds,
            device      = args.device,
        )
    else:
        print("[跳过推理] 使用已有预测结果")

    # ── Step 2: 评估 ──────────────────────────────────────────────
    df = evaluate_predictions(
        pred_dir    = output_dir,
        labels_dir  = labels_dir,
        num_classes = args.num_classes,
    )

    if df.empty:
        print("[错误] 没有评估结果, 请检查预测文件是否存在.")
        return

    # ── Step 3: 输出 ──────────────────────────────────────────────
    per_case_path = report_dir / "nnunet_test_metrics_per_case.csv"
    df.to_csv(per_case_path, index=False)
    print(f"\n[保存] Per-case 指标 → {per_case_path}")

    # 汇总统计
    metric_cols = [c for c in df.columns if c != "case_id"]
    summary_rows = []
    for col in metric_cols:
        vals = df[col].dropna().values
        if len(vals) == 0:
            continue
        summary_rows.append({
            "metric"  : col,
            "mean"    : float(np.mean(vals)),
            "std"     : float(np.std(vals)),
            "median"  : float(np.median(vals)),
            "min"     : float(np.min(vals)),
            "max"     : float(np.max(vals)),
            "n_valid" : int(len(vals)),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = report_dir / "nnunet_test_metrics_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"[保存] Summary 指标 → {summary_path}")

    # 打印到控制台
    print("\n" + "=" * 60)
    print("  nnUNet 测试集评估结果 (均值 ± 标准差):")
    print("=" * 60)
    for _, r in summary_df.iterrows():
        print(f"  {r['metric']:<24s}: {r['mean']:.4f} ± {r['std']:.4f}  "
              f"[median={r['median']:.4f}, n={r['n_valid']}]")
    print("=" * 60)

    # AUC-ROC 曲线 (保存图片, 如果 matplotlib 可用)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        roc_col = "AUC_ROC_cls1"
        if roc_col in df.columns:
            # 如果有概率图, 重新汇总计算全局 AUC
            prob_all, label_all = [], []
            case_ids = get_case_ids_from_labels(labels_dir)
            for cid in case_ids:
                prob_path  = output_dir / f"{cid}.npz"
                label_path = labels_dir / f"{cid}.nii.gz"
                if not prob_path.exists() or not label_path.exists():
                    continue
                label_img = sitk.ReadImage(str(label_path))
                label_arr = sitk.GetArrayFromImage(label_img).astype(np.int32)
                npz = np.load(str(prob_path))
                key = "probabilities" if "probabilities" in npz else list(npz.keys())[0]
                probs  = npz[key]
                prob_fg = probs[min(1, probs.shape[0]-1)]
                prob_all.append(prob_fg.ravel())
                label_all.append((label_arr == 1).ravel().astype(np.int32))

            if prob_all:
                y_score = np.concatenate(prob_all).astype(np.float32)
                y_true  = np.concatenate(label_all)
                if len(np.unique(y_true)) == 2:
                    fpr, tpr, _ = roc_curve(y_true, y_score)
                    auc_global  = roc_auc_score(y_true, y_score)

                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc_global:.4f}")
                    ax.plot([0,1],[0,1],"k--", lw=1)
                    ax.set_xlabel("False Positive Rate")
                    ax.set_ylabel("True Positive Rate")
                    ax.set_title("nnUNet ROC Curve (全局体素级)")
                    ax.legend(loc="lower right")
                    roc_path = report_dir / "nnunet_test_roc_curve.png"
                    fig.savefig(str(roc_path), dpi=150, bbox_inches="tight")
                    plt.close(fig)
                    print(f"[保存] ROC 曲线图 → {roc_path}")
                    print(f"       全局 AUC-ROC = {auc_global:.4f}")

    except ImportError:
        print("[提示] matplotlib 未安装, 跳过 ROC 曲线绘制")

    print("\n评估完成!")


if __name__ == "__main__":
    main()

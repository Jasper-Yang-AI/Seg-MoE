# Seg-MoE 训练手册

Two-Layer Ensemble 管线：**Layer1（三专家官方训练）→ Layer1 OOF → Layer2 → Layer2 OOF → Gating → 评估**  
三专家：nnUNet · SwinUNETR · SegResNet

---

## 0. 安装

```powershell
conda create -n segmoe python=3.10 -y
conda activate segmoe
pip install -r requirements.txt
pip install -e .
pip install "nnunetv2>=2.2"
```

---

## 1. 数据集与配置对照表

| 数据集 | 维度 | 类别 | 通道 | 数据集配置 | 实验配置 | 模型配置 |
|--------|------|------|------|-----------|---------|---------|
| Prostate（本地） | 2D PNG | 4 (bg/PZ/TZ/lesion) | 3 | `configs/2d/datasets/prostate_local.yaml` | `configs/2d/exp/exp_prostate_local.yaml` | `configs/2d/models.yaml` |
| Prostate（本地） | 3D NIfTI | 4 | 3 | `configs/3d/datasets/prostate_local_3d.yaml` | `configs/3d/exp/exp_prostate_local_3d.yaml` | `configs/3d/models_3d.yaml` |
| Liver（Dataset003） | 3D NIfTI | 2 (bg/tumor) | 1 | `configs/3d/datasets/liver_3d.yaml` | `configs/3d/exp/exp_liver_3d.yaml` | `configs/3d/models_3d.yaml` |

**nnUNet 环境变量**（每个新终端执行一次）：

```powershell
$env:nnUNet_raw          = "D:\Seg-MoE\nnunet_data\nnUNet_raw"
$env:nnUNet_preprocessed = "D:\Seg-MoE\nnunet_data\nnUNet_preprocessed"
$env:nnUNet_results      = "D:\Seg-MoE\nnunet_data\nnUNet_results"
```

---

## 2. 2D 前列腺实验

**执行顺序**：数据准备 → nnUNet+SwinUNETR+SegResNet 并行训练 → 导入权重 → Layer1 OOF → Layer2 → Layer2 OOF → Gating → 评估

**入口说明（重要）**：
- **Layer1 正式训练** 使用官方入口：`nnUNetv2_train`、`scripts/monai/train_swinunetr_official.py`、`scripts/monai/train_segresnet_official.py`。
- **Layer2 训练** 使用工程统一入口：`scripts/train/train_layer2.py` + `configs/2d/training_layer2.yaml`。

### Step 1：数据准备（仅需一次）

```powershell
python scripts/data/prepare_prostate.py --config configs/2d/datasets/prostate_local.yaml
python scripts/data/make_splits.py --dataset-config configs/2d/datasets/prostate_local.yaml
```

### Step 2：nnUNet 2D（5 折）

```powershell
python scripts/nnunet/setup_nnunet_task.py `
  --data-dir E:/nnunetv2_WebUI/nnUNet_raw/Dataset002_ProstateCrop_seg `
  --dataset-id 2 --dataset-name ProstateCrop_Seg `
  --channel-names T2w ADC DWI --labels background PZ TZ lesion `
  --exp configs/2d/exp/exp_prostate_local.yaml --verify

foreach ($fold in 0..4) { nnUNetv2_train 2 2d $fold --npz }   # 续训加 --c

python scripts/nnunet/import_nnunet_weights.py `
  --nnunet-base nnunet_data --dataset-id 2 --config 2d `
  --folds 0 1 2 3 4 --exp configs/2d/exp/exp_prostate_local.yaml `
  --models configs/2d/models.yaml --expert-name nnunet-2d `
  --update-models-yaml configs/2d/models.yaml
```

### Step 3：SwinUNETR 2D（5 折）

```powershell
foreach ($fold in 0..4) {
  python scripts/monai/train_swinunetr_official.py `
    --exp configs/2d/exp/exp_prostate_local.yaml --models configs/2d/models.yaml `
    --fold $fold --gpus 0,1 --epochs 300 --batch-size 16 `
    --amp --amp-dtype bfloat16 --num-workers 2
    # 续训加: --resume "runs/swinunetr_official_prostate_local/fold$fold/latest_model.pt"
}
foreach ($fold in 0..4) {
  python scripts/monai/import_swinunetr_weights.py `
    --source runs/swinunetr_official_prostate_local/fold$fold/best_model.pt `
    --exp configs/2d/exp/exp_prostate_local.yaml --models configs/2d/models.yaml `
    --fold $fold --expert-name swinunetr-2d
}
```

### Step 4：SegResNet 2D（5 折）

```powershell
foreach ($fold in 0..4) {
  python scripts/monai/train_segresnet_official.py `
    --exp configs/2d/exp/exp_prostate_local.yaml --models configs/2d/models.yaml `
    --fold $fold --gpus 0,1 --epochs 300 --batch-size 32 --dsdepth 2 `
    --amp --amp-dtype bfloat16 --num-workers 2
    # 续训加: --resume "runs/segresnet_official_prostate_local/fold$fold/latest_model.pt"
}
foreach ($fold in 0..4) {
  python scripts/monai/import_segresnet_weights.py `
    --source runs/segresnet_official_prostate_local/fold$fold/best_model.pt `
    --exp configs/2d/exp/exp_prostate_local.yaml --models configs/2d/models.yaml `
    --fold $fold --expert-name segresnet-2d
}
```

### Step 5：Layer1 OOF → Layer2 → Layer2 OOF → Gating → 评估

> 说明：Step 5 开始，Layer1 使用的是 Step 2/3/4 导入后的官方专家权重；Layer2 由统一训练器 `train_layer2.py` 按工程配置训练。

```powershell
python scripts/inference/generate_layer1_oof.py `
  --exp configs/2d/exp/exp_prostate_local.yaml --models configs/2d/models.yaml `
  --which best --batch-size 32 --tta

foreach ($fold in 0..4) {
  python scripts/train/train_layer2.py `
    --exp configs/2d/exp/exp_prostate_local.yaml --training configs/2d/training_layer2.yaml `
    --models configs/2d/models.yaml --augs configs/2d/augs.yaml --fold $fold --gpus 0,1
}

python scripts/inference/generate_layer2_oof.py `
  --exp configs/2d/exp/exp_prostate_local.yaml --models configs/2d/models.yaml `
  --which best --batch-size 32 --tta

foreach ($fold in 0..4) {
  python scripts/train/train_gating.py `
    --exp configs/2d/exp/exp_prostate_local.yaml --gating-config configs/2d/gating.yaml `
    --models configs/2d/models.yaml --fold $fold --gpus 0,1
}

python scripts/inference/gating_inference.py `
  --exp configs/2d/exp/exp_prostate_local.yaml --gating-config configs/2d/gating.yaml `
  --models configs/2d/models.yaml --fold 0

python scripts/eval/eval_methods.py `
  --exp configs/2d/exp/exp_prostate_local.yaml --training configs/2d/training_layer2.yaml `
  --models configs/2d/models.yaml --fold 0
python scripts/eval/export_tables.py --exp configs/2d/exp/exp_prostate_local.yaml --folds 0
```

**显存参考（双 RTX 5090，BF16）**：nnUNet ~12GB | SwinUNETR 25M bs=16 ~14GB | SegResNet 29M bs=32 ~8GB

---

## 3. OOF 可复核审计与失败病例早筛（2D）

### 3.1 Layer1 / Layer2 OOF 审计（建议每次生成 OOF 后执行）

```powershell
# Layer1 OOF 审计
python scripts/utils/audit_oof_manifest.py `
  --manifest runs/segmoe_2d_prostate/cache/oof/layer1/oof_manifest.jsonl `
  --splits data/splits/prostate_local/splits_train5fold_testfixed.jsonl `
  --out runs/segmoe_2d_prostate/results/oof_audit_layer1.json `
  --check-ckpt-fold

# Layer2 OOF 审计
python scripts/utils/audit_oof_manifest.py `
  --manifest runs/segmoe_2d_prostate/cache/oof/layer2/oof_manifest_layer2.jsonl `
  --splits data/splits/prostate_local/splits_train5fold_testfixed.jsonl `
  --out runs/segmoe_2d_prostate/results/oof_audit_layer2.json `
  --check-ckpt-fold
```

审计脚本会检查：
- `sample_fold / predictor_fold / split(val_foldk)` 是否一致
- `prob_path` 文件是否真实存在
- manifest 覆盖率是否与 splits 中各 fold 的 `val_foldk` 数量一致

通过标准：`is_strict_oof=True` 且 `errors=[]`。

### 3.2 失败病例早筛（uncertainty / disagreement vs Dice）

```powershell
# 快速验证（推荐先跑 fold0）
python scripts/eval/eval_failure_detection_oof.py `
  --manifest runs/segmoe_2d_prostate/cache/oof/layer1/oof_manifest.jsonl `
  --splits data/splits/prostate_local/splits_train5fold_testfixed.jsonl `
  --outdir runs/segmoe_2d_prostate/results/failure_detection_oof_fold0 `
  --sample-fold 0 --max-samples 2000

# 全量评估（全部 folds）
python scripts/eval/eval_failure_detection_oof.py `
  --manifest runs/segmoe_2d_prostate/cache/oof/layer1/oof_manifest.jsonl `
  --splits data/splits/prostate_local/splits_train5fold_testfixed.jsonl `
  --outdir runs/segmoe_2d_prostate/results/failure_detection_oof
```

输出文件：
- `per_sample_scores.csv`：切片级 uncertainty/disagreement 与 Dice
- `per_patient_scores.csv`：病例级聚合（更接近临床分诊）
- `summary.json`：相关性、AUROC、Top-k 风险筛查命中率/召回率

---

## 4. 3D 实验（Prostate / Liver 通用流程）

以下以 **Liver** 配置为例，Prostate 3D 只需替换变量：

```powershell
# Liver 3D（当前数据集）
$EXP="configs/3d/exp/exp_liver_3d.yaml"; $MODELS="configs/3d/models_3d.yaml"; $DS_ID=3

# Prostate 3D（切换时用这行）
# $EXP="configs/3d/exp/exp_prostate_local_3d.yaml"; $MODELS="configs/3d/models_3d.yaml"; $DS_ID=2
```

**执行顺序**：smoke test → 数据切分 → nnUNet Task → nnUNet训练 → Swin训练 → SegResNet训练 → 导入权重(×3) → Layer1 OOF → Layer2 → Layer2 OOF → Gating → 评估

**入口说明（重要）**：
- **Layer1 正式训练** 使用官方入口：`nnUNetv2_train`、`scripts/monai/train_swinunetr_official_3d.py`、`scripts/monai/train_segresnet_official_3d.py`。
- **Step 1 的 `train_layer1_3d.py --smoke` 仅用于快速环境自检**，不是正式 Layer1 结果来源。
- **Layer2 训练** 使用工程统一入口：`scripts/train/train_layer2_3d.py` + `configs/3d/training_layer2_3d.yaml`。

### Step 1：Smoke Test（5 分钟，验证环境）

> 说明：此步只做管线联通性检查（数据读取、前向、loss、保存），不替代 Step 4/5/6 的官方专家训练。

```powershell
python scripts/train/train_layer1_3d.py `
  --exp $EXP --training configs/3d/training.yaml --models $MODELS `
  --augs configs/3d/augs_3d.yaml --fold 0 --gpus 0 --smoke
```

### Step 2：数据切分（仅需一次）

```powershell
# Liver（自动发现 case，无需 --source-splits）
python scripts/data/make_splits_3d.py --dataset-config configs/3d/datasets/liver_3d.yaml

# Prostate 3D（从 2D splits 衍生）
# python scripts/data/make_splits_3d.py `
#   --dataset-config configs/3d/datasets/prostate_local_3d.yaml `
#   --source-splits data/splits/prostate_local/splits_train5fold_testfixed.jsonl
```

### Step 3：nnUNet Task 初始化（仅需一次）

> 若 `nnUNetv2_plan_and_preprocess --verify_dataset_integrity` 报错：
> `Spacing mismatch between segmentation and corresponding images`，先执行：

> 若报错为 `Shape mismatch`（如 image 是 2D/少切片，seg 是 3D），
> 用 `--quarantine-invalid` 自动隔离坏样本（不参与训练）：

```powershell
python scripts/data/fix_nnunet_spacing_mismatch.py `
  --dataset-root D:/Seg-MoE/nnunet_data/nnUNet_raw/Dataset003_v2_LiverTumorSeg

# 确认输出后再真正修复
python scripts/data/fix_nnunet_spacing_mismatch.py `
  --dataset-root D:/Seg-MoE/nnunet_data/nnUNet_raw/Dataset003_v2_LiverTumorSeg --apply

# 同时隔离 shape/dim 不匹配的无效样本
python scripts/data/fix_nnunet_spacing_mismatch.py `
  --dataset-root D:/Seg-MoE/nnunet_data/nnUNet_raw/Dataset003_v2_LiverTumorSeg `
  --apply --quarantine-invalid
```

```powershell
# Liver
python scripts/nnunet/setup_nnunet_task.py `
  --data-dir D:\Dataset003_v2_LiverTumorSeg --dataset-id 3 `
  --dataset-name v2_LiverTumorSeg `
  --channel-names channel0 --labels background lab1 `
  --nnunet-base nnunet_data --verify

# Prostate 3D（取消注释使用）
# python scripts/nnunet/setup_nnunet_task.py `
#   --data-dir E:/nnunetv2_WebUI/nnUNet_raw/Dataset002_ProstateCrop_seg `
#   --dataset-id 2 --dataset-name ProstateCrop_Seg `
#   --channel-names T2w ADC DWI --labels background PZ TZ lesion `
#   --nnunet-base nnunet_data --verify
```

### Step 4：nnUNet 3D 训练

```powershell
nnUNetv2_plan_and_preprocess -d $DS_ID --verify_dataset_integrity

foreach ($fold in 0..4) { nnUNetv2_train $DS_ID 3d_fullres $fold --npz }  # 续训加 --c

python scripts/nnunet/import_nnunet_weights_3d.py `
  --nnunet-base nnunet_data --dataset-id $DS_ID --config 3d_fullres `
  --folds 0 1 2 3 4 --exp $EXP --models $MODELS `
  --expert-name nnunet-3d --update-models-yaml $MODELS
```

### Step 5：SwinUNETR 3D 训练

保存路径：`runs/swinunetr_official_3d_{dataset_name}/fold{k}/`  
（Liver: `swinunetr_official_3d_liver_3d` · Prostate 3D: `swinunetr_official_3d_prostate_local_3d`）

```powershell
foreach ($fold in 0..4) {
  python scripts/monai/train_swinunetr_official_3d.py `
    --exp $EXP --models $MODELS --fold $fold --gpus 0 `
    --epochs 300 --batch-size 2 --amp --amp-dtype bfloat16 --num-workers 2
    # 续训加: --resume "runs/swinunetr_official_3d_liver_3d/fold$fold/latest_model.pt"
}
foreach ($fold in 0..4) {
  python scripts/monai/import_swinunetr_weights_3d.py `
    --source "runs/swinunetr_official_3d_liver_3d/fold$fold/best_model.pt" `
    --exp $EXP --models $MODELS --fold $fold --expert-name swinunetr-3d
}
```

### Step 6：SegResNet 3D 训练

保存路径：`runs/segresnet_official_3d_{dataset_name}/fold{k}/`

```powershell
foreach ($fold in 0..4) {
  python scripts/monai/train_segresnet_official_3d.py `
    --exp $EXP --models $MODELS --fold $fold --gpus 0 `
    --epochs 300 --batch-size 2 --dsdepth 2 --amp --amp-dtype bfloat16 --num-workers 2
    # 续训加: --resume "runs/segresnet_official_3d_liver_3d/fold$fold/latest_model.pt"
}
foreach ($fold in 0..4) {
  python scripts/monai/import_segresnet_weights_3d.py `
    --source "runs/segresnet_official_3d_liver_3d/fold$fold/best_model.pt" `
    --exp $EXP --models $MODELS --fold $fold --expert-name segresnet-3d
}
```

### Step 7：Layer1 OOF → Layer2 → Layer2 OOF → Gating → 评估

> 说明：Step 7 开始，Layer1 使用的是 Step 4/5/6 导入后的官方专家权重；Layer2 由统一训练器 `train_layer2_3d.py` 按工程配置训练。

```powershell
python scripts/inference/generate_layer1_oof_3d.py --exp $EXP --models $MODELS --which best

foreach ($fold in 0..4) {
  python scripts/train/train_layer2_3d.py `
    --exp $EXP --training configs/3d/training_layer2_3d.yaml `
    --models $MODELS --augs configs/3d/augs_3d.yaml --fold $fold --gpus 0
}

python scripts/inference/generate_layer2_oof_3d.py --exp $EXP --models $MODELS --which best

foreach ($fold in 0..4) {
  python scripts/train/train_gating_3d.py `
    --exp $EXP --gating-config configs/3d/gating_3d.yaml --models $MODELS --fold $fold --gpus 0
}

python scripts/eval/eval_3d.py --exp $EXP --models $MODELS --gating-config configs/3d/gating_3d.yaml
```

**TensorBoard**：`tensorboard --logdir runs/ --port 6006`

**显存参考（单 RTX 5090 32GB，BF16，roi=160×160×32）**：  
nnUNet ~14GB | SwinUNETR 62M ~22GB（`use_checkpoint=true`） | SegResNet 15M ~10GB


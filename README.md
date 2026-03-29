# Seg-MoE 2D Prostate 实验指南

本仓库当前只保留 `2D prostate` 主实验说明，用于完成本地三模态前列腺分割实验、复现实验结果与整理论文表格。

当前 README 不再展开：
- 3D 流程
- 其他数据集
- 与主实验无关的扩展说明

主实验管线为：

`nnUNet + SwinUNETR + SegResNet -> Layer1 OOF -> Layer2 -> Layer2 OOF -> Gating -> Evaluation`

## 1. 实验范围

- 任务：前列腺 MRI 2D 分割
- 输入：3 模态 MRI，`T2w / ADC / DWI`
- 类别：4 类，`background / PZ / TZ / lesion`
- 数据集配置：`configs/2d/datasets/prostate_local.yaml`
- 实验配置：`configs/2d/exp/exp_prostate_local.yaml`

## 2. 关键配置文件

| 文件 | 作用 |
| --- | --- |
| `configs/2d/datasets/prostate_local.yaml` | 数据路径、类别数、输入尺寸、切片策略 |
| `configs/2d/exp/exp_prostate_local.yaml` | 实验名、缓存路径、结果输出目录 |
| `configs/2d/models.yaml` | 三个 Layer1 专家模型配置 |
| `configs/2d/training_layer2.yaml` | Layer2 训练配置 |
| `configs/2d/gating_dual_5090.yaml` | 推荐的 gating 训练配置 |

## 3. 安装

```powershell
conda create -n segmoe python=3.10 -y
conda activate segmoe
pip install -r requirements.txt
pip install -e .
pip install "nnunetv2>=2.2"
```

每次开启新终端后，先设置 `nnUNet` 环境变量：

```powershell
$env:nnUNet_raw          = "D:\Seg-MoE\nnunet_data\nnUNet_raw"
$env:nnUNet_preprocessed = "D:\Seg-MoE\nnunet_data\nnUNet_preprocessed"
$env:nnUNet_results      = "D:\Seg-MoE\nnunet_data\nnUNet_results"
```

## 4. 数据准备

`configs/2d/datasets/prostate_local.yaml` 默认读取：

```text
E:/nnunetv2_WebUI/nnUNet_raw/Dataset002_ProstateCrop_seg
├─ imagesTr
├─ labelsTr
├─ imagesTs
└─ labelsTs
```

先将 NIfTI 数据整理为 2D PNG 切片，再生成固定测试集 + 5 折训练划分：

```powershell
python scripts/data/prepare_prostate.py --config configs/2d/datasets/prostate_local.yaml
python scripts/data/make_splits.py --dataset-config configs/2d/datasets/prostate_local.yaml
```

## 5. Layer1：三个基础专家

### 5.1 nnUNet 2D

```powershell
python scripts/nnunet/setup_nnunet_task.py `
  --data-dir E:/nnunetv2_WebUI/nnUNet_raw/Dataset002_ProstateCrop_seg `
  --dataset-id 2 --dataset-name ProstateCrop_Seg `
  --channel-names T2w ADC DWI --labels background PZ TZ lesion `
  --exp configs/2d/exp/exp_prostate_local.yaml --verify

foreach ($fold in 0..4) { nnUNetv2_train 2 2d $fold --npz }

python scripts/nnunet/import_nnunet_weights.py `
  --nnunet-base nnunet_data --dataset-id 2 --config 2d `
  --folds 0 1 2 3 4 --exp configs/2d/exp/exp_prostate_local.yaml `
  --models configs/2d/models.yaml --expert-name nnunet-2d `
  --update-models-yaml configs/2d/models.yaml
```

### 5.2 SwinUNETR 2D

```powershell
foreach ($fold in 0..4) {
  python scripts/monai/train_swinunetr_official.py `
    --exp configs/2d/exp/exp_prostate_local.yaml --models configs/2d/models.yaml `
    --fold $fold --gpus 0,1 --epochs 300 --batch-size 16 `
    --amp --amp-dtype bfloat16 --num-workers 2
}

foreach ($fold in 0..4) {
  python scripts/monai/import_swinunetr_weights.py `
    --source runs/swinunetr_official_prostate_local/fold$fold/best_model.pt `
    --exp configs/2d/exp/exp_prostate_local.yaml --models configs/2d/models.yaml `
    --fold $fold --expert-name swinunetr-2d
}
```

### 5.3 SegResNet 2D

```powershell
foreach ($fold in 0..4) {
  python scripts/monai/train_segresnet_official.py `
    --exp configs/2d/exp/exp_prostate_local.yaml --models configs/2d/models.yaml `
    --fold $fold --gpus 0,1 --epochs 300 --batch-size 32 --dsdepth 2 `
    --amp --amp-dtype bfloat16 --num-workers 2
}

foreach ($fold in 0..4) {
  python scripts/monai/import_segresnet_weights.py `
    --source runs/segresnet_official_prostate_local/fold$fold/best_model.pt `
    --exp configs/2d/exp/exp_prostate_local.yaml --models configs/2d/models.yaml `
    --fold $fold --expert-name segresnet-2d
}
```

## 6. Layer2、Gating 与评估

### 6.1 生成 Layer1 OOF，训练 Layer2 与 Gating

```powershell
python scripts/inference/generate_layer1_oof.py `
  --exp configs/2d/exp/exp_prostate_local.yaml --models configs/2d/models.yaml `
  --which best --batch-size 32 --tta

foreach ($fold in 0..4) {
  python scripts/train/train_layer2.py `
    --exp configs/2d/exp/exp_prostate_local.yaml `
    --training configs/2d/training_layer2.yaml `
    --models configs/2d/models.yaml --augs configs/2d/augs.yaml `
    --fold $fold --gpus 0,1
}

python scripts/inference/generate_layer2_oof.py `
  --exp configs/2d/exp/exp_prostate_local.yaml --models configs/2d/models.yaml `
  --which best --batch-size 32 --tta

foreach ($fold in 0..4) {
  python scripts/train/train_gating.py `
    --exp configs/2d/exp/exp_prostate_local.yaml `
    --gating-config configs/2d/gating_dual_5090.yaml `
    --models configs/2d/models.yaml --fold $fold --gpus 0,1
}
```

### 6.2 五折验证集评估

```powershell
foreach ($fold in 0..4) {
  python scripts/inference/gating_inference.py `
    --exp configs/2d/exp/exp_prostate_local.yaml `
    --gating-config configs/2d/gating_dual_5090.yaml `
    --models configs/2d/models.yaml --fold $fold --split "val_fold$fold"
}

python scripts/eval/eval_methods.py `
  --exp configs/2d/exp/exp_prostate_local.yaml `
  --training configs/2d/training_layer2.yaml `
  --models configs/2d/models.yaml --fold all
```

### 6.3 固定测试集评估

```powershell
python scripts/eval/run_test_pipeline.py `
  --exp configs/2d/exp/exp_prostate_local.yaml `
  --training configs/2d/training_layer2.yaml `
  --gating-config configs/2d/gating_dual_5090.yaml `
  --models configs/2d/models.yaml `
  --predictor-fold 0 --gpus 0,1 --batch-size 32 --tta
```

## 7. 写论文时主要看这些结果

验证集汇总结果默认输出到：

- `runs/segmoe_2d_prostate/results/metrics_prostate_local_all_val_folds.csv`
- `runs/segmoe_2d_prostate/results/metrics_by_fold_prostate_local_all_val_folds.csv`
- `runs/segmoe_2d_prostate/results/metrics_per_sample_prostate_local_all_val_folds.csv`

固定测试集结果示例：

- `runs/segmoe_2d_prostate/results/metrics_prostate_local_fold0_test.csv`

如需导出论文表格，可额外执行：

```powershell
python scripts/eval/export_tables.py --exp configs/2d/exp/exp_prostate_local.yaml --folds 0 1 2 3 4
```

## 8. 推荐执行顺序

```text
1. prepare_prostate
2. make_splits
3. 训练并导入 nnUNet / SwinUNETR / SegResNet
4. generate_layer1_oof
5. train_layer2
6. generate_layer2_oof
7. train_gating
8. gating_inference
9. eval_methods
10. run_test_pipeline
```

如果你的目标是尽快完成论文主实验，只需要沿着上面的主线跑通即可，不必再考虑 3D 或其他数据集配置。

# Seg-MoE（本地前列腺数据完整训练手册）

本项目实现 Two-Layer Ensemble（Layer1 → Layer1 OOF → Layer2 → Layer2 OOF → Gating），
支持三专家融合：nnUNet、SwinUNETR、SegResNet。

本 README 按你当前本地数据场景编写：
- 本地数据目录：`D:\Dataset002_ProstateCrop_Seg`
- 任务：2D 训练流程（但 nnUNet 官方输入仍是 3D NIfTI，属于正常设计）
- 实验配置：`configs/2d/exp/exp_prostate_local.yaml`

---

## 1. 先决条件

### 1.1 Python 环境

```powershell
conda create -n segmoe python=3.10 -y
conda activate segmoe
pip install -r requirements.txt
pip install -e .
```

### 1.2 nnUNet 依赖

```powershell
pip install "nnunetv2>=2.2"
```

### 1.3 GPU 检查（可选但推荐）

```powershell
nvidia-smi
```

如果 `torch.cuda.is_available()` 在你机器上崩溃，先修复驱动后再开始训练。

---

## 2. 数据与配置对应关系

你的本地数据应为 nnUNet 风格目录：

```text
D:\Dataset002_ProstateCrop_Seg/
  imagesTr/
    xxx_0000.nii.gz
    xxx_0001.nii.gz
    xxx_0002.nii.gz
  labelsTr/
    xxx.nii.gz
  imagesTs/
  labelsTs/
```

项目中对应配置：
- 数据集配置：`configs/2d/datasets/prostate_local.yaml`
- 实验配置：`configs/2d/exp/exp_prostate_local.yaml`
- 模型配置：`configs/2d/models.yaml`
- Layer2 训练配置：`configs/2d/training_layer2.yaml`
- 门控配置：`configs/2d/gating.yaml`

---

## 3. 一次性数据准备（本地前列腺）

### 3.1 生成 2D PNG（供 SwinUNETR / SegResNet / Seg-MoE 2D 管线）

```powershell
python scripts/data/prepare_prostate.py --config configs/2d/datasets/prostate_local.yaml
```

输出：
- `data/processed/prostate_local/images/*.png`
- `data/processed/prostate_local/masks/*.png`
- `data/splits/prostate_local/index_all.jsonl`

### 3.2 生成 5-fold 切分（固定 raw_test）

```powershell
python scripts/data/make_splits.py --dataset-config configs/2d/datasets/prostate_local.yaml
```

输出：
- `data/splits/prostate_local/splits_train5fold_testfixed.jsonl`

---

## 4. Layer1：三专家官方训练

> 三专家均可独立训练，互不依赖。推荐并行跑 nnUNet（多终端）+ SwinUNETR/SegResNet（主终端）。

---

## 4.1 Expert A：nnUNet v2（官方 CLI）

> nnUNet 仍使用 3D NIfTI 输入，配置选 `2d` 即为 2D 切片训练，这是官方标准设计。

### Step A1. 建立 nnUNet task（仅需一次）

```powershell
python scripts/nnunet/setup_nnunet_task.py `
  --data-dir D:/Dataset002_ProstateCrop_Seg `
  --dataset-id 2 `
  --dataset-name ProstateCrop_Seg `
  --channel-names T2w ADC DWI `
  --labels background PZ TZ lesion `
  --exp configs/2d/exp/exp_prostate_local.yaml `
  --verify
```

### Step A2. 设置 nnUNet 环境变量（每个新终端都要执行）

```powershell
$env:nnUNet_raw          = "D:\Seg-MoE\nnunet_data\nnUNet_raw"
$env:nnUNet_preprocessed = "D:\Seg-MoE\nnunet_data\nnUNet_preprocessed"
$env:nnUNet_results      = "D:\Seg-MoE\nnunet_data\nnUNet_results"
```

### Step A3. 官方训练（2D，5 折）

```powershell
# 首次训练
foreach ($fold in 0..4) {
  nnUNetv2_train 2 2d $fold --npz
}

# 中断后恢复
foreach ($fold in 0..4) {
  nnUNetv2_train 2 2d $fold --npz --c
}
```

### Step A4. 导入 nnUNet 权重到 Seg-MoE

> 导入脚本默认会做目标冲突检查：若已有 `best.pt` 且来源不同会报错，避免覆盖；确需覆盖请加 `--overwrite`。

```powershell
python scripts/nnunet/import_nnunet_weights.py `
  --nnunet-base nnunet_data `
  --dataset-id 2 `
  --config 2d `
  --folds 0 1 2 3 4 `
  --exp configs/2d/exp/exp_prostate_local.yaml `
  --models configs/2d/models.yaml `
  --expert-name nnunet-2d `
  --update-models-yaml configs/2d/models.yaml
```

---

## 4.2 Expert B：SwinUNETR（MONAI 1.5 官方 recipe）

**已验证参数**（MONAI 1.5.2，`img_size` 已移除，用 `patch_size=2` 代替）：
- `feature_size=48`（base 规格，25M 参数）
- `spatial_dims=2`，`depths=(2,2,2,2)`，`num_heads=(3,6,12,24)`
- 优化器 AdamW `lr=1e-4`，WarmupCosine warmup=50 epoch（step-level）
- 损失 DiceCELoss（`to_onehot_y=True, softmax=True`）

### Step B1. 训练

单折（快速验证）：

```powershell
python scripts/monai/train_swinunetr_official.py `
  --exp    configs/2d/exp/exp_prostate_local.yaml `
  --models configs/2d/models.yaml `
  --fold 0 --gpus 0,1 `
  --epochs 300 --batch-size 16 `
  --amp --amp-dtype bfloat16 `
  --num-workers 2
```

全 5 折：

```powershell
foreach ($fold in 0..4) {
  python scripts/monai/train_swinunetr_official.py `
    --exp    configs/2d/exp/exp_prostate_local.yaml `
    --models configs/2d/models.yaml `
    --fold $fold --gpus 0,1 `
    --epochs 300 --batch-size 16 `
    --amp --amp-dtype bfloat16 `
    --num-workers 2
}
```

中断恢复：

```powershell
# SwinUNETR 5-fold 续训
foreach ($fold in 0..4) {
  python scripts/monai/train_swinunetr_official.py `
    --exp    configs/2d/exp/exp_prostate_local.yaml `
    --models configs/2d/models.yaml `
    --fold $fold --gpus 0,1 `
    --epochs 300 --batch-size 16 `
    --amp --amp-dtype bfloat16 `
    --num-workers 2 `
    --resume "runs/swinunetr_official_prostate_local/fold$fold/latest_model.pt"
}
```

### Step B2. 导入权重到 Seg-MoE

> 建议始终显式指定 `--expert-name`，避免同类型多专家时写入冲突。

```powershell
foreach ($fold in 0..4) {
  python scripts/monai/import_swinunetr_weights.py `
    --source runs/swinunetr_official_prostate_local/fold$fold/best_model.pt `
    --exp    configs/2d/exp/exp_prostate_local.yaml `
    --models configs/2d/models.yaml `
    --fold $fold `
    --expert-name swinunetr-2d
}
```

---

## 4.3 Expert C：SegResNet（MONAI Auto3DSeg 官方 recipe）

**已验证参数**（MONAI 1.5.2，29M 参数）：
- `SegResNetDS(dsdepth=2)` 训练（深度监督）→ `dsdepth=1` 推理
- `init_filters=32`，`blocks_down=(1,2,2,4,4)`（5 阶），`norm=BATCH`
- 优化器 AdamW `lr=2e-4`，WarmupCosine warmup=3 epoch（epoch-level）
- 损失 `DiceCELoss(squared_pred=True, batch=True)`（Auto3DSeg 官方设定）

### Step C1. 训练

单折（快速验证）：

```powershell
python scripts/monai/train_segresnet_official.py `
  --exp    configs/2d/exp/exp_prostate_local.yaml `
  --models configs/2d/models.yaml `
  --fold 0 --gpus 0,1 `
  --epochs 300 --batch-size 32 `
  --dsdepth 2 `
  --amp --amp-dtype bfloat16 `
  --num-workers 2
```

全 5 折：

```powershell
foreach ($fold in 0..4) {
  python scripts/monai/train_segresnet_official.py `
    --exp    configs/2d/exp/exp_prostate_local.yaml `
    --models configs/2d/models.yaml `
    --fold $fold --gpus 1 `
    --epochs 300 --batch-size 32 `
    --dsdepth 2 `
    --amp --amp-dtype bfloat16 `
    --num-workers 2
}
```

中断恢复：

```powershell
# SegResNet 5-fold 续训
foreach ($fold in 0..4) {
  python scripts/monai/train_segresnet_official.py `
    --exp    configs/2d/exp/exp_prostate_local.yaml `
    --models configs/2d/models.yaml `
    --fold $fold --gpus 1 `
    --epochs 300 --batch-size 32 `
    --dsdepth 2 `
    --amp --amp-dtype bfloat16 `
    --num-workers 2 `
    --resume "runs/segresnet_official_prostate_local/fold$fold/latest_model.pt"
}
```

### Step C2. 导入权重到 Seg-MoE

> 若需要强制覆盖已导入权重，可在命令末尾追加 `--overwrite`。

```powershell
foreach ($fold in 0..4) {
  python scripts/monai/import_segresnet_weights.py `
    --source runs/segresnet_official_prostate_local/fold$fold/best_model.pt `
    --exp    configs/2d/exp/exp_prostate_local.yaml `
    --models configs/2d/models.yaml `
    --fold $fold `
    --expert-name segresnet-2d
}
```

---

## 4.4 TensorBoard 监控（三专家共用）

```powershell
# 新终端启动（保持后台运行）
tensorboard --logdir runs/ --port 6006
# 浏览器打开 http://localhost:6006
```

验收指标：
| 专家 | Epoch 1 loss 参考 | 收敛后 val Dice |
|------|-------------------|----------------|
| nnUNet | 0.5–1.0（DS loss） | > 0.70 |
| SwinUNETR | 1.0–2.0 | > 0.65 |
| SegResNet | 0.8–1.5（DS loss） | > 0.65 |

---

## 4.5 显存与 batch size 参考（双 RTX 5090，bfloat16）

| 专家 | 参数量 | 推荐 batch-size | 单卡显存 |
|------|-------|----------------|---------|
| nnUNet | ~46M | 由 nnUNet 自动规划 | ~12 GB |
| SwinUNETR | 25M | 16（DataParallel） | ~14 GB |
| SegResNet | 29M | 32（DataParallel） | ~8 GB |

OOM 时依次尝试：`--batch-size 16 → 8 → 4`；或加 `--num-workers 0`。

---

## 5. 生成 Layer1 OOF（训练 Layer2 的前提）

```powershell
python scripts/inference/generate_layer1_oof.py `
  --exp configs/2d/exp/exp_prostate_local.yaml `
  --models configs/2d/models.yaml `
  --which best `
  --batch-size 32 `
  --tta
```

输出（默认）：
- `runs/segmoe_2d_prostate/cache/oof/layer1/fold_*/{sample_id}.npz`
- `runs/segmoe_2d_prostate/cache/oof/layer1/oof_manifest.jsonl`

---

## 6. 训练 Layer2（三专家）

单折：

```powershell
python scripts/train/train_layer2.py `
  --exp configs/2d/exp/exp_prostate_local.yaml `
  --training configs/2d/training_layer2.yaml `
  --models configs/2d/models.yaml `
  --augs configs/2d/augs.yaml `
  --fold 0 --gpus 0,1
```

全 5 折：

```powershell
foreach ($fold in 0..4) {
  python scripts/train/train_layer2.py `
    --exp configs/2d/exp/exp_prostate_local.yaml `
    --training configs/2d/training_layer2.yaml `
    --models configs/2d/models.yaml `
    --augs configs/2d/augs.yaml `
    --fold $fold --gpus 0,1
}
```

---

## 7. 生成 Layer2 OOF（训练 Gating 的前提）

```powershell
python scripts/inference/generate_layer2_oof.py `
  --exp configs/2d/exp/exp_prostate_local.yaml `
  --models configs/2d/models.yaml `
  --which best `
  --batch-size 32 `
  --tta
```

输出（默认）：
- `runs/segmoe_2d_prostate/cache/oof/layer2/fold_*/{sample_id}.npz`
- `runs/segmoe_2d_prostate/cache/oof/layer2/oof_manifest_layer2.jsonl`

---

## 8. 训练 Gating（动态融合）

单折：

```powershell
python scripts/train/train_gating.py `
  --exp configs/2d/exp/exp_prostate_local.yaml `
  --gating-config configs/2d/gating.yaml `
  --models configs/2d/models.yaml `
  --fold 0 --gpus 0,1
```

全 5 折：

```powershell
foreach ($fold in 0..4) {
  python scripts/train/train_gating.py `
    --exp configs/2d/exp/exp_prostate_local.yaml `
    --gating-config configs/2d/gating.yaml `
    --models configs/2d/models.yaml `
    --fold $fold --gpus 0,1
}
```

---

## 9. 门控推理与评估导出

### 9.1 Gating 推理（可导出门控权重可视化）

```powershell
python scripts/inference/gating_inference.py `
  --exp configs/2d/exp/exp_prostate_local.yaml `
  --gating-config configs/2d/gating.yaml `
  --models configs/2d/models.yaml `
  --fold 0 --save-weights
```

### 9.2 缓存概率图（可选）

```powershell
python scripts/inference/cache_probs.py `
  --exp configs/2d/exp/exp_prostate_local.yaml `
  --models configs/2d/models.yaml `
  --layer layer1 --fold 0

python scripts/inference/cache_probs.py `
  --exp configs/2d/exp/exp_prostate_local.yaml `
  --models configs/2d/models.yaml `
  --layer layer2 --fold 0
```

### 9.3 评估并导出表格

```powershell
python scripts/eval/eval_methods.py `
  --exp configs/2d/exp/exp_prostate_local.yaml `
  --training configs/2d/training_layer2.yaml `
  --models configs/2d/models.yaml `
  --fold 0

python scripts/eval/export_tables.py --exp configs/2d/exp/exp_prostate_local.yaml --folds 0
python scripts/eval/export_weights.py --exp configs/2d/exp/exp_prostate_local.yaml --models configs/2d/models.yaml --folds 0
```

---

## 10. 推荐执行顺序（最简版）

```text
prepare_prostate.py
  -> make_splits.py
  -> setup_nnunet_task.py
  -> nnUNetv2_train (5 folds)
  -> train_swinunetr_official.py (5 folds)
  -> train_segresnet_official.py (5 folds)
  -> import_nnunet_weights.py
  -> import_swinunetr_weights.py
  -> import_segresnet_weights.py
  -> generate_layer1_oof.py
  -> train_layer2.py (5 folds)
  -> generate_layer2_oof.py
  -> train_gating.py (5 folds)
  -> gating_inference.py / eval_methods.py / export_tables.py
```

---

## 11. 常见问题

### Q1：我明明跑 2D，为什么 nnUNet 用的是 3D NIfTI？

这是官方设计：
- 输入文件是 3D NIfTI（保留体数据和 spacing 信息）
- 训练配置选择 `2d`（`nnUNetv2_train <id> 2d <fold>`）
- nnUNet 内部完成 2D 切片训练

### Q2：这会影响 OOF 吗？

不会。OOF 的关键是 fold 隔离和 `sample_id` 对齐。只要你按本 README 的链路生成 OOF，
Layer2/Gating 使用的是统一对齐后的 OOF 缓存。

### Q3：如何快速验证流程是否跑通？

可先单折（`--fold 0`）跑完整链路，确认成功后再扩展到 5 折。

---

## 12. 相关文档

- `docs/TRAINING_PROSTATE.md`：前列腺专题训练说明（补充版）
- `docs/ARCH_2D.md`：2D 架构说明
- `docs/ROADMAP_3D.md`：3D 规划

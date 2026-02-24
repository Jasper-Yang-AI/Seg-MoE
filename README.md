# Seg-MoE: Two-Layer Ensemble for Medical Image Segmentation

实现 Dang et al. (2024) "Two-layer Ensemble of Deep Learning Models for Medical Image Segmentation" 的两层集成框架。

**支持数据集**：MSD Task03 Liver | MSD Task07 Pancreas | ACDC | BTCV

## 核心特性

- **两层 Stacking + 动态门控融合**：Layer1 → L1 OOF → Layer2 → **L2 OOF** → Patch 门控网络 → 空间自适应融合
- **三专家组合**：nnUNet (CNN) + SwinUNETR (Transformer) + SegResNet (ResEncoder)
- **Layer1 官方训练**：三专家均使用各自官方训练流程，严格复现论文性能
  - nnUNet: 1000 epochs, SGD, PolyLR, 深度监督
  - SwinUNETR: MONAI Recipe, AdamW 1e-4, WarmupCosine, DiceCELoss
  - SegResNet: MONAI Auto3DSeg Recipe, AdamW 2e-4, WarmupCosine, DeepSupervisionLoss
- **Patch-Level 动态门控网络** (课题核心创新)：
  - 概率图切 patch (64×64, stride=32, 50% 重叠)
  - 轻量 ConvNet 门控 (~25K params): 3层Conv+GAP+FC
  - Softmax 温度退火 (τ: 2.0→0.5)
  - 负载平衡正则 (Shazeer 2017, 防专家坼缩)
  - Gaussian blending 平滑合并
- **Layer2 科研级优化** (B1-B5)：
  - **B1** Layer1→Layer2 权重迁移（zero-init 额外通道）
  - **B2** 独立 Layer2 训练配置（低学习率 4e-5, 100 epochs）
  - **B3** Boundary Loss（Kervadec et al. MIDL 2019, CE + Dice + Boundary）
  - **B4** Per-expert 差分化训练配置（nnUNet→SGD+PolyLR, SwinUNETR→AdamW+Cosine, SegResNet→AdamW+Cosine）
  - **B5** 不确定性通道（entropy map + disagreement map, in_channels: 3→16）
- **OOF 优化**：批量推理 (batch_size=32) + TTA（水平/垂直翻转三路平均）
- **融合方法**：动态门控 (Gating) / OLE / DT / WE-CLPSO
- **多格式支持**：NIfTI (3D→2D切片) / PNG / JPEG / DICOM
- **Windows 原生兼容**：DataParallel 多卡训练，不依赖 torchrun/DDP
- **严格复现**：固定随机种子 + 5-fold 交叉验证

## 项目结构

```
configs/2d/
  ├── models.yaml                # 三专家配置 (nnUNet/SwinUNETR/SegResNet)
  ├── datasets/                  # 数据集配置
  ├── training.yaml              # Layer1 基础训练超参
  ├── training_layer2.yaml       # Layer2 训练超参 (B2: 低LR+短epochs, B3: BoundaryLoss, B4: per-expert)
  ├── training_dual_5090.yaml    # 双卡优化超参 (AMP+AdamW+Cosine)
  ├── gating.yaml                # 门控网络配置 (patch大小/stride/温度退火/负载平衡)
  ├── augs.yaml                  # 数据增强
  ├── debug.yaml                 # Debug 快速覆写
  └── exp/                       # 实验入口配置
src/seg_moe/
  ├── models/factory_2d.py       # 统一模型工厂 (build_expert + B1: transfer_layer1_to_layer2)
  ├── models/wrappers/           # nnUNet wrapper (支持 deep_supervision)
  ├── training/engine.py         # 训练引擎 (DP + AMP + checkpoint + config-driven loss)
  ├── training/losses.py         # 损失函数 (ce_plus_dice / ce_dice_boundary / build_loss_fn)
  ├── gating/patch_gating_2d.py  # Patch 门控网络 (PatchConvGate2D, ~25K params)
  ├── gating/patch_gating_3d.py  # 3D 门控接口 (reserved stub)
  ├── data/layer2_oof_dataset.py # Layer2 数据集 (B5: entropy + disagreement 不确定性通道)
  ├── data/gating_patch_dataset.py # 门控训练数据集 (概率图 patch 切分)
  ├── utils/patches.py           # Patch split/merge + Gaussian blending
  ├── combiners/                 # OLE / DT / WE-CLPSO 融合器
  ├── data/                      # 多格式数据加载
  └── evaluation/                # Dice / IoU / HD / MAD 指标
scripts/
  ├── data/                      # 数据准备 (prepare, splits, labels)
  ├── train/                     # 训练 (2D/3D experts, layer2, gating)
  │   ├── train_layer2.py        # Layer2 训练 (B1-B5 全集成)
  │   └── train_gating.py        # 门控网络训练 (概率图 patch → 动态融合权重)
  ├── inference/                 # 推理缓存
  │   ├── generate_layer1_oof.py # Layer1 OOF 生成 (P1: batch推理, P2: TTA)
  │   ├── generate_layer2_oof.py # Layer2 OOF 生成 (门控网络输入)
  │   └── gating_inference.py    # 门控动态融合推理 + 权重可视化
  ├── eval/                      # 评估导出 (metrics, tables, viz)
  ├── nnunet/                    # nnUNet 官方训练集成
  │   ├── setup_nnunet_task.py   # 数据集转换 + 预处理
  │   └── import_nnunet_weights.py  # 官方权重导入
  ├── monai/                     # MONAI 官方训练集成
  │   ├── train_swinunetr_official.py   # SwinUNETR 官方训练
  │   ├── import_swinunetr_weights.py   # SwinUNETR 权重导入
  │   ├── train_segresnet_official.py   # SegResNet 官方训练
  │   └── import_segresnet_weights.py   # SegResNet 权重导入
  └── utils/                     # 工具 (sanity, validate, smoke)
tests/                           # 单元测试
```

## 快速开始

### 1. 环境安装

```bash
conda create -n segmoe python=3.10
conda activate segmoe
pip install -r requirements.txt
pip install -e .
```

### 2. 数据准备

```bash
python scripts/data/prepare_msd.py --config configs/2d/datasets/msd_task03_liver.yaml
python scripts/data/make_splits.py --dataset-config configs/2d/datasets/msd_task03_liver.yaml
python scripts/data/check_labels.py --dataset-config configs/2d/datasets/msd_task03_liver.yaml --splits --sample 50
```

### 3. 完整训练与评估流程（双卡 RTX 5090 最优配置）

本项目的训练分为三部分 — **所有 Layer1 专家均使用各自官方训练流程**:
- **nnUNet**: 使用 **官方 nnUNet v2 训练流程** (1000 epochs, SGD + PolyLR, 深度监督)
- **SwinUNETR**: 使用 **官方 MONAI Recipe** (300 epochs, AdamW 1e-4, WarmupCosine, DiceCELoss)
- **SegResNet**: 使用 **官方 MONAI Auto3DSeg Recipe** (300 epochs, AdamW 2e-4, WarmupCosine, DeepSupervisionLoss)

#### Phase 1: nnUNet 官方训练

```powershell
# ---- Step 0: 安装 nnUNet v2 ----
pip install nnunetv2>=2.2

# ---- Step 1: 数据集转换 + 预处理 (自动设置环境变量) ----
python scripts/nnunet/setup_nnunet_task.py `
  --msd-dir D:\MSD_Dataset/Task03_Liver `
  --dataset-id 3 `
  --nnunet-base nnunet_data `
  --verify --overwrite

# ---- Step 2: 设置环境变量 (每次新开终端需要) ----
$env:nnUNet_raw = "$PWD\nnunet_data\nnUNet_raw"
$env:nnUNet_preprocessed = "$PWD\nnunet_data\nnUNet_preprocessed"
$env:nnUNet_results = "$PWD\nnunet_data\nnUNet_results"

# ---- Step 3: nnUNet 官方训练 (每折 ~1000 epochs, SGD, PolyLR) ----
# 单折训练 (~6-12 小时/折 on RTX 5090):
nnUNetv2_train 3 2d 0 --npz

# 或 5 折全部训练:
foreach ($fold in 0..4) { nnUNetv2_train 3 2d $fold --npz }
```

> **nnUNet 官方训练参数** (自动配置, 无需手动调整):
> | 参数 | 值 |
> |------|------|
> | Epochs | 1000 |
> | 优化器 | SGD (lr=0.01, momentum=0.99, nesterov=True) |
> | 学习率策略 | PolyLR `(1 - epoch/max_epoch)^0.9` |
> | 深度监督 | 是 (Deep Supervision, 多尺度损失) |
> | 损失函数 | CE + Dice |
> | 数据增强 | nnUNet 内置 batchgenerators (旋转/缩放/镜像/Gamma) |
> | 架构 | PlainConvUNet (自动规划 stages/channels/patch_size) |

#### Phase 2: 导入 nnUNet 权重到 Seg-MoE

```powershell
# ---- Step 4: 导入训练好的 nnUNet 权重 + 自动更新 models.yaml ----
python scripts/nnunet/import_nnunet_weights.py `
  --nnunet-base nnunet_data `
  --dataset-id 3 `
  --config 2d `
  --folds 0 1 2 3 4 `
  --exp configs/2d/exp/exp_msd_task03_liver.yaml `
  --update-models-yaml configs/2d/models.yaml

# 脚本会:
#   1. 读取 nnUNet plans → 提取网络架构参数
#   2. 转换 nnUNet checkpoint → Seg-MoE 格式
#   3. 保存到 runs/segmoe_2d_msd03/checkpoints/layer1/fold{k}/nnunet-2d/best.pt
#   4. (--update-models-yaml) 自动将 nnUNet plans 中的架构参数写入 models.yaml
#      (features_per_stage / conv_kernel_sizes / pool_op_kernel_sizes 等)
```

#### Phase 2.5: SwinUNETR 官方训练 (MONAI Recipe)

```powershell
# ---- Step 5: SwinUNETR 官方训练 (每折 300 epochs) ----
# 单折训练 (~3-6 小时/折 on RTX 5090):
python scripts/monai/train_swinunetr_official.py `
  --exp configs/2d/exp/exp_msd_task03_liver.yaml `
  --models configs/2d/models.yaml `
  --fold 0 --gpus 0,1

# 或 5 折全部训练:
foreach ($fold in 0..4) {
  python scripts/monai/train_swinunetr_official.py `
    --exp configs/2d/exp/exp_msd_task03_liver.yaml `
    --models configs/2d/models.yaml `
    --fold $fold --gpus 0,1
}

# ---- Step 6: 导入 SwinUNETR 权重到 Seg-MoE ----
foreach ($fold in 0..4) {
  python scripts/monai/import_swinunetr_weights.py `
    --source runs/swinunetr_official_msd_task03_liver/fold$fold/best_model.pt `
    --exp configs/2d/exp/exp_msd_task03_liver.yaml `
    --models configs/2d/models.yaml --fold $fold
}
```

> **SwinUNETR 官方训练参数** (Tang et al., CVPR 2022):
> | 参数 | 值 |
> |------|------|
> | Epochs | 300 |
> | 优化器 | AdamW (lr=1e-4, weight_decay=1e-5) |
> | 学习率策略 | WarmupCosineSchedule (warmup 50 epochs) |
> | 损失函数 | DiceCELoss (softmax + one-hot) |
> | 数据增弼 | MONAI transforms (Flip/Rotate90/ScaleIntensity/ShiftIntensity) |
> | 架构 | SwinUNETR (feature_size=48, depths=[2,2,2,2]) |

#### Phase 2.75: SegResNet 官方训练 (MONAI Auto3DSeg Recipe)

```powershell
# ---- Step 7: SegResNet 官方训练 (每折 300 epochs) ----
# 单折训练 (~2-4 小时/折 on RTX 5090):
python scripts/monai/train_segresnet_official.py `
  --exp configs/2d/exp/exp_msd_task03_liver.yaml `
  --models configs/2d/models.yaml `
  --fold 0 --gpus 0,1

# 或 5 折全部训练:
foreach ($fold in 0..4) {
  python scripts/monai/train_segresnet_official.py `
    --exp configs/2d/exp/exp_msd_task03_liver.yaml `
    --models configs/2d/models.yaml `
    --fold $fold --gpus 0,1
}

# ---- Step 8: 导入 SegResNet 权重到 Seg-MoE ----
foreach ($fold in 0..4) {
  python scripts/monai/import_segresnet_weights.py `
    --source runs/segresnet_official_msd_task03_liver/fold$fold/best_model.pt `
    --exp configs/2d/exp/exp_msd_task03_liver.yaml `
    --models configs/2d/models.yaml --fold $fold
}
```

> **SegResNet 官方训练参数** (MONAI Auto3DSeg SegResNet2D):
> | 参数 | 值 |
> |------|------|
> | Epochs | 300 |
> | 优化器 | AdamW (lr=2e-4, weight_decay=1e-5) |
> | 学习率策略 | WarmupCosineSchedule (warmup 3 epochs, epoch-level) |
> | 损失函数 | DeepSupervisionLoss(DiceCELoss(squared_pred=True, batch=True)) |
> | 数据增强 | RandAffined + RandFlipd + GaussSmooth + ScaleIntensity + GaussNoise |
> | 架构 | SegResNetDS (init_filters=32, blocks_down=[1,2,2,4,4], dsdepth=2) |

#### Phase 3: L1 OOF 生成 + Layer2 训练

```powershell
# ---- Step 9: 生成 Layer1 OOF 概率图 ----
# P1: 批量推理 (默认 batch_size=32, 比逐样本快 10-20x)
# P2: TTA (--tta 开启水平+垂直翻转三路平均, 提升 OOF 质量)
python scripts/inference/generate_layer1_oof.py `
  --exp configs/2d/exp/exp_msd_task03_liver.yaml `
  --models configs/2d/models.yaml --which best `
  --batch-size 32 --tta

# ---- Step 10: Layer2 训练 (使用独立 training_layer2.yaml 配置) ----
# B1: 自动从 Layer1 权重迁移 (skip --no-pretrain 则启用)
# B2: 独立配置 (lr=4e-5, epochs=100, warmup=5)
# B3: Boundary Loss (ce_dice_boundary, boundary_weight=0.5)
# B4: Per-expert 差分化训练 (nnUNet→SGD+PolyLR, SwinUNETR/SegResNet→AdamW+Cosine)
# B5: 不确定性通道 (entropy + disagreement, 默认开启; --no-uncertainty 关闭)
python scripts/train/train_layer2.py `
  --exp configs/2d/exp/exp_msd_task03_liver.yaml `
  --training configs/2d/training_layer2.yaml `
  --models configs/2d/models.yaml `
  --augs configs/2d/augs.yaml `
  --fold 0 --gpus 0,1
```

> **Layer2 训练优化详解** (B1-B5):
>
> | 优化 | 技术 | 原理 |
> |------|------|------|
> | **B1** 权重迁移 | `transfer_layer1_to_layer2()` | 复制 Layer1 已训练的 shared params，额外通道 zero-init，避免从零学习 |
> | **B2** 独立配置 | `training_layer2.yaml` | lr=4e-5 (Layer1的40%), epochs=100 (Layer1的33%), 防止覆盖迁移权重 |
> | **B3** Boundary Loss | Kervadec et al. MIDL 2019 | 有符号距离变换 × softmax 概率，直接优化边缘对齐，weight=0.5 |
> | **B4** 差分化训练 | `expert_overrides` | 各 expert 沿用 Layer1 官方 optimizer/scheduler recipe |
> | **B5** 不确定性 | entropy + disagreement | entropy=−Σp·log(p)/log(M); disagreement=std across K experts; in_channels: 3→16 |

#### Phase 4: Layer2 OOF 生成 (门控输入)

```powershell
# ---- Step 11: 生成 Layer2 OOF 概率图 ----
# Layer2 专家以 [image + L1_probs + uncertainty] 为输入 (in_channels=16)
# OOF 原理同 Layer1: fold k 的 Layer2 模型预测 val_fold{k}
# 输出: cache/oof/layer2/fold_{k}/{sample_id}.npz, probs shape [K,M,H,W]
python scripts/inference/generate_layer2_oof.py `
  --exp configs/2d/exp/exp_msd_task03_liver.yaml `
  --models configs/2d/models.yaml --which best `
  --batch-size 32 --tta
```

#### Phase 5: 门控网络训练 + 动态融合推理

```powershell
# ---- Step 12: 训练门控网络 (Layer2 OOF probs → patch → 动态融合权重) ----
# 正确流程: Layer1 → L1_OOF → Layer2 → **L2_OOF** → Gating
# 门控输入 = Layer2 专家的 OOF 概率图 (非 Layer1!)
# 训练时间: 双 5090 约 5-10 分钟 (网络仅 ~25K params, 50 epochs)
python scripts/train/train_gating.py `
  --exp configs/2d/exp/exp_msd_task03_liver.yaml `
  --gating-config configs/2d/gating.yaml `
  --models configs/2d/models.yaml `
  --fold 0 --gpus 0,1

# ---- Step 13: 门控动态融合推理 ----
# 全图推理: L2 probs → 切patch → 预测门控权重 → 加权融合 → Gaussian blending合并
python scripts/inference/gating_inference.py `
  --exp configs/2d/exp/exp_msd_task03_liver.yaml `
  --gating-config configs/2d/gating.yaml `
  --models configs/2d/models.yaml `
  --fold 0 --save-weights

# ---- Step 14: 缓存概率图 + 评估 + 导出 ----
python scripts/inference/cache_probs.py `
  --exp configs/2d/exp/exp_msd_task03_liver.yaml `
  --models configs/2d/models.yaml `
  --layer layer1 --fold 0

python scripts/eval/eval_methods.py `
  --exp configs/2d/exp/exp_msd_task03_liver.yaml `
  --training configs/2d/training_layer2.yaml `
  --models configs/2d/models.yaml

python scripts/eval/export_tables.py --exp configs/2d/exp/exp_msd_task03_liver.yaml
python scripts/eval/export_weights.py `
  --exp configs/2d/exp/exp_msd_task03_liver.yaml `
  --models configs/2d/models.yaml
```

> **门控网络架构详解**:
>
> | 组件 | 设计 | 参数量 |
> |------|------|--------|
> | 输入 | expert prob patches `[K×M, pH, pW]` = `[9, 64, 64]` | — |
> | Stem | Conv2d(9, 64, 3) + BN + GELU | ~3.5K |
> | Down1 | Conv2d(64, 64, 3, s=2) + BN + GELU | ~37K |
> | Down2 | Conv2d(64, 64, 3, s=2) + BN + GELU | ~37K |
> | Pool | AdaptiveAvgPool2d(1) | — |
> | Head | FC(64→32) + GELU + Dropout + FC(32→3) | ~2.2K |
> | 输出 | softmax/τ → `[K]` = `[3]` | — |
> | **合计** | | **~25K** |
>
> **训练策略**:
> - 温度退火: τ = 2.0 × (0.5/2.0)^(epoch/max_epoch), 初期均匀→后期锐化
> - 负载平衡: $\lambda_{lb} \cdot K \cdot \sum_k f_k^2$, 防止某专家独占所有 patch
> - 监督信号: fused = Σ_k w_k · probs_k, loss = DiceCE(fused, GT)

#### Debug 快速验证 (epochs=2, samples=16)

```powershell
python scripts/train/train_2d_experts.py `
  --exp configs/2d/exp/exp_msd_task03_liver.yaml `
  --training configs/2d/training_dual_5090.yaml `
  --models configs/2d/models.yaml `
  --augs configs/2d/augs.yaml `
  --debug configs/2d/debug.yaml `
  --fold 0 --layer layer1 --gpus 0,1
```

### 5 折交叉验证（一键 PowerShell 流水线）

```powershell
# =========================================================
# Phase 1: nnUNet 官方训练 (5 折, 每折 1000 epochs)
# =========================================================
$env:nnUNet_raw = "$PWD\nnunet_data\nnUNet_raw"
$env:nnUNet_preprocessed = "$PWD\nnunet_data\nnUNet_preprocessed"
$env:nnUNet_results = "$PWD\nnunet_data\nnUNet_results"

foreach ($fold in 0..4) { nnUNetv2_train 3 2d $fold --npz }

# =========================================================
# Phase 2: 导入 nnUNet 权重
# =========================================================
python scripts/nnunet/import_nnunet_weights.py `
  --nnunet-base nnunet_data --dataset-id 3 --config 2d `
  --folds 0 1 2 3 4 --exp configs/2d/exp/exp_msd_task03_liver.yaml `
  --update-models-yaml configs/2d/models.yaml

# =========================================================
# Phase 2.5: SwinUNETR 官方训练 + 导入 (MONAI Recipe)
# =========================================================
foreach ($fold in 0..4) {
  python scripts/monai/train_swinunetr_official.py `
    --exp configs/2d/exp/exp_msd_task03_liver.yaml `
    --models configs/2d/models.yaml `
    --fold $fold --gpus 0,1
}

foreach ($fold in 0..4) {
  python scripts/monai/import_swinunetr_weights.py `
    --source runs/swinunetr_official_msd_task03_liver/fold$fold/best_model.pt `
    --exp configs/2d/exp/exp_msd_task03_liver.yaml `
    --models configs/2d/models.yaml --fold $fold
}

# =========================================================
# Phase 2.75: SegResNet 官方训练 + 导入 (MONAI Auto3DSeg Recipe)
# =========================================================
foreach ($fold in 0..4) {
  python scripts/monai/train_segresnet_official.py `
    --exp configs/2d/exp/exp_msd_task03_liver.yaml `
    --models configs/2d/models.yaml `
    --fold $fold --gpus 0,1
}

foreach ($fold in 0..4) {
  python scripts/monai/import_segresnet_weights.py `
    --source runs/segresnet_official_msd_task03_liver/fold$fold/best_model.pt `
    --exp configs/2d/exp/exp_msd_task03_liver.yaml `
    --models configs/2d/models.yaml --fold $fold
}

# =========================================================
# Phase 3: Layer1 验证 — 所有专家权重已由官方训练导入
# --skip-if-done 会检测 best.pt 存在并跳过所有专家
# =========================================================
foreach ($fold in 0..4) {
  python scripts/train/train_2d_experts.py `
    --exp configs/2d/exp/exp_msd_task03_liver.yaml `
    --training configs/2d/training_dual_5090.yaml `
    --models configs/2d/models.yaml --augs configs/2d/augs.yaml `
    --fold $fold --layer layer1 --gpus 0,1 --skip-if-done
}

# =========================================================
# Phase 4: L1 OOF 生成 + Layer2 训练
# =========================================================
# P1+P2: 批量推理 + TTA
python scripts/inference/generate_layer1_oof.py `
  --exp configs/2d/exp/exp_msd_task03_liver.yaml `
  --models configs/2d/models.yaml --which best `
  --batch-size 32 --tta

# B1-B5: Layer2 训练 (使用独立 training_layer2.yaml)
foreach ($fold in 0..4) {
  python scripts/train/train_layer2.py `
    --exp configs/2d/exp/exp_msd_task03_liver.yaml `
    --training configs/2d/training_layer2.yaml `
    --models configs/2d/models.yaml --augs configs/2d/augs.yaml `
    --fold $fold --gpus 0,1 --skip-if-done
}

# =========================================================
# Phase 4.5: Layer2 OOF 生成 (门控网络的输入)
# =========================================================
# Layer2 专家以 [image + L1_probs + uncertainty] 为输入 (in_channels=16)
# OOF 原理同 Layer1: fold k 的 Layer2 模型预测 val_fold{k}
python scripts/inference/generate_layer2_oof.py `
  --exp configs/2d/exp/exp_msd_task03_liver.yaml `
  --models configs/2d/models.yaml --which best `
  --batch-size 32 --tta

python scripts/eval/eval_methods.py `
  --exp configs/2d/exp/exp_msd_task03_liver.yaml `
  --training configs/2d/training_layer2.yaml `
  --models configs/2d/models.yaml

python scripts/eval/export_tables.py `
  --exp configs/2d/exp/exp_msd_task03_liver.yaml --folds 0 1 2 3 4

# =========================================================
# Phase 5: 门控网络训练 + 动态融合 (课题核心创新)
# 门控输入 = Layer2 OOF probs (非 Layer1!)
# =========================================================
foreach ($fold in 0..4) {
  python scripts/train/train_gating.py `
    --exp configs/2d/exp/exp_msd_task03_liver.yaml `
    --gating-config configs/2d/gating.yaml `
    --models configs/2d/models.yaml `
    --fold $fold --gpus 0,1 --skip-if-done
}

# 门控动态融合推理 + 可视化权重
foreach ($fold in 0..4) {
  python scripts/inference/gating_inference.py `
    --exp configs/2d/exp/exp_msd_task03_liver.yaml `
    --gating-config configs/2d/gating.yaml `
    --models configs/2d/models.yaml `
    --fold $fold --save-weights
}
```

### 断点续训 / 意外中断恢复

```powershell
# 从最后 checkpoint 继续
python scripts/train/train_2d_experts.py ... --resume last

# 跳过已有 best.pt 的专家
python scripts/train/train_2d_experts.py ... --skip-if-done
```

### 输出目录结构

```
runs/segmoe_2d_msd03/
├── checkpoints/
│   ├── layer1/fold{0-4}/{nnunet-2d,swinunetr-2d,segresnet-2d}/best.pt
│   ├── layer2/fold{0-4}/{nnunet-2d,swinunetr-2d,segresnet-2d}/best.pt
│   └── gating/fold{0-4}/best.pt          # 门控网络权重
├── cache/
│   ├── oof/layer1/oof_manifest.jsonl    # L1 OOF manifest (含 TTA 标记)
│   ├── oof/layer1/fold{0-4}/*.npz      # L1 OOF 概率图 [K,M,H,W]
│   ├── oof/layer2/oof_manifest_layer2.jsonl  # L2 OOF manifest (门控输入)
│   └── oof/layer2/fold{0-4}/*.npz      # L2 OOF 概率图 [K,M,H,W]
├── results/
│   ├── metrics_*.csv                    # 逐方法指标
│   └── gating/fold{0-4}/               # 门控融合预测 + 权重图
│       ├── *.npz                        # fused + pred
│       ├── weight_maps/*.npz            # per-expert 空间权重
│       └── metrics.json                 # 门控融合 Dice 指标
├── tables/
│   ├── table1_single_experts.csv        # 单专家结果
│   ├── table2_ensemble_methods.csv      # 融合方法结果
│   └── expert_weights.json              # OLE/WE-CLPSO 融合权重
└── logs/                                # TensorBoard 日志
```

### 双卡 5090 关键优化说明

| 优化项 | 配置 | 理由 |
|--------|------|------|
| BF16 混合精度 | `amp.dtype: bfloat16` | Blackwell 原生 Tensor Core 支持，动态范围=FP32，医学分割 Dice Loss 不会 NaN |
| 等效 bs=64 | `bs=32 × grad_accum=2` | 类别极不平衡（肿瘤 <1%），大 batch 采到更多前景，梯度更稳定 |
| Warmup 15 epochs | `scheduler.warmup_epochs: 15` | 大 batch 初始梯度方差大，防止训练发散 |
| Weight decay 0.05 | `optimizer.weight_decay: 0.05` | 大等效 batch 容易过拟合，较强 L2 正则化帮助泛化 |
| 梯度检查点 | `use_checkpoint: true` (SwinUNETR) | 用计算换显存，SwinUNETR 62M 参数仍可保持大 batch |

## CLI 参数说明

### 通用训练参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--gpus` | GPU ID 列表 | 所有可用 GPU |
| `--amp` | 强制开启 AMP | 按 yaml 配置 |
| `--num-workers` | DataLoader 进程数 | Windows=2, Linux=4 |
| `--grad-accum` | 梯度累积步数 | 1 |
| `--seed` | 随机种子 | 42 |
| `--resume` | 续训模式 (none/last/best/path) | none |
| `--skip-if-done` | 跳过已训练完的专家 | False |

### OOF 生成参数 (`generate_layer1_oof.py`)

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--batch-size` | 推理批大小 (P1 优化) | 32 |
| `--tta` | 开启 TTA (P2: 水平+垂直翻转三路平均) | False |

### Layer2 训练参数 (`train_layer2.py`)

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--no-pretrain` | 禁用 B1 Layer1→Layer2 权重迁移 | False (启用迁移) |
| `--no-uncertainty` | 禁用 B5 不确定性通道 | False (启用不确定性) |

## 三专家架构

| 角色 | 名称 | 类型 | 实现 | 训练方式 |
|------|------|------|------|----------|
| Expert A (CNN) | `nnunet-2d` | nnUNet v2 PlainConvUNet | `dynamic_network_architectures` | **官方 nnUNet** (1000 epochs, SGD, PolyLR, 深度监督) |
| Expert B (Transformer) | `swinunetr-2d` | Swin-UNetR 2D | `monai.networks.nets.SwinUNETR` | **官方 MONAI Recipe** (300 epochs, AdamW 1e-4, WarmupCosine) |
| Expert C (ResEncoder) | `segresnet-2d` | SegResNetDS 2D | `monai.networks.nets.SegResNetDS` | **官方 MONAI Auto3DSeg** (300 epochs, AdamW 2e-4, DeepSupervision) |

所有专家统一输出 logits shape: `[B, M, H, W]`（M = 类别数）

## Layer2 架构与优化

### 数据流

```
原始图像 [B, 3, H, W]
    ↓
Layer1 三专家各自推理 (OOF: Out-of-Fold)
    ↓
L1 OOF 概率 [K, M, H, W] = [3, 3, 256, 256]  (float16, .npz)
    ↓
Channel concat: [image(3) + L1_oof_probs(9) + entropy(1) + disagreement(3)] = [16, H, W]
    ↓
Layer2 三专家 (同架构, in_channels=16, 预训练权重迁移)
    ↓
Layer2 OOF 推理 (Out-of-Fold, 同样无泄漏)
    ↓
L2 OOF 概率 [K, M, H, W] = [3, 3, 256, 256]
    ↓
Patch 切分 (64×64, stride=32) → 门控网络 → 动态融合
    ↓
融合预测 [M, H, W] + 门控权重可视化 [K, H, W]
```

### Layer2 输入通道 (B5: 不确定性通道)

| 通道 | 维度 | 来源 |
|------|------|------|
| 原始图像 | [3, H, W] | RGB 输入 |
| OOF 概率 | [K×M, H, W] = [9, H, W] | K=3 专家 × M=3 类别 softmax |
| Entropy map | [1, H, W] | $-\sum_k p_k \log p_k / \log M$，归一化到 [0,1] |
| Disagreement | [M, H, W] = [3, H, W] | 各类别上 K 专家的 std |
| **总计** | **[16, H, W]** | |

### Layer2 损失函数 (B3: Boundary Loss)

$$\mathcal{L} = \underbrace{\mathcal{L}_{\text{CE}}}_{\text{ce\_weight}} + \underbrace{\mathcal{L}_{\text{Dice}}}_{\text{dice\_weight}} + \underbrace{\lambda_b \sum_{c} \langle p_c, \phi_c \rangle}_{\text{boundary\_weight}}$$

其中 $\phi_c$ 为类别 $c$ 的有符号距离变换（EDT），$\lambda_b = 0.5$。

## 融合方法

| 方法 | 类型 | 权重粒度 | 说明 |
|------|------|---------|------|
| **Gating** (课题核心) | 学习型动态 | per-patch per-expert `[K]` | ConvNet 门控网络, 空间自适应, 端到端训练 |
| OLE | 解析求解 | per-class per-expert `[K,M]` | Bounded LSQ, 全局静态 |
| DT | 模板匹配 | per-class `[M,K,M]` | Decision Template, 欧氏距离 |
| WE-CLPSO | 元启发搜索 | per-class per-expert `[K,M]` | 粒子群优化, 全局静态 |

## 测试

```bash
pytest tests/ -v
python scripts/utils/smoke_test_train.py
```

## Windows 专项说明

> 本项目使用 DataParallel 替代 DDP。Windows 上 torchrun 因 libuv 库缺失
> 无法正常工作（PyTorch 已知限制），DataParallel 在 Windows 上 100% 稳定。

## 3D 扩展规划

详见 [docs/ROADMAP_3D.md](docs/ROADMAP_3D.md)

## Citation

```bibtex
@article{dang2024two,
  title={Two-layer Ensemble of Deep Learning Models for Medical Image Segmentation},
  author={Dang, et al.},
  journal={Springer},
  year={2024}
}
```

## License

MIT License

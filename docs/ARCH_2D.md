# Seg-MoE 2D Pipeline 架构文档

## Pipeline 总览

```
数据准备 → Splits → 训练3专家 → 缓存概率图 → 融合评估
```

### 阶段说明

| 阶段 | 入口脚本 | 说明 |
|------|---------|------|
| 数据准备 | `scripts/data/prepare_msd.py` | NIfTI → 2D PNG 切片 |
| Splits | `scripts/data/make_splits.py` | 5-fold 交叉验证索引 |
| nnUNet 官方训练 | `nnUNetv2_train` + `scripts/nnunet/import_nnunet_weights.py` | 官方流程训练 + 权重导入 |
| SwinUNETR 官方训练 | `scripts/monai/train_swinunetr_official.py` + `import_swinunetr_weights.py` | MONAI Recipe 训练 + 权重导入 |
| SegResNet 官方训练 | `scripts/monai/train_segresnet_official.py` + `import_segresnet_weights.py` | MONAI Auto3DSeg Recipe 训练 + 权重导入 |
| 训练专家 | `scripts/train/train_2d_experts.py` | Layer1 验证 (所有专家已官方训练, --skip-if-done 跳过) |
| 缓存概率 | `scripts/inference/cache_probs.py` | 推理每个专家，存 `[K,M,H,W]` npz |
| OOF 概率 | `scripts/inference/generate_layer1_oof.py` | 生成 Layer1 OOF 概率图 |
| Layer2 训练 | `scripts/train/train_layer2.py` | 基于 I* = 原图 + Layer1 概率图 |
| 评估融合 | `scripts/eval/eval_methods.py` | Layer1/2 + Mean/MV/OLE/DT/WE + Gating + Wilcoxon |
| 导出表格 | `scripts/eval/export_tables.py` | 汇总 CSV / LaTeX / 统计显著性 |
| 导出权重 | `scripts/eval/export_weights.py` | 融合器权重导出 |
| 可视化 | `scripts/eval/visualize_overlay.py` | 预测叠加原图 |
| Sanity | `scripts/utils/sanity_experts_2d.py` | 随机张量前向检查 shape |

## 三专家组合

| 角色 | 名称 | 类型 | 实现 | 训练方式 |
|------|------|------|------|
| Expert A (CNN) | `nnunet-2d` | nnUNet v2 PlainConvUNet | `dynamic_network_architectures` | **官方 nnUNet** (1000 ep, SGD, PolyLR) |
| Expert B (Transformer) | `swinunetr-2d` | Swin-UNetR 2D | `monai.networks.nets.SwinUNETR` | **官方 MONAI Recipe** (300 ep, AdamW 1e-4, WarmupCosine) |
| Expert C (ResEncoder) | `segresnet-2d` | SegResNetDS 2D | `monai.networks.nets.SegResNetDS` | **官方 MONAI Auto3DSeg** (300 ep, AdamW 2e-4, DeepSupervision) |

三专家统一输出 logits shape: `[B, M, H, W]`（M = 类别数）

## 关键文件路径

### 配置

| 文件 | 用途 |
|------|------|
| `configs/2d/models.yaml` | **三专家配置** |
| `configs/2d/training.yaml` | 基础训练超参 |
| `configs/2d/training_dual_5090.yaml` | 双卡 AMP 训练超参 |
| `configs/2d/augs.yaml` | Albumentations 增强 |
| `configs/2d/datasets/msd_task03_liver.yaml` | 数据集定义 |
| `configs/2d/exp/exp_msd_task03_liver.yaml` | 实验入口配置 |
| `configs/2d/debug.yaml` | Debug 快速覆写（epochs=2, samples=16） |

### 源码

| 模块 | 文件 | 说明 |
|------|------|------|
| 统一模型工厂 | `src/seg_moe/models/factory_2d.py` | `build_expert()` / `list_experts()` / `expert_name()` |
| 3D 专家工厂 | `src/seg_moe/models/experts/factory.py` | `ExpertFactory` 3D 专家注册与构建 |
| ~~MONAI SOTA~~ | ~~`src/seg_moe/models/factory_sota.py`~~ | (已废弃, 功能已合并至 factory_2d 和 experts/factory) |
| nnUNet wrapper | `src/seg_moe/models/wrappers/nnunet_wrapper.py` | PlainConvUNet 封装 |
| 训练引擎 | `src/seg_moe/training/engine.py` | DP 多卡 + AMP + checkpoint |
| 数据集 | `src/seg_moe/data/dataset_2d.py` | 2D PNG 分割数据集 |
| 指标 | `src/seg_moe/evaluation/metrics_2d.py` | Per-class Dice/IoU/HD95/NSD/ASD/Sens/Prec |
| 融合器 | `src/seg_moe/combiners/` | MajorityVoting / OLE / DT / WE-CLPSO |

### 目录约定

```
runs/<exp_name>/
├── checkpoints/
│   └── layer1/
│       └── fold0/
│           ├── nnunet-2d/           best.pt / last.pt
│           ├── swinunetr-2d/       best.pt / last.pt
│           └── segresnet-2d/       best.pt / last.pt
├── cache/
│   ├── layer1_probs/
│   │   └── msd_task03_liver/
│   │       ├── train_fold0/        {sample_id}.npz  [K,M,H,W]
│   │       └── val_fold0/          {sample_id}.npz  [K,M,H,W]
│   └── oof/
│       └── layer1/                 OOF 概率图
├── results/
│   ├── metrics_*.csv               per-method 聚合指标
│   ├── metrics_per_sample_*.csv    per-sample 指标 (统计检验用)
│   └── significance_*.csv          Wilcoxon signed-rank p-values
├── tables/
│   ├── table1_L1_experts.csv       Layer1 单专家
│   ├── table2_L2_experts.csv       Layer2 单专家
│   ├── table3_ensemble_methods.csv 集成方法
│   ├── table4_all_methods.csv      全方法汇总
│   ├── table_summary_mean_std.csv  5-fold mean±std
│   ├── table_significance.csv      统计显著性
│   └── expert_weights.json
└── logs/
    └── layer1/
        └── fold0/                  TensorBoard
```

## 脚本分组

### data（数据准备）
- `scripts/data/prepare_msd.py` — NIfTI 3D → 2D 切片
- `scripts/data/prepare_nifti_slices.py` — 通用 NIfTI 切片
- `scripts/data/make_splits.py` — 生成 5-fold splits
- `scripts/data/check_labels.py` — 标签审计

### nnunet（nnUNet 官方训练集成）
- `scripts/nnunet/setup_nnunet_task.py` — 数据集转换 + 预处理
- `scripts/nnunet/import_nnunet_weights.py` — nnUNet 官方权重导入

### monai（MONAI 官方训练集成）
- `scripts/monai/train_swinunetr_official.py` — SwinUNETR 官方 MONAI Recipe 训练
- `scripts/monai/import_swinunetr_weights.py` — SwinUNETR 官方权重导入
- `scripts/monai/train_segresnet_official.py` — SegResNet 官方 MONAI Auto3DSeg Recipe 训练
- `scripts/monai/import_segresnet_weights.py` — SegResNet 官方权重导入

### train（训练）
- `scripts/train/train_2d_experts.py` — Layer1 验证 (所有专家已官方训练, --skip-if-done 跳过)
- `scripts/train/train_layer2.py` — Layer2 训练
- `scripts/train/train_expert_3d.py` — 3D 专家训练

### inference（缓存概率图）
- `scripts/inference/cache_probs.py` — 推理存 npz
- `scripts/inference/generate_layer1_oof.py` — OOF 概率图生成

### eval（评估与导出）
- `scripts/eval/eval_methods.py` — L1/L2 单模型 + 集成 + Gating + Wilcoxon 统计检验
- `scripts/eval/export_tables.py` — 多表汇总 (per-class / significance / mean±std)
- `scripts/eval/export_weights.py` — 融合器权重导出
- `scripts/eval/visualize_overlay.py` — 分割可视化

### utils（工具）
- `scripts/utils/sanity_experts_2d.py` — 2D 专家 shape 检查
- `scripts/utils/sanity_experts.py` — 3D 专家 shape 检查
- `scripts/utils/validate.py` — 环境/数据/模型验证
- `scripts/utils/smoke_test_train.py` — GPU 训练 smoke test

## Quick Reference 命令

```bash
# 0. Sanity check
python scripts/utils/sanity_experts_2d.py --models configs/2d/models.yaml

# 1. 训练三专家
python scripts/train/train_2d_experts.py \
  --exp configs/2d/exp/exp_msd_task03_liver.yaml \
  --training configs/2d/training.yaml \
  --models configs/2d/models.yaml \
  --augs configs/2d/augs.yaml \
  --fold 0 --layer layer1

# 2. 缓存概率图
python scripts/inference/cache_probs.py \
  --exp configs/2d/exp/exp_msd_task03_liver.yaml \
  --models configs/2d/models.yaml \
  --layer layer1 --fold 0

# 3. 评估
python scripts/eval/eval_methods.py \
  --exp configs/2d/exp/exp_msd_task03_liver.yaml \
  --training configs/2d/training.yaml \
  --models configs/2d/models.yaml
```

# Seg-MoE: Two-Layer Ensemble for Medical Image Segmentation

实现 Dang et al. (2024) "Two-layer Ensemble of Deep Learning Models for Medical Image Segmentation" 的两层集成框架。

**支持数据集**：MSD Task03 Liver | MSD Task07 Pancreas | ACDC | BTCV

## 核心特性

- **两层 Stacking 架构**：Layer1 三专家 → 概率拼接 → Layer2 → 融合器
- **三专家组合**：nnUNet (CNN) + SwinUNETR (Transformer) + SegResNet (ResEncoder)
- **nnUNet 官方训练**：1000 epochs, SGD, PolyLR, 深度监督 — 完整复现官方性能
- **融合方法**：OLE / DT / WE-CLPSO 多种策略
- **多格式支持**：NIfTI (3D→2D切片) / PNG / JPEG / DICOM
- **Windows 原生兼容**：DataParallel 多卡训练，不依赖 torchrun/DDP
- **严格复现**：固定随机种子 + 5-fold 交叉验证

## 项目结构

```
configs/2d/
  ├── models.yaml                # 三专家配置 (nnUNet/SwinUNETR/SegResNet)
  ├── datasets/                  # 数据集配置
  ├── training.yaml              # 基础训练超参
  ├── training_dual_5090.yaml    # 双卡优化超参 (AMP+AdamW+Cosine)
  ├── augs.yaml                  # 数据增强
  ├── debug.yaml                 # Debug 快速覆写
  └── exp/                       # 实验入口配置
src/seg_moe/
  ├── models/factory_2d.py       # 统一模型工厂 (build_expert)
  ├── models/wrappers/           # nnUNet wrapper (支持 deep_supervision)
  ├── training/engine.py         # 训练引擎 (DP + AMP + checkpoint)
  ├── combiners/                 # OLE / DT / WE-CLPSO 融合器
  ├── data/                      # 多格式数据加载
  └── evaluation/                # Dice / IoU / HD / MAD 指标
scripts/
  ├── data/                      # 数据准备 (prepare, splits, labels)
  ├── train/                     # 训练 (2D/3D experts, layer2)
  ├── inference/                 # 推理缓存 (cache_probs, OOF)
  ├── eval/                      # 评估导出 (metrics, tables, viz)
  ├── nnunet/                    # nnUNet 官方训练集成
  │   ├── setup_nnunet_task.py   # 数据集转换 + 预处理
  │   └── import_nnunet_weights.py  # 官方权重导入
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

本项目的训练分为两部分:
- **nnUNet**: 使用 **官方 nnUNet v2 训练流程** (1000 epochs, SGD + PolyLR, 深度监督)
- **SwinUNETR / SegResNet**: 使用 **Seg-MoE 自定义流程** (300 epochs, AdamW + Cosine, BF16)

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

#### Phase 3: SwinUNETR + SegResNet 训练 & 融合

> **训练参数概要** (SwinUNETR / SegResNet):
> BF16 混合精度 | 全局 bs=32 (16/卡) | 梯度累积 2 步 → 等效 bs=64
> AdamW (wd=0.05) | lr=4e-4 + Cosine warmup 15 epochs | 300 epochs

```powershell
# ---- Step 5: Layer1 训练 SwinUNETR + SegResNet ----
# --skip-if-done 会自动跳过已导入的 nnUNet (best.pt 已存在)
python scripts/train/train_2d_experts.py `
  --exp configs/2d/exp/exp_msd_task03_liver.yaml `
  --training configs/2d/training_dual_5090.yaml `
  --models configs/2d/models.yaml `
  --augs configs/2d/augs.yaml `
  --fold 0 --layer layer1 --gpus 0,1 --skip-if-done

# ---- Step 6: 生成 Layer1 OOF 概率图 ----
python scripts/inference/generate_layer1_oof.py `
  --exp configs/2d/exp/exp_msd_task03_liver.yaml `
  --models configs/2d/models.yaml --which best

# ---- Step 7: Layer2 训练 (三专家均使用自定义流程) ----
# Layer2 输入 = 原图 + Layer1 OOF 概率拼接, nnUNet 也需重新训练
python scripts/train/train_layer2.py `
  --exp configs/2d/exp/exp_msd_task03_liver.yaml `
  --training configs/2d/training_dual_5090.yaml `
  --models configs/2d/models.yaml `
  --augs configs/2d/augs.yaml `
  --fold 0 --gpus 0,1

# ---- Step 8: 缓存概率图 + 评估 + 导出 ----
python scripts/inference/cache_probs.py `
  --exp configs/2d/exp/exp_msd_task03_liver.yaml `
  --models configs/2d/models.yaml `
  --layer layer1 --fold 0

python scripts/eval/eval_methods.py `
  --exp configs/2d/exp/exp_msd_task03_liver.yaml `
  --training configs/2d/training_dual_5090.yaml `
  --models configs/2d/models.yaml

python scripts/eval/export_tables.py --exp configs/2d/exp/exp_msd_task03_liver.yaml
python scripts/eval/export_weights.py `
  --exp configs/2d/exp/exp_msd_task03_liver.yaml `
  --models configs/2d/models.yaml
```

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
# Phase 3: SwinUNETR + SegResNet Layer1 训练 (nnUNet 自动跳过)
# =========================================================
foreach ($fold in 0..4) {
  python scripts/train/train_2d_experts.py `
    --exp configs/2d/exp/exp_msd_task03_liver.yaml `
    --training configs/2d/training_dual_5090.yaml `
    --models configs/2d/models.yaml --augs configs/2d/augs.yaml `
    --fold $fold --layer layer1 --gpus 0,1 --skip-if-done
}

# =========================================================
# Phase 4: OOF + Layer2 + 评估
# =========================================================
python scripts/inference/generate_layer1_oof.py `
  --exp configs/2d/exp/exp_msd_task03_liver.yaml `
  --models configs/2d/models.yaml --which best

foreach ($fold in 0..4) {
  python scripts/train/train_layer2.py `
    --exp configs/2d/exp/exp_msd_task03_liver.yaml `
    --training configs/2d/training_dual_5090.yaml `
    --models configs/2d/models.yaml --augs configs/2d/augs.yaml `
    --fold $fold --gpus 0,1 --skip-if-done
}

python scripts/eval/eval_methods.py `
  --exp configs/2d/exp/exp_msd_task03_liver.yaml `
  --training configs/2d/training_dual_5090.yaml `
  --models configs/2d/models.yaml

python scripts/eval/export_tables.py `
  --exp configs/2d/exp/exp_msd_task03_liver.yaml --folds 0 1 2 3 4
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
│   └── layer2/fold{0-4}/{nnunet-2d,swinunetr-2d,segresnet-2d}/best.pt
├── cache/oof/layer1/                    # OOF 概率图
├── results/metrics_*.csv                # 逐方法指标
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

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--gpus` | GPU ID 列表 | 所有可用 GPU |
| `--amp` | 强制开启 AMP | 按 yaml 配置 |
| `--num-workers` | DataLoader 进程数 | Windows=2, Linux=4 |
| `--grad-accum` | 梯度累积步数 | 1 |
| `--seed` | 随机种子 | 42 |
| `--resume` | 续训模式 (none/last/best/path) | none |
| `--skip-if-done` | 跳过已训练完的专家 | False |

## 三专家架构

| 角色 | 名称 | 类型 | 实现 | 训练方式 |
|------|------|------|------|----------|
| Expert A (CNN) | `nnunet-2d` | nnUNet v2 PlainConvUNet | `dynamic_network_architectures` | **官方 nnUNet** (1000 epochs, SGD, PolyLR, 深度监督) |
| Expert B (Transformer) | `swinunetr-2d` | Swin-UNetR 2D | `monai.networks.nets.SwinUNETR` | 自定义 (300 epochs, AdamW, Cosine) |
| Expert C (ResEncoder) | `segresnet-2d` | SegResNet 2D | `monai.networks.nets.SegResNet` | 自定义 (300 epochs, AdamW, Cosine) |

所有专家统一输出 logits shape: `[B, M, H, W]`（M = 类别数）

## 融合方法

- **OLE**: Optimal Linear Ensemble — 加权概率融合
- **DT**: Decision Template — 基于决策模板的融合
- **WE-CLPSO**: 粒子群优化权重融合

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

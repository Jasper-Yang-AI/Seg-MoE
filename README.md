# Seg_MoE: Two-Layer Ensemble for Medical Image Segmentation

实现 Dang et al. (Springer 2024) "Two-layer Ensemble of Deep Learning Models for Medical Image Segmentation" 的两层集成 2D pipeline，并为 3D patch 级动态门控融合预留接口。

说明：由于原论文部分 2D 数据集获取受限，本仓库当前默认适配并验证了以下替代数据集的端到端链路（3D NIfTI → 2D slices → 训练/融合/评估）：
- MSD Task03 Liver
- MSD Task07 Pancreas
- ACDC
- BTCV（当前这份 dump 经过审计为二值标签 {0,1}，见下文）

## 🔬 项目特点

- ✅ **论文方法复现**：9个专家模型 + 两层 stacking + 多种融合方法（OLE/DT/WE-CLPSO）
- ✅ **多格式支持**：PNG/JPEG + NIfTI/MetaImage/DICOM 等医学格式
- ✅ **两层集成架构**：Layer1专家 → 概率图拼接 → Layer2专家 → 融合器
- ✅ **严格可复现**：固定随机种子、5-fold交叉验证、详细日志
- ✅ **3D扩展接口**：预留patch级动态门控模块

## 📦 支持的数据格式

### 常规图像格式
- PNG, JPEG, BMP, TIFF

### 医学图像格式
- **NIfTI**: `.nii`, `.nii.gz` (3D volumes → 自动切片提取)
- **MetaImage**: `.mhd`, `.mha` (如 CAMUS 数据集)
- **DICOM**: `.dcm` (单帧或多帧)

### 3D数据处理
对于3D volume（如NIfTI/DICOM series），系统会：
1. 自动检测3D维度
2. 按指定轴（通常z轴）切片
3. 转换为2D slices用于当前pipeline
4. 保留原始3D metadata用于重建

## 🗂️ 项目结构

```
Seg_MoE/
├── configs/              # YAML配置驱动
│   ├── 2d/               # 2D pipeline 配置
│   │   ├── datasets/     # 数据集配置（MSD/ACDC/BTCV 等）
│   │   ├── models/       # 9个专家+融合器配置
│   │   ├── training/     # 训练超参
│   │   └── 3d/           # 3D扩展预留
├── data/
│   ├── raw/              # 原始数据（手动放置）
│   ├── processed/        # 统一格式
│   └── splits/           # 5-fold划分
├── src/seg_moe/          # 主代码包
│   ├── data/             # 数据加载（多格式支持）
│   ├── models/           # 专家模型
│   ├── combiners/        # 融合器（OLE/DT/WE-CLPSO）
│   ├── training/         # 训练流程
│   ├── evaluation/       # 评估指标
│   ├── gating/           # 3D门控预留
│   └── utils/            # 工具函数
├── scripts/              # 可执行脚本
└── tests/                # 单元测试
```

## 🚀 快速开始

### 1. 环境安装

```bash
# 创建虚拟环境（Python 3.10+）
conda create -n segmoe python=3.10
conda activate segmoe

# 安装依赖
cd C:\Users\XNAS\PycharmProjects\Seg_MoE
pip install -r requirements.txt

# 安装本项目包
pip install -e .
```

### 2. 数据准备

#### 2.1 放置/下载原始数据到 data/raw

本仓库期望你把数据放到以下结构（Windows 路径示例）：

```
data/raw/
├── MSD_Task03_Liver/
│   └── Task03_Liver/
│       ├── imagesTr/        # *.nii.gz
│       ├── labelsTr/        # *.nii.gz
│       └── dataset.json
├── MSD_Task07_Pancreas/
│   └── Task07_Pancreas/
│       ├── imagesTr/
│       ├── labelsTr/
│       └── dataset.json
├── ACDC/
│   └── ACDC/database/
│       ├── training/
│       └── testing/
└── BTCV_Synapse/
   ├── imagesTr/            # *.nii.gz
   ├── labelsTr/            # *_seg.nii.gz（常见命名）
   └── imagesTs/            # 可选
```

#### 2.2 运行 prepare + splits + 核验（强烈建议）

```bash
# ACDC
python scripts/prepare_acdc.py --config configs/2d/datasets/acdc.yaml
python scripts/make_splits.py --dataset-config configs/2d/datasets/acdc.yaml
python scripts/check_labels.py --dataset-config configs/2d/datasets/acdc.yaml --splits --sample 50
python scripts/visualize_overlay.py --dataset-config configs/2d/datasets/acdc.yaml --n 12

# MSD Task03 Liver（统一用 prepare_msd.py）
python scripts/prepare_msd.py --config configs/2d/datasets/msd_task03_liver.yaml
python scripts/make_splits.py --dataset-config configs/2d/datasets/msd_task03_liver.yaml
python scripts/check_labels.py --dataset-config configs/2d/datasets/msd_task03_liver.yaml --splits --sample 50
python scripts/visualize_overlay.py --dataset-config configs/2d/datasets/msd_task03_liver.yaml --n 12

# MSD Task07 Pancreas
python scripts/prepare_msd.py --config configs/2d/datasets/msd_task07_pancreas.yaml
python scripts/make_splits.py --dataset-config configs/2d/datasets/msd_task07_pancreas.yaml
python scripts/check_labels.py --dataset-config configs/2d/datasets/msd_task07_pancreas.yaml --splits --sample 50
python scripts/visualize_overlay.py --dataset-config configs/2d/datasets/msd_task07_pancreas.yaml --n 12

# 其他 MSD Task：复制 configs/2d/datasets/msd_template.yaml，按注释修改后运行
# python scripts/prepare_msd.py --config configs/2d/datasets/msd_taskXX_xxx.yaml

# BTCV
python scripts/prepare_btcv_synapse.py --config configs/2d/datasets/btcv_synapse.yaml
python scripts/make_splits.py --dataset-config configs/2d/datasets/btcv_synapse.yaml
python scripts/check_labels.py --dataset-config configs/2d/datasets/btcv_synapse.yaml --splits --sample 50
python scripts/visualize_overlay.py --dataset-config configs/2d/datasets/btcv_synapse.yaml --n 12
```

### 3. Debug验证（快速测试pipeline）

```bash
# 小规模快速测试（少量数据+少量 epoch），验证训练/评估/缓存主链路
# 建议先选 ACDC 或 MSD 的单个数据集跑 fold0
python scripts/train_2d_experts.py --exp configs/2d/experiment.yaml --training configs/2d/training.yaml --models configs/2d/models.yaml --augs configs/2d/augs.yaml --debug configs/2d/debug.yaml --fold 0 --layer layer1 --dataset-config configs/2d/datasets/acdc.yaml
```

### 4. 训练9个专家（Layer1）

```bash
# 单数据集训练所有专家（默认 5-fold）
python scripts/train_2d_experts.py --exp configs/2d/experiment.yaml --training configs/2d/training.yaml --models configs/2d/models.yaml --augs configs/2d/augs.yaml --layer layer1 --dataset-config configs/2d/datasets/acdc.yaml

# UNet++ baseline（强单模型）
python scripts/train_unetpp.py --exp configs/2d/experiment.yaml --training configs/2d/training.yaml --models configs/2d/models.yaml --augs configs/2d/augs.yaml --fold 0 --dataset-config configs/2d/datasets/acdc.yaml
```

### 5. 训练Layer2（基于I*）

```bash
# 1) 生成 layer1 概率缓存（float16, npz）
python scripts/cache_probs.py --exp configs/2d/experiment.yaml --models configs/2d/models.yaml --layer layer1 --dataset-config configs/2d/datasets/acdc.yaml

# 2) 基于 I* (image + layer1 probs) 训练 layer2 专家
python scripts/train_layer2.py --exp configs/2d/experiment.yaml --training configs/2d/training.yaml --models configs/2d/models.yaml --augs configs/2d/augs.yaml --dataset-config configs/2d/datasets/acdc.yaml

# 3) 生成 layer2 概率缓存（用于 proposed_2layer 最终融合与评估）
python scripts/cache_probs.py --exp configs/2d/experiment.yaml --models configs/2d/models.yaml --layer layer2 --dataset-config configs/2d/datasets/acdc.yaml
```

### 6. 训练融合器并评估

```bash
# 评估单模型/融合/两层方法，并输出每数据集每方法的 CSV
python scripts/eval_methods.py --exp configs/2d/experiment.yaml --training configs/2d/training.yaml --models configs/2d/models.yaml --dataset-config configs/2d/datasets/acdc.yaml

# 导出融合权重表（Table 6 风格）
python scripts/export_weights.py --exp configs/2d/experiment.yaml
```

### 7. 导出论文表格

```bash
# 生成论文风格的结果表格（Table 2-6）
python scripts/export_tables.py --exp configs/2d/experiment.yaml

# 输出：
# - table2_single_models.csv       (9个单模型结果)
# - table3_ensemble_methods.csv    (OLE/DT/WE-CLPSO对比)
# - table4_proposed_vs_baselines.csv
# - table5_cross_dataset.csv
# - table6_weights.csv             (融合权重)
```

## 🧾 复现关键默认假设（可配置）

### Label 映射（默认）

以 `configs/2d/datasets/*.yaml` 为准。

- MSD Task03 Liver：常见为 0=background, 1=liver, 2=tumor
- MSD Task07 Pancreas：常见为 0=background, 1=pancreas, 2=tumor
- ACDC：常见为 0=background, 1=RV, 2=myocardium, 3=LV
- BTCV：不同 release 的 label id 差异较大。你当前这份 dump 审计结果为二值标签 {0,1}，因此配置默认 num_classes=2。

强烈建议每次 prepare 后都跑：
`python scripts/check_labels.py --dataset-config <DATASET_YAML> --splits --sample 50`

### Hausdorff / MAD(ASD)（2D）

- 边界：`skimage.segmentation.find_boundaries(mode="outer")`
- 距离：欧氏距离最近邻（对称）
- MAD：对称平均最近边界距离（ASD）
- HD：默认 full Hausdorff；可选输出 HD95
- spacing：默认像素单位；若从 medical header 读到 spacing，则按 mm 输出 HD/MAD

### 固定 test 数据集的 5-fold

固定 test 的数据集仅在 train 内做 5-fold，最终在官方 test 汇总报告。

## 📊 数据集配置

### 1. MSD Task03 Liver
- **任务**: 3类分割（常见：background/liver/tumor）
- **格式**: 3D NIfTI (`.nii.gz`) → 自动切片为 2D
- **配置**: `configs/2d/datasets/msd_task03_liver.yaml`

### 2. MSD Task07 Pancreas
- **任务**: 3类分割（常见：background/pancreas/tumor）
- **格式**: 3D NIfTI → 自动切片
- **配置**: `configs/2d/datasets/msd_task07_pancreas.yaml`

### 3. ACDC
- **任务**: 4类分割（常见：background/RV/myocardium/LV）
- **格式**: 3D NIfTI → 自动切片
- **配置**: `configs/2d/datasets/acdc.yaml`

### 4. BTCV
- **任务**: 依数据版本而定（你当前这份为二值标签 {0,1}）
- **格式**: 3D NIfTI → 自动切片
- **配置**: `configs/2d/datasets/btcv_synapse.yaml`

## 🔧 核心组件

### 9个专家模型（segmentation_models_pytorch）

| 架构 | Backbone | 预训练 |
|------|----------|--------|
| UNet | VGG16, ResNet34, ResNet101 | ImageNet |
| LinkNet | VGG16, ResNet34, ResNet101 | ImageNet |
| FPN | VGG16, ResNet34, ResNet101 | ImageNet |

### 融合方法

1. **OLE-9** (One-Layer Ensemble)
   - Weight-based combining
   - Per-class weights: $w_m \in [0,1]$
   - 优化：LSQ bounded / SGD

2. **DT-9** (Decision Template)
   - 为每类构建template（训练样本预测均值）
   - 推理：计算与各template的距离

3. **WE-CLPSO** (Weighted Ensemble with CLPSO)
   - 粒子群优化（Comprehensive Learning PSO）
   - 搜索最优融合权重

4. **Proposed Two-Layer**
   - Layer1: 9专家 → 概率图
   - I*: 原图 + 9×M概率图拼接
   - Layer2: 9专家基于I*
   - Final: 融合器

### 评估指标

- **Dice Coefficient**: $\frac{2|X \cap Y|}{|X| + |Y|}$
- **IoU**: $\frac{|X \cap Y|}{|X \cup Y|}$
- **Hausdorff Distance**: $\max(h(X,Y), h(Y,X))$
- **MAD (ASD)**: 对称平均表面距离

## 🧪 测试

```bash
# 运行所有测试
pytest tests/ -v
```

## 📈 实验复现指南

### Debug 快速验证（强烈建议先跑通）

下面是一套“单数据集/单fold/极少样本/少 epoch”的 smoke run，用于验证环境、数据、训练、缓存与评估链路都能跑通：

```bash
# 0) 建议先输出环境报告（可选）
python scripts/env_report.py

# 1) 先 prepare + splits（以 ACDC 为例）
python scripts/prepare_acdc.py --config configs/2d/datasets/acdc.yaml
python scripts/make_splits.py --dataset-config configs/2d/datasets/acdc.yaml
python scripts/check_labels.py --dataset-config configs/2d/datasets/acdc.yaml --splits --sample 50
python scripts/visualize_overlay.py --dataset-config configs/2d/datasets/acdc.yaml --n 12

# 2) layer1: 9 专家（debug 配置会把 epochs/samples 压到很小）
python scripts/train_2d_experts.py --exp configs/2d/experiment.yaml --training configs/2d/training.yaml --models configs/2d/models.yaml --augs configs/2d/augs.yaml --debug configs/2d/debug.yaml --fold 0 --layer layer1 --dataset-config configs/2d/datasets/acdc.yaml

# 3) cache layer1 probs
python scripts/cache_probs.py --exp configs/2d/experiment.yaml --models configs/2d/models.yaml --layer layer1 --fold 0 --which best --skip-existing --dataset-config configs/2d/datasets/acdc.yaml

# 4) layer2（需要 layer1 cache）
python scripts/train_layer2.py --exp configs/2d/experiment.yaml --training configs/2d/training.yaml --models configs/2d/models.yaml --augs configs/2d/augs.yaml --fold 0 --dataset-config configs/2d/datasets/acdc.yaml

# 5) cache layer2 probs
python scripts/cache_probs.py --exp configs/2d/experiment.yaml --models configs/2d/models.yaml --layer layer2 --fold 0 --which best --skip-existing --dataset-config configs/2d/datasets/acdc.yaml

# 6) eval + export
python scripts/eval_methods.py --exp configs/2d/experiment.yaml --training configs/2d/training.yaml --models configs/2d/models.yaml --fold 0 --dataset-config configs/2d/datasets/acdc.yaml
python scripts/export_tables.py --exp configs/2d/experiment.yaml
python scripts/export_weights.py --exp configs/2d/experiment.yaml
```

### 单卡全量复现的调度建议（多天级别）

全量设置（5 数据集 × 9 专家 × 5-fold × 300 epoch + layer2）在单卡上通常是“多天”级别。建议按下面方式分批跑：

- **按数据集逐个跑**：每次只跑一个数据集，避免混乱。
   - 推荐为每个数据集拷贝一份 `configs/2d/experiment.yaml`，改 `exp_name` 和 `dataset.config`，例如 `configs/2d/exp_acdc.yaml`、`configs/2d/exp_msd_task03.yaml`、`configs/2d/exp_msd_task07.yaml`、`configs/2d/exp_btcv.yaml`。
   - 或者在训练/缓存脚本上用 `--dataset-config ...` 覆盖 `exp.dataset.config`（适合批处理/调度脚本）。

- **按 fold 顺序跑**：`fold=0..4` 逐个完成，每个 fold 的典型顺序是：
   1) `train_2d_experts.py --layer layer1 --fold k`
   2) `cache_probs.py --layer layer1 --fold k --skip-existing`
   3) `train_layer2.py --fold k`
   4) `cache_probs.py --layer layer2 --fold k --skip-existing`
   5) `eval_methods.py --fold k`

- **断点续训**：训练脚本支持 `--resume last`（每个专家从对应 `last.pt` 继续）；也支持 `--skip-if-done` 跳过已完成（`best.pt` 已存在）的专家。
   - 示例：`python scripts/train_2d_experts.py ... --fold 2 --layer layer1 --resume last --skip-if-done`

- **缓存断点**：`cache_probs.py` 支持 `--skip-existing`，中断后重跑不会重复写已有 `.npz`。

## 🔬 关键假设与实现细节

### Label 映射
- 以各数据集 YAML 的 `task.num_classes` / `task.label_map` 为准。
- 每次 prepare 后用 `check_labels.py` 做强制核验，并用 `visualize_overlay.py` 做肉眼抽查。

### Hausdorff Distance
- 默认：Full HD（非HD95）
- 可选：HD95（95th percentile）
- 边界提取：形态学操作
- 距离：欧氏（像素单位，假设spacing=1mm）

### 可复现性
- 全局seed=42
- `torch.backends.cudnn.deterministic=True`
- 保存所有随机状态

说明：本仓库把关键假设集中写在本 README 的“复现关键默认假设（可配置）”部分，并尽量做到可配置（dataset YAML / debug overrides）。

（注：本仓库当前 scripts 目录不包含 CAMUS submission 导出器；如你需要 CAMUS server 对齐/提交导出，我可以按官方提交规范再补回相应脚本与 README 指南。）

## 🚧 3D扩展规划

当前实现为2D pipeline，未来扩展计划：

1. **3D数据支持**
   - 完整3D volume训练（而非切片）
   - 3D UNet/VNet架构

2. **Patch级动态门控**
   - 对3D patches分别预测
   - 动态权重：$w_k(x) = \text{Gating}(x)$
   - 接口已预留：`src/seg_moe/gating/patch_gating_3d.py`

3. **3D评估指标**
   - 3D Hausdorff
   - Volume-based metrics

详见：[docs/ROADMAP_3D.md](docs/ROADMAP_3D.md)

## 📝 Citation

如使用本代码复现论文，请引用：

```bibtex
@article{dang2024two,
  title={Two-layer Ensemble of Deep Learning Models for Medical Image Segmentation},
  author={Dang, et al.},
  journal={Springer},
  year={2024}
}
```

## 📄 License

MIT License

## 🤝 贡献

欢迎提交Issue和PR！

## 📧 联系

如有问题请联系：[your-email]

---

**项目状态**: ✅ 可运行 | 📊 实验中 | 📄 论文复现

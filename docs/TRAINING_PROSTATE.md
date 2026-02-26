# Seg-MoE 前列腺数据完整训练流程

## 一、GPU 问题诊断

### 当前状态

| 项目 | 状态 |
|------|------|
| GPU 硬件 | 2× NVIDIA RTX 5090 (32GB, Blackwell, SM 12.0) |
| 驱动版本 | 591.86 (CUDA 13.1 capable) |
| PyTorch | 2.10.0+cu130 |
| NVML (管理 API) | ✅ 正常，能枚举 2 块 GPU |
| CUDA 驱动 API | ❌ `cuInit(0)` 崩溃 (access violation 0xC0000005) |
| CUDA 运行时 | ❌ `cudaGetDeviceCount()` 崩溃 |
| `torch.cuda.is_available()` | ❌ native crash |
| `loss.backward()` | ❌ autograd 引擎内部触发 CUDA 初始化导致崩溃 |

### 根因分析

**NVIDIA CUDA 驱动 (`nvcuda.dll`) 在 `cuInit(0)` 时发生 native access violation**。
这是驱动层面的问题，与 PyTorch 无关。PyTorch 的 autograd 引擎 (`_engine_run_backward`)
在执行反向传播时会触及 CUDA 调度，导致即使 CPU-only 的 `backward()` 也崩溃。

### 解决方案（按优先级）

1. **更新 NVIDIA 驱动**（推荐）
   - 访问 https://www.nvidia.com/download/index.aspx
   - 选择: GeForce RTX 5090 → Windows 11 → Game Ready Driver
   - 下载并安装最新版本
   - **重启电脑**

2. **干净重装驱动**
   - 下载 [DDU (Display Driver Uninstaller)](https://www.guru3d.com/files-details/display-driver-uninstaller-download.html)
   - 进入安全模式 → 运行 DDU 完全卸载 NVIDIA 驱动
   - 重启后安装最新驱动

3. **检查冲突进程**
   - nvidia-smi 显示 Veee.exe、HalshCloud.exe 等使用 GPU
   - 关闭这些程序后重试：
     ```powershell
     taskkill /f /im Veee.exe
     taskkill /f /im HalshCloud.exe
     python _check_gpu.py
     ```

4. **重启电脑**——有时僵尸 CUDA 进程会锁住驱动

### GPU 修复后验证

```powershell
python _check_gpu.py
Get-Content _gpu_result.txt
# 预期输出: CUDA available: True, GPU 0/1: NVIDIA GeForce RTX 5090
```

---

## 二、三个专家的官方训练流程

### 数据准备前置步骤

前列腺数据 (`D:\Dataset002_ProstateCrop_Seg`) 已经是 **nnUNet v2 格式**：

```
D:\Dataset002_ProstateCrop_Seg/
  imagesTr/  njmu_xxx_0000.nii.gz (T2w), _0001.nii.gz (ADC), _0002.nii.gz (DWI)
  labelsTr/  njmu_xxx.nii.gz
  imagesTs/  ...
  labelsTs/  ...
```

- 3364 个训练 case + 75 个测试 case
- 3 模态, 4 类别, shape (240, 240, 40)

**Step 0: 生成 2D PNG 切片（SwinUNETR / SegResNet 用）**

```powershell
python scripts/data/prepare_prostate.py --config configs/2d/datasets/prostate_local.yaml
# 输出: data/processed/prostate_local/images/*.png (RGB)
#       data/processed/prostate_local/masks/*.png
#       data/splits/prostate_local/index_all.jsonl
```

**Step 0.5: 生成交叉验证 splits**

```powershell
python scripts/data/make_splits.py --config configs/2d/datasets/prostate_local.yaml
# 输出: data/splits/prostate_local/splits_train5fold_testfixed.jsonl
```

---

### Expert 1: nnUNet (官方 CLI)

nnUNet 使用 **3D NIfTI 数据直接训练**（不用 2D PNG），通过官方 CLI 完成。

#### 1.1 数据设置

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

这会：
- 创建 `nnunet_data/nnUNet_raw/Dataset002_ProstateCrop_Seg/` → 数据目录的 junction
- 生成 `dataset.json`（通道名、标签、训练数量）
- 运行 `nnUNetv2_plan_and_preprocess -d 2` 进行数据指纹分析和预处理

#### 1.2 设置环境变量

```powershell
$env:nnUNet_raw = "D:\Seg-MoE\nnunet_data\nnUNet_raw"
$env:nnUNet_preprocessed = "D:\Seg-MoE\nnunet_data\nnUNet_preprocessed"
$env:nnUNet_results = "D:\Seg-MoE\nnunet_data\nnUNet_results"
```

#### 1.3 训练（官方 5 折）

```powershell
# 2D 配置训练 (每折 1000 epochs, SGD + PolyLR)
foreach ($fold in 0..4) {
    nnUNetv2_train 2 2d $fold --npz
}

# 或单折测试:
nnUNetv2_train 2 2d 0 --npz
```

**官方训练配置**:
| 参数 | 值 |
|------|-----|
| Optimizer | SGD (lr=0.01, momentum=0.99, nesterov) |
| Scheduler | PolyLR |
| Epochs | 1000 |
| Loss | DC + CE |
| 数据增强 | rotation, scaling, mirroring, gamma, noise |
| Batch size | 自动 (fingerprint-based) |

#### 1.4 导入权重到 Seg-MoE

```powershell
python scripts/nnunet/import_nnunet_weights.py `
    --nnunet-base nnunet_data `
    --dataset-id 2 `
    --config 2d `
    --folds 0 1 2 3 4 `
    --exp configs/2d/exp/exp_prostate_local.yaml
```

---

### Expert 2: SwinUNETR (MONAI 官方 recipe)

SwinUNETR 使用 **2D PNG 切片训练**，复现 Tang et al. CVPR 2022 官方策略。

#### 2.1 训练（官方 5 折）

```powershell
foreach ($fold in 0..4) {
    python scripts/monai/train_swinunetr_official.py `
        --exp configs/2d/exp/exp_prostate_local.yaml `
        --models configs/2d/models.yaml `
        --fold $fold `
        --gpus 0,1 `
        --epochs 300 --batch-size 16
}
```

**官方训练配置 (Tang et al. 2022)**:
| 参数 | 值 |
|------|-----|
| Optimizer | AdamW (lr=1e-4, weight_decay=1e-5) |
| Scheduler | WarmupCosine (warmup=50 epochs) |
| Loss | DiceCELoss (softmax, to_onehot_y) |
| Epochs | 300 |
| AMP | BFloat16 |
| 数据增强 | RandFlip, RandRotate90, RandScaleIntensity, RandShiftIntensity |

#### 2.2 导入权重

```powershell
foreach ($fold in 0..4) {
    python scripts/monai/import_swinunetr_weights.py `
        --source runs/swinunetr_official_prostate_local/fold$fold/best_model.pt `
        --exp configs/2d/exp/exp_prostate_local.yaml `
        --models configs/2d/models.yaml `
        --fold $fold
}
```

---

### Expert 3: SegResNet (MONAI Auto3DSeg recipe)

SegResNet 使用 **2D PNG 切片训练**，复现 MONAI Auto3DSeg 官方策略。

#### 3.1 训练（官方 5 折）

```powershell
foreach ($fold in 0..4) {
    python scripts/monai/train_segresnet_official.py `
        --exp configs/2d/exp/exp_prostate_local.yaml `
        --models configs/2d/models.yaml `
        --fold $fold `
        --gpus 0,1 `
        --epochs 300 --batch-size 16
}
```

**官方训练配置 (MONAI Auto3DSeg)**:
| 参数 | 值 |
|------|-----|
| Optimizer | AdamW (lr=2e-4, weight_decay=1e-5) |
| Scheduler | WarmupCosine (warmup=3 epochs, epoch-level) |
| Loss | DeepSupervisionLoss(DiceCELoss(squared_pred, batch)) |
| Deep supervision | dsdepth=2 |
| Epochs | 300 |
| AMP | BFloat16 |
| 数据增强 | RandAffine, RandFlip, RandGaussianSmooth, RandScaleIntensity, RandShiftIntensity, RandGaussianNoise |

#### 3.2 导入权重

```powershell
foreach ($fold in 0..4) {
    python scripts/monai/import_segresnet_weights.py `
        --source runs/segresnet_official_prostate_local/fold$fold/best_model.pt `
        --exp configs/2d/exp/exp_prostate_local.yaml `
        --models configs/2d/models.yaml `
        --fold $fold
}
```

---

## 三、数据预处理兼容性

### 前列腺数据如何适配三个专家？

| 专家 | 输入格式 | 预处理 | 兼容性 |
|------|---------|--------|--------|
| nnUNet | 3D NIfTI (_0000/_0001/_0002) | 官方 fingerprint-based | ✅ 数据已是 nnUNet 格式 |
| SwinUNETR | 2D RGB PNG (256×256) | per-modality percentile → RGB | ✅ `prepare_prostate.py` |
| SegResNet | 2D RGB PNG (256×256) | per-modality percentile → RGB | ✅ `prepare_prostate.py` |

### 关键设计

- **nnUNet**: 直接使用原始 3D NIfTI 数据，nnUNet 自己做数据预处理（spacing 重采样、z-score 归一化等）
- **SwinUNETR / SegResNet**: 使用 `prepare_prostate.py` 生成的 2D RGB PNG 切片
  - 每个模态独立 percentile clipping [p0.5, p99.5] → [0,255] uint8
  - 3 模态 stack 为 RGB PNG
  - 训练时 ImageNet normalize per-channel
  - **已更新脚本**：现在自动检测 RGB（多模态）/ 灰度（单模态），通过 `dataset_cfg["input"]["image_channels"]`

---

## 四、完整执行顺序

```
1. 修复 GPU 驱动 (见第一节)
2. prepare_prostate.py → 2D PNG 切片
3. make_splits.py → 交叉验证 splits
4. 并行训练三个专家:
   4a. nnUNet:     setup_nnunet_task.py → nnUNetv2_train (GPU 0)
   4b. SwinUNETR:  train_swinunetr_official.py (GPU 1)
   4c. SegResNet:  train_segresnet_official.py (GPU 0 or 1)
5. 导入权重: import_*_weights.py
6. 生成 Layer1 OOF: generate_layer1_oof.py
7. 训练 Layer2 / Gating
8. 评估: eval_methods.py
```

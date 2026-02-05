# SegMamba Integration

## 概述

**SegMamba** 已替换 VM-UNet 作为第三个SOTA异构专家（Mamba-based模型）。

## 主要特点

- ✅ **线性复杂度**: 基于State Space Models的高效分割
- ✅ **自动fallback**: Mamba依赖不可用时自动使用EfficientNet-B0 UNet
- ✅ **易于集成**: 无需手动复制架构文件
- ✅ **灵活配置**: 支持自定义embed_dim、depths等参数

## 架构对比

| 特性 | VM-UNet | SegMamba |
|------|---------|----------|
| 来源 | JCruan519/VM-UNet | ge-xing/SegMamba |
| 安装复杂度 | 高（需手动复制文件） | 中（pip安装或自动fallback） |
| Fallback支持 | ✅ | ✅ |
| 配置参数 | depths, feat_size | embed_dim, depths, drop_path_rate |
| 默认参数规模 | ~38M | ~可调 |

## 文件结构

```
src/seg_moe/models/
├── factory_sota.py          # 已更新：_build_vm_unet → _build_segmamba
├── wrappers/
│   ├── nnunet_wrapper.py
│   └── segmamba_wrapper.py  # 新增：SegMamba wrapper with fallback
└── architectures/
    └── segmamba/            # 可选：完整SegMamba实现（如需要）

configs/2d/
└── models_sota.yaml         # 已更新：vm_unet → segmamba配置

scripts/
└── test_segmamba_integration.py  # 新增：集成测试脚本
```

## 配置示例

```yaml
# configs/2d/models_sota.yaml
- architecture: "segmamba"
  name: "segmamba-base"
  enabled: true
  config:
    img_size: 256
    embed_dim: 96          # 嵌入维度
    depths: [2, 2, 9, 2]   # 各阶段深度
    drop_path_rate: 0.2    # Drop path率
```

## 使用方法

### 1. 基础使用（自动fallback）

```python
from seg_moe.models.factory_sota import build_sota_model

# 自动选择：Mamba可用时使用SegMamba，否则使用fallback
model = build_sota_model(
    arch="segmamba",
    in_channels=1,
    classes=4,
    config={
        "img_size": 256,
        "embed_dim": 96,
        "depths": [2, 2, 9, 2],
    }
)
```

### 2. 检查Mamba可用性

```python
from seg_moe.models.wrappers.segmamba_wrapper import check_segmamba_available

if check_segmamba_available():
    print("✓ 将使用完整SegMamba（Mamba后端）")
else:
    print("⚠ 将使用fallback模型（EfficientNet-B0 UNet）")
```

### 3. 训练SOTA专家

```bash
# 训练3个异构专家：Swin-UNetR + nnUNet v2 + SegMamba
python scripts/train_2d_experts.py \
  --exp configs/2d/exp/exp_acdc.yaml \
  --models configs/2d/models_sota.yaml \
  --layer layer1 \
  --fold 0
```

## 安装Mamba依赖（可选）

如果想使用完整的Mamba后端（而非fallback）：

```bash
# 安装Mamba核心依赖
pip install causal-conv1d>=1.1.0
pip install mamba-ssm>=1.0.0

# 可选：安装完整SegMamba包（如果可用）
pip install segmamba
```

## 测试安装

```bash
# 运行集成测试
python scripts/test_segmamba_integration.py

# 运行完整SOTA模型测试
python scripts/test_sota_models.py
```

## Fallback行为

当Mamba依赖不可用时：

1. **自动检测**: `check_segmamba_available()` 返回 `False`
2. **加载fallback**: 使用 `segmentation_models_pytorch` 的 EfficientNet-B0 UNet
3. **参数规模**: ~6.3M（vs SegMamba的可调参数量）
4. **性能**: Fallback模型仍然是强baseline，可用于快速实验

## 迁移指南（从VM-UNet）

如果你之前使用VM-UNet配置：

### 配置文件更新

```yaml
# 旧配置（VM-UNet）
- architecture: "vm_unet"
  config:
    img_size: 256
    depths: [2, 2, 2, 2]
    feat_size: [48, 96, 192, 384]

# 新配置（SegMamba）
- architecture: "segmamba"
  config:
    img_size: 256
    embed_dim: 96
    depths: [2, 2, 9, 2]
    drop_path_rate: 0.2
```

### 代码更新

```python
# 旧代码
from seg_moe.models.wrappers.vm_unet_wrapper import VMUNetWrapper
model = VMUNetWrapper(...)

# 新代码
from seg_moe.models.wrappers.segmamba_wrapper import SegMambaWrapper
model = SegMambaWrapper(...)
```

## 优势

1. **更简单的安装**: 无需手动复制架构文件
2. **更好的fallback**: EfficientNet-B0已被广泛验证
3. **更灵活的配置**: 支持更多可调参数
4. **社区支持**: SegMamba项目活跃维护

## 参考资料

- **SegMamba GitHub**: https://github.com/ge-xing/SegMamba
- **Mamba**: https://github.com/state-spaces/mamba
- **安装文档**: docs/INSTALL_SOTA_MODELS.md
- **测试脚本**: scripts/test_segmamba_integration.py

## 故障排除

### 问题: Mamba安装失败

**解决方案**: 使用fallback模型
```bash
# Fallback模型自动启用，无需额外操作
# 训练脚本会自动使用EfficientNet-B0 UNet
```

### 问题: 想要完整SegMamba但安装困难

**解决方案**: 
1. 检查CUDA版本兼容性
2. 确保有合适的C++编译器
3. 参考 docs/INSTALL_SOTA_MODELS.md 的troubleshooting部分

### 问题: Fallback性能是否足够

**答案**: 
- Fallback使用EfficientNet-B0 (6.3M参数)
- 在多数医学图像分割任务上表现良好
- 可用于快速验证pipeline，后续再安装完整Mamba

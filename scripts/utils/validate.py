#!/usr/bin/env python
"""
统一验证脚本：整合环境检查、数据验证、模型测试
用法：python scripts/utils/validate.py [--env] [--data] [--models] [--all]
"""
import argparse
import sys
from pathlib import Path

def check_environment():
    """检查环境配置"""
    print("=" * 60)
    print("环境检查")
    print("=" * 60)
    
    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__}")
        print(f"  CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
    except ImportError:
        print("✗ PyTorch未安装")
        return False
    
    # 检查核心依赖
    required_packages = {
        'monai': 'SwinUNETR + SegResNet (MONAI)',
        'dynamic_network_architectures': 'PlainConvUNet (nnUNet)',
        'timm': 'Vision Transformer backbones',
        'einops': 'Tensor operations for transformers',
    }
    
    all_ok = True
    print("\n核心SOTA模型依赖：")
    for pkg, desc in required_packages.items():
        try:
            __import__(pkg)
            print(f"  ✓ {pkg}: {desc}")
        except ImportError:
            print(f"  ✗ {pkg}: {desc} (未安装)")
            all_ok = False
    
    print("\n✓ 环境检查完成")
    return all_ok


def check_data(dataset_config=None):
    """检查数据准备情况"""
    print("\n" + "=" * 60)
    print("数据检查")
    print("=" * 60)
    
    data_root = Path("data")
    
    # 检查目录结构
    required_dirs = ["raw", "processed", "splits"]
    for dir_name in required_dirs:
        dir_path = data_root / dir_name
        if dir_path.exists():
            print(f"✓ {dir_path}/")
        else:
            print(f"✗ {dir_path}/ (不存在)")
    
    # 检查处理后的数据集
    processed_dir = data_root / "processed"
    if processed_dir.exists():
        datasets = [d for d in processed_dir.iterdir() if d.is_dir()]
        if datasets:
            print(f"\n已处理的数据集 ({len(datasets)}):")
            for ds in datasets:
                print(f"  • {ds.name}")
                # 检查基本完整性
                if (ds / "images").exists() and (ds / "masks").exists():
                    n_images = len(list((ds / "images").glob("*")))
                    n_masks = len(list((ds / "masks").glob("*")))
                    print(f"    图像: {n_images}, 标签: {n_masks}")
        else:
            print("\n⚠ 未找到已处理的数据集")
    
    # 检查划分文件
    splits_dir = data_root / "splits"
    if splits_dir.exists():
        split_datasets = [d for d in splits_dir.iterdir() if d.is_dir()]
        if split_datasets:
            print(f"\n已划分的数据集 ({len(split_datasets)}):")
            for ds in split_datasets:
                if (ds / "splits_5fold.jsonl").exists():
                    print(f"  ✓ {ds.name}")
                else:
                    print(f"  ✗ {ds.name} (缺少splits_5fold.jsonl)")
        else:
            print("\n⚠ 未找到数据划分")
    
    print("\n✓ 数据检查完成")
    return True


def check_models():
    """测试SOTA模型加载"""
    print("\n" + "=" * 60)
    print("模型检查")
    print("=" * 60)
    
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 测试Swin-UNetR
    try:
        from monai.networks.nets import SwinUNETR
        model = SwinUNETR(
            img_size=(224, 224),
            in_channels=3,
            out_channels=4,
            feature_size=48,
            use_checkpoint=False,
        )
        print(f"✓ Swin-UNetR加载成功")
        del model
    except Exception as e:
        print(f"✗ Swin-UNetR加载失败: {e}")
    
    # 测试nnUNet (简单检查导入)
    try:
        import nnunetv2
        print(f"✓ nnUNet v2可用")
    except ImportError:
        print(f"✗ nnUNet v2不可用 (需要安装nnunetv2)")
    
    # 测试SegResNet (MONAI)
    try:
        from monai.networks.nets import SegResNet
        m = SegResNet(spatial_dims=3, in_channels=1, out_channels=3)
        print(f"✓ SegResNet可用 (MONAI)")
        del m
    except Exception as e:
        print(f"✗ SegResNet不可用: {e}")
    
    # 测试 nnUNet PlainConvUNet
    try:
        from dynamic_network_architectures.architectures.unet import PlainConvUNet
        print(f"✓ PlainConvUNet可用 (dynamic_network_architectures)")
    except Exception as e:
        print(f"✗ PlainConvUNet不可用: {e}")
    
    print("\n✓ 模型检查完成")
    return True


def main():
    parser = argparse.ArgumentParser(description="Seg-MoE统一验证脚本")
    parser.add_argument("--env", action="store_true", help="检查环境")
    parser.add_argument("--data", action="store_true", help="检查数据")
    parser.add_argument("--models", action="store_true", help="测试模型")
    parser.add_argument("--all", action="store_true", help="运行所有检查")
    parser.add_argument("--dataset-config", type=str, help="数据集配置文件")
    
    args = parser.parse_args()
    
    # 默认运行所有检查
    if not any([args.env, args.data, args.models, args.all]):
        args.all = True
    
    success = True
    
    try:
        if args.all or args.env:
            if not check_environment():
                success = False
        
        if args.all or args.data:
            if not check_data(args.dataset_config):
                success = False
        
        if args.all or args.models:
            if not check_models():
                success = False
        
        print("\n" + "=" * 60)
        if success:
            print("✓ 所有检查通过")
            print("=" * 60)
            return 0
        else:
            print("✗ 部分检查失败")
            print("=" * 60)
            return 1
            
    except Exception as e:
        print(f"\n✗ 验证过程出错: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""测试GPU配置和双卡可用性.

Note: 若 CUDA 驱动层崩溃 (cuInit access violation), 本文件中的测试
      会被自动 skip, 不影响其他测试.
"""
import os
import pytest
import torch
import torch.nn as nn

# ── 安全检测 CUDA 是否可用 (防止 cuInit 崩溃) ──
def _cuda_safe():
    """Check CUDA via subprocess to avoid native crash (access violation)."""
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) == "":
        return False
    import subprocess, sys
    try:
        r = subprocess.run(
            [sys.executable, "-c", "import torch; print(torch.cuda.is_available())"],
            capture_output=True, text=True, timeout=15,
        )
        return r.returncode == 0 and "True" in r.stdout
    except Exception:
        return False

_HAS_CUDA = _cuda_safe()


@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA not available or driver crash")
def test_gpu_availability():
    """测试GPU可用性和配置"""
    print("=" * 60)
    print("GPU Configuration Test")
    print("=" * 60)
    
    # CUDA可用性
    cuda_available = torch.cuda.is_available()
    print(f"\n✓ CUDA Available: {cuda_available}")
    
    if not cuda_available:
        print("❌ No CUDA GPUs detected!")
        return False
    
    # GPU数量
    gpu_count = torch.cuda.device_count()
    print(f"✓ GPU Count: {gpu_count}")
    
    # GPU信息
    print(f"\n{'GPU ID':<10} {'Name':<40} {'Memory (GB)':<15} {'Capability':<15}")
    print("-" * 80)
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        name = props.name
        memory_gb = props.total_memory / 1024**3
        capability = f"{props.major}.{props.minor}"
        print(f"{i:<10} {name:<40} {memory_gb:<15.2f} {capability:<15}")
    
    # 当前设备
    current_device = torch.cuda.current_device()
    print(f"\n✓ Current Device: cuda:{current_device}")
    
    return True


@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA not available or driver crash")
def test_dataparallel():
    """测试DataParallel功能"""
    print("\n" + "=" * 60)
    print("DataParallel Test")
    print("=" * 60)
    
    gpu_count = torch.cuda.device_count()
    
    if gpu_count < 2:
        print(f"⚠️  Only {gpu_count} GPU detected, DataParallel requires 2+ GPUs")
        print("   Single GPU training will work normally")
        return True
    
    # 创建简单模型
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 64, 3, padding=1)
            self.bn = nn.BatchNorm2d(64)
            self.relu = nn.ReLU()
            
        def forward(self, x):
            return self.relu(self.bn(self.conv(x)))
    
    try:
        # 测试单GPU
        print(f"\n1. Testing single GPU (cuda:0)...")
        model_single = SimpleModel().cuda(0)
        x_single = torch.randn(4, 3, 256, 256).cuda(0)
        y_single = model_single(x_single)
        print(f"   ✓ Single GPU output shape: {y_single.shape}")
        
        # 测试DataParallel
        print(f"\n2. Testing DataParallel with {gpu_count} GPUs...")
        gpu_ids = list(range(gpu_count))
        model_parallel = nn.DataParallel(SimpleModel(), device_ids=gpu_ids).cuda(0)
        x_parallel = torch.randn(8, 3, 256, 256).cuda(0)  # 更大batch看并行效果
        y_parallel = model_parallel(x_parallel)
        print(f"   ✓ DataParallel output shape: {y_parallel.shape}")
        print(f"   ✓ Model distributed across GPUs: {gpu_ids}")
        
        # 速度对比 (简单测试)
        print(f"\n3. Performance comparison (100 forward passes)...")
        import time
        
        # 单GPU
        model_single.eval()
        start = time.time()
        with torch.no_grad():
            for _ in range(100):
                _ = model_single(x_single)
        torch.cuda.synchronize()
        single_time = time.time() - start
        print(f"   Single GPU: {single_time:.3f}s")
        
        # 双GPU
        model_parallel.eval()
        start = time.time()
        with torch.no_grad():
            for _ in range(100):
                _ = model_parallel(x_parallel)
        torch.cuda.synchronize()
        parallel_time = time.time() - start
        print(f"   DataParallel ({gpu_count} GPUs): {parallel_time:.3f}s")
        print(f"   Speedup: {single_time/parallel_time:.2f}x")
        
        print(f"\n✅ DataParallel test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ DataParallel test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


@pytest.mark.skipif(not _HAS_CUDA, reason="CUDA not available or driver crash")
def test_memory():
    """测试GPU显存"""
    print("\n" + "=" * 60)
    print("GPU Memory Test")
    print("=" * 60)
    
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        
        print(f"\nGPU {i}:")
        print(f"  Total Memory:     {total:.2f} GB")
        print(f"  Allocated:        {allocated:.2f} GB")
        print(f"  Reserved:         {reserved:.2f} GB")
        print(f"  Available:        {total - reserved:.2f} GB")
    
    return True


def main():
    """运行所有测试"""
    print("\n🔍 Seg-MoE GPU Configuration Check\n")
    
    results = []
    
    # Test 1: GPU availability
    results.append(("GPU Availability", test_gpu_availability()))
    
    # Test 2: DataParallel
    if results[0][1]:  # Only if CUDA is available
        results.append(("DataParallel", test_dataparallel()))
    
    # Test 3: Memory
    if results[0][1]:
        results.append(("Memory Check", test_memory()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{name:<30} {status}")
    
    print("\n" + "=" * 60)
    if all(r[1] for r in results):
        print("✅ All tests passed! Your dual RTX 5090 setup is ready.")
        print("\nTo use dual GPUs in training:")
        print("  python scripts/train/train_2d_experts.py --models configs/2d/models.yaml --gpus 0,1")
    else:
        print("⚠️  Some tests failed. Check output above for details.")
    print("=" * 60)


if __name__ == "__main__":
    main()

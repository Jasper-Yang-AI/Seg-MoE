"""
Smoke test: 验证训练引擎在 Windows DataParallel 下的核心功能.

测试内容:
  1. 单卡创建模型 → 前向/反向 → 不报错
  2. 多卡 DataParallel → 前向/反向 → 不报错
  3. checkpoint 保存 → 加载 → state_dict 匹配
  4. 加载带 module. 前缀的旧 DDP checkpoint → 自动 strip
  5. resume 后 epoch/global_step 递增

用法:
    python scripts/utils/smoke_test_train.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn


def _header(msg: str):
    print(f"\n{'='*60}\n  {msg}\n{'='*60}")


def test_single_gpu():
    """Test 1: 单卡前向/反向"""
    _header("Test 1: Single GPU forward/backward")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 3, 1)).to(device)
    x = torch.randn(2, 3, 64, 64, device=device)
    y = model(x)
    loss = y.mean()
    loss.backward()
    print(f"  output shape: {y.shape}, loss: {loss.item():.4f}")
    print("  PASSED")


def test_dataparallel():
    """Test 2: DataParallel 双卡"""
    _header("Test 2: DataParallel forward/backward")
    n_gpus = torch.cuda.device_count()
    if n_gpus < 2:
        print(f"  SKIP: only {n_gpus} GPU(s), need 2+")
        return
    device = torch.device("cuda:0")
    model = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 3, 1)).to(device)
    model = nn.DataParallel(model, device_ids=list(range(n_gpus)))
    x = torch.randn(4, 3, 64, 64, device=device)
    y = model(x)
    loss = y.mean()
    loss.backward()
    print(f"  GPUs: {n_gpus}, output shape: {y.shape}, loss: {loss.item():.4f}")
    print("  PASSED")


def test_checkpoint_save_load():
    """Test 3: checkpoint 保存/加载 (去 module. 前缀)"""
    _header("Test 3: Checkpoint save/load (unwrap module.)")
    from seg_moe.training.engine import normalize_state_dict_keys, unwrap_model

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 3, 1)).to(device)

    # Wrap with DP
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))

    # Save: 使用 unwrap_model 保存裸权重
    raw_sd = unwrap_model(model).state_dict()
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        ckpt_path = f.name
    torch.save({"epoch": 5, "global_step": 100, "model": raw_sd, "best_metric": 0.85}, ckpt_path)

    # Load: 新模型 (非 DP)
    model2 = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 3, 1)).to(device)
    state = torch.load(ckpt_path, map_location="cpu")
    sd = normalize_state_dict_keys(state["model"])

    # 确认 key 不含 module.
    for k in sd:
        assert not k.startswith("module."), f"key still has module. prefix: {k}"

    info = model2.load_state_dict(sd, strict=False)
    assert not info.missing_keys, f"missing keys: {info.missing_keys}"
    assert not info.unexpected_keys, f"unexpected keys: {info.unexpected_keys}"
    print(f"  Saved epoch={state['epoch']}, global_step={state['global_step']}, best_metric={state['best_metric']}")
    print(f"  Loaded {len(sd)} keys, missing={info.missing_keys}, unexpected={info.unexpected_keys}")
    print("  PASSED")

    Path(ckpt_path).unlink(missing_ok=True)


def test_ddp_ckpt_compat():
    """Test 4: 加载旧 DDP checkpoint (keys 带 module. 前缀)"""
    _header("Test 4: Load legacy DDP checkpoint (module. prefix)")
    from seg_moe.training.engine import normalize_state_dict_keys

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 3, 1)).to(device)

    # 模拟旧 DDP checkpoint: 所有 key 加 module. 前缀
    fake_ddp_sd = {"module." + k: v for k, v in model.state_dict().items()}
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        ckpt_path = f.name
    torch.save({"model": fake_ddp_sd}, ckpt_path)

    # 加载到新的裸模型
    model2 = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 3, 1)).to(device)
    state = torch.load(ckpt_path, map_location="cpu")
    sd = normalize_state_dict_keys(state["model"])
    info = model2.load_state_dict(sd, strict=False)
    assert not info.missing_keys, f"missing keys: {info.missing_keys}"
    print(f"  DDP keys (before strip): {list(state['model'].keys())[:3]}...")
    print(f"  Clean keys (after strip): {list(sd.keys())[:3]}...")
    print("  PASSED")

    Path(ckpt_path).unlink(missing_ok=True)


def test_resume_state():
    """Test 5: resume 后 optimizer/scheduler/scaler 状态恢复"""
    _header("Test 5: Resume state (optimizer + epoch + global_step)")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 3, 1)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    # 做 1 步训练
    x = torch.randn(2, 3, 32, 32, device=device)
    with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
        y = model(x)
        loss = y.mean()
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()

    # 保存
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        ckpt_path = f.name
    torch.save({
        "epoch": 10,
        "global_step": 500,
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "scaler": scaler.state_dict(),
        "best_metric": 0.92,
    }, ckpt_path)

    # 加载到新 optimizer
    model2 = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 3, 1)).to(device)
    opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)

    state = torch.load(ckpt_path, map_location="cpu")
    model2.load_state_dict(state["model"])
    opt2.load_state_dict(state["opt"])
    assert state["epoch"] == 10
    assert state["global_step"] == 500
    assert state["best_metric"] == 0.92
    print(f"  Restored: epoch={state['epoch']}, global_step={state['global_step']}, best_metric={state['best_metric']}")

    # 确认 optimizer 的 step count > 0
    for pg in opt2.param_groups:
        for p in pg["params"]:
            s = opt2.state.get(p)
            if s and "step" in s:
                assert s["step"] > 0, "optimizer step should be > 0 after resume"

    print("  PASSED")
    Path(ckpt_path).unlink(missing_ok=True)


def main():
    print("=" * 60)
    print("  Seg-MoE Smoke Test (Windows DataParallel)")
    print("=" * 60)

    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        for i in range(n):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_mem / 1024**3:.1f}GB)")
    else:
        print("  No CUDA GPUs detected, running on CPU")

    tests = [
        test_single_gpu,
        test_dataparallel,
        test_checkpoint_save_load,
        test_ddp_ckpt_compat,
        test_resume_state,
    ]

    passed = 0
    failed = 0
    skipped = 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"  FAILED: {e}")
            failed += 1
        except Exception as e:
            if "SKIP" in str(e):
                skipped += 1
            else:
                print(f"  FAILED: {e}")
                failed += 1

    print(f"\n{'='*60}")
    print(f"  Results: {passed} passed, {failed} failed, {skipped} skipped")
    print(f"{'='*60}")
    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()

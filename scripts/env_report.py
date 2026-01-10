from __future__ import annotations

import argparse
import os
import platform


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    print("=== Seg_MoE Environment Report ===")
    print(f"OS: {platform.platform()}")
    print(f"Python: {platform.python_version()}")

    try:
        import torch

        print(f"torch: {torch.__version__}")
        print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
        print(f"torch.version.cuda: {torch.version.cuda}")
        print(f"cudnn: {torch.backends.cudnn.version()}")

        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            print(f"cuda device_count: {n}")
            for i in range(n):
                p = torch.cuda.get_device_properties(i)
                total_gb = p.total_memory / (1024**3)
                print(f"GPU[{i}]: {p.name} | VRAM: {total_gb:.1f} GB | CC: {p.major}.{p.minor}")

        print(f"TF32 matmul allow: {torch.backends.cuda.matmul.allow_tf32}")
        print(f"TF32 cudnn allow: {torch.backends.cudnn.allow_tf32}")
        try:
            print(f"matmul_precision: {torch.get_float32_matmul_precision()}")
        except Exception:
            pass

        if args.verbose:
            print("--- Determinism flags ---")
            print(f"cudnn.deterministic: {torch.backends.cudnn.deterministic}")
            print(f"cudnn.benchmark: {torch.backends.cudnn.benchmark}")

    except Exception as e:
        print(f"torch import failed: {e}")

    # Optional: print key env vars that often affect determinism/perf
    keys = [
        "CUDA_VISIBLE_DEVICES",
        "CUBLAS_WORKSPACE_CONFIG",
        "PYTHONHASHSEED",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
    ]
    print("--- Env ---")
    for k in keys:
        if k in os.environ:
            print(f"{k}={os.environ[k]}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
from __future__ import annotations

import platform
import subprocess
import sys


def main() -> None:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"WSL detected: {_is_wsl()}")
    _print_nvidia_smi()

    try:
        import torch
    except ImportError:
        print("PyTorch: not installed")
        print("Install GPU training dependencies with: make install-finetune-wsl")
        raise SystemExit(1)

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA build: {torch.version.cuda}")

    if not torch.cuda.is_available():
        print("CUDA is not available to PyTorch. Check the Windows NVIDIA driver and WSL integration.")
        raise SystemExit(1)

    device_index = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(device_index)
    capability = torch.cuda.get_device_capability(device_index)
    total_gb = torch.cuda.get_device_properties(device_index).total_memory / 1024**3
    print(f"GPU: {device_name}")
    print(f"Compute capability: {capability[0]}.{capability[1]}")
    print(f"VRAM: {total_gb:.1f} GiB")
    print(f"BF16 supported: {torch.cuda.is_bf16_supported()}")


def _is_wsl() -> bool:
    release = platform.release().lower()
    if "microsoft" in release or "wsl" in release:
        return True
    try:
        return "microsoft" in open("/proc/version", encoding="utf-8").read().lower()
    except OSError:
        return False


def _print_nvidia_smi() -> None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        print("nvidia-smi: not found")
        return

    if result.returncode == 0:
        print(f"nvidia-smi: {result.stdout.strip()}")
    else:
        print(f"nvidia-smi error: {result.stderr.strip()}")


if __name__ == "__main__":
    main()

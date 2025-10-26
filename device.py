import logging
import psutil

try:
    import ctypes
    import subprocess
    _HAS_NVML = True
except Exception:
    _HAS_NVML = False


def get_ram_gb() -> float:
    return round(psutil.virtual_memory().total / (1024**3), 1)


def get_vram_info():
    """Return (vram_gb, cuda_version_str) if NVIDIA present; else (None, None)."""
    try:
        # Try nvidia-smi for VRAM and CUDA
        result = subprocess.run([
            'nvidia-smi', '--query-gpu=memory.total,driver_version', '--format=csv,noheader'
        ], capture_output=True, text=True, check=True)
        line = result.stdout.strip().splitlines()[0]
        mem_str, drv = [x.strip() for x in line.split(',')]
        if mem_str.lower().endswith(' mib'):
            mem_gb = round(float(mem_str[:-4]) / 1024.0, 1)
        else:
            mem_gb = None
        # CUDA version is harder; show driver version instead for clarity
        return mem_gb, f"NVIDIA Driver {drv}"
    except Exception:
        return None, None


def device_banner(is_gpu: bool) -> str:
    ram = get_ram_gb()
    vram, cuda_str = get_vram_info()
    if is_gpu:
        if vram is not None:
            extra = f"VRAM: {vram} GB"
        else:
            extra = "VRAM: n/a"
        tail = f"GPU • {extra}"
        if cuda_str:
            tail += f" • {cuda_str}"
    else:
        tail = "CPU"
    return f"Device: {tail} • RAM: {ram} GB"

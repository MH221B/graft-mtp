from __future__ import annotations

import torch


def resolve_device(preference: str = "auto") -> torch.device:
    """Pick the training device.

    An explicit preference wins. "auto" prefers the accelerator: torch exposes CUDA and HIP
    through the same `cuda` device type (HIP sets torch.version.hip, CUDA leaves it None), so
    one code path covers both. If nothing is visible it falls back to the CPU with a warning
    rather than failing - a CPU run is slow but valid, and a silent two-hour regression to the
    CPU is worse than a warning.
    """
    if preference and preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    import warnings

    warnings.warn(
        "no accelerator visible to torch; training on CPU (10-30x slower). "
        "Use .venv-rocm (ROCm wheel) to train on the iGPU.",
        RuntimeWarning,
        stacklevel=2,
    )
    return torch.device("cpu")


def describe_device() -> dict:
    """What `doctor` records and every run stamps into `state.json`."""
    info = {
        "torch": torch.__version__,
        "hip": getattr(torch.version, "hip", None),
        "cuda": getattr(torch.version, "cuda", None),
        "accelerator": None,
        "device_count": 0,
        "name": None,
        "total_memory_gib": None,
    }
    if torch.cuda.is_available():
        info["accelerator"] = "hip" if getattr(torch.version, "hip", None) else "cuda"
        info["device_count"] = torch.cuda.device_count()
        info["name"] = torch.cuda.get_device_name(0)
        info["total_memory_gib"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1024**3, 1
        )
    return info

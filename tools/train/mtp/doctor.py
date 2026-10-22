"""Preflight: everything a stage is about to assume, checked in ten seconds.

The failure this exists to prevent is not a missing file — that fails loudly on its own. It is
the quiet ones: a CPU-only torch that trains for an hour on the wrong device, a llama.cpp build
that is not the pinned commit, a harvest binary without the dump flags, a disk that fills up
mid-run. Each of those is cheap to detect here and expensive to discover later.

Run it under the interpreter that will do the work:

    ./.venv-rocm/Scripts/python.exe -m tools.train.mtp.doctor tools/train/mtp/configs/mtp-alignment.toml
"""

from __future__ import annotations

import ctypes
import json
import platform
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

from tools.train.mtp.device import describe_device
from tools.train.mtp.state import sha256_file

REQUIRED = ["torch", "transformers", "safetensors", "datasets", "gguf", "numpy"]
NEEDED_FILES = ("verifier_gguf", "mmproj", "llama_server", "llama_imatrix")
STAGE_FILES = ("prompts",)
MIN_DISK_GB = 60.0
RUNTIME_BUILD = "11228"
HARVEST_FLAGS = ("--tokens", "--dump-nextn-dir", "--dump-nextn-seqs", "--chunk-index")


def total_ram_gb() -> float:
    if sys.platform != "win32":
        return 0.0

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
    return stat.ullTotalPhys / (1024**3)


def _run(path: Path, *args: str) -> str:
    try:
        out = subprocess.run(
            [str(path), *args], capture_output=True, text=True, check=False, timeout=120
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"error: {exc}"
    return (out.stdout or "") + (out.stderr or "")


def _first_line(text: str) -> str:
    lines = [line for line in text.splitlines() if line.strip()]
    return lines[0] if lines else "no output"


def check(config_path: Path) -> dict:
    cfg = tomllib.loads(config_path.read_text(encoding="utf-8"))
    paths = cfg["paths"]
    report: dict = {
        "platform": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
        "ram_gb": round(total_ram_gb(), 1),
        "deps": {},
        "files": {},
        "llama": {},
        "problems": [],
    }
    for mod in REQUIRED:
        try:
            __import__(mod)
            report["deps"][mod] = "ok"
        except Exception as exc:  # noqa: BLE001
            report["deps"][mod] = f"missing: {exc}"
            report["problems"].append(f"python module missing: {mod}")

    for name in NEEDED_FILES:
        path = Path(paths[name])
        report["files"][name] = str(path)
        if not path.exists():
            report["problems"].append(f"missing file: {path}")
    for name in STAGE_FILES:
        path = Path(paths[name])
        report["files"][name] = str(path)
        if not path.exists():
            report["problems"].append(f"missing file: {path} (a later stage needs it)")
    # The acceptance harness measures on a prompt set whose generations are NOT in the corpus.
    # `paths.prompts` is the selfgen set, so the two must differ.
    eval_prompts = Path(cfg.get("eval", {}).get("prompts") or paths["prompts"])
    report["files"]["eval_prompts"] = str(eval_prompts)
    if not eval_prompts.exists():
        report["problems"].append(f"missing file: {eval_prompts} (the acceptance harness needs it)")
    elif eval_prompts.resolve() == Path(paths["prompts"]).resolve():
        report["problems"].append(
            "eval prompts are the selfgen prompts: acceptance would be measured on the corpus"
        )

    verifier = Path(paths["verifier_gguf"])
    if verifier.exists():
        report["verifier_sha256"] = sha256_file(verifier)

    server = Path(paths["llama_server"])
    if server.exists():
        text = _run(server, "--version")
        report["llama"]["runtime_version"] = _first_line(text)
        # The trained head is only valid against the runtime whose forward it was aligned to.
        # A different build is a different function, so this is a hard stop, not a note.
        if RUNTIME_BUILD not in text:
            report["problems"].append(
                f"llama.cpp runtime is not build {RUNTIME_BUILD}: {_first_line(text)}"
            )

    imatrix = Path(paths["llama_imatrix"])
    if imatrix.exists():
        # The stock imatrix tool is fine for imatrix work; the harvest uses the patched
        # build in `dump_tool`. Recorded so the report shows both.
        report["llama"]["stock_imatrix"] = _first_line(_run(imatrix, "--version"))

    dump = Path(paths["dump_tool"])
    if dump.exists():
        text = _run(dump, "--help")
        report["llama"]["harvest_build"] = _first_line(_run(dump, "--version"))
        missing = [flag for flag in HARVEST_FLAGS if flag not in text]
        if missing:
            report["problems"].append(
                f"harvest build is missing {missing}; rebuild it from Task 6"
            )
    else:
        report["llama"]["harvest_build"] = "missing (Task 6 builds it)"

    free = shutil.disk_usage(Path.cwd()).free / (1024**3)
    report["disk_free_gb"] = round(free, 1)
    if free < MIN_DISK_GB:
        report["problems"].append(f"only {free:.1f} GB free; need ~{MIN_DISK_GB:.0f} GB")

    # The trainer runs on the iGPU. A CPU-only torch build is the failure mode that costs
    # the most time and is the easiest to miss, so it is a hard problem, not a warning.
    device = describe_device()
    report["device"] = device
    if device["accelerator"] is None:
        report["problems"].append(
            "torch sees no accelerator; training would fall back to the CPU. "
            f"Use the ROCm interpreter (.venv-rocm\\Scripts\\python.exe), not {sys.executable}"
        )
    elif device["total_memory_gib"] is not None and device["total_memory_gib"] < 6.0:
        report["problems"].append(
            f"accelerator reports only {device['total_memory_gib']:.1f} GiB; "
            "the training window needs ~6 GiB (head + frozen LM head + one logits block)"
        )

    return report


def main() -> int:
    cfg = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tools/train/mtp/configs/mtp-alignment.toml")
    report = check(cfg)
    print(json.dumps(report, indent=2))
    return 1 if report["problems"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

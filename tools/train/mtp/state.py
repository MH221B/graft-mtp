"""Run state, content hashes, and the run lock.

This module is deliberately dependency-free: it is imported by every stage, including the ones
that run before torch is anywhere near, and it must keep working when a run is resumed by a
different process on a different day.

Three rules it exists to enforce:

1. Every write is atomic. A power cut leaves either the old file or the new one, never a
   half-written `state.json` that makes a finished stage look pending.
2. A stage is done only if its inputs still hash to what they hashed when it ran. Editing the
   config or regenerating an artifact invalidates everything downstream, automatically.
3. Two processes cannot drive one run. A lock left by a dead PID is reclaimed; a lock held by a
   live one is an error, with a manual override for the case where the PID check is wrong.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

SCHEMA = 1


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            data = f.read(chunk)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: Path, obj: object) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True) + "\n")


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        ERROR_ACCESS_DENIED = 5
        STILL_ACTIVE = 259
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            # A live process we are not allowed to open still owns the lock. Only a genuine
            # "no such process" (ERROR_INVALID_PARAMETER) means the PID is free, so treating
            # any failure as dead would let a second run steal a lock that is held.
            return kernel32.GetLastError() == ERROR_ACCESS_DENIED
        exit_code = ctypes.c_ulong(STILL_ACTIVE)
        ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
        kernel32.CloseHandle(handle)
        if not ok:
            return True  # cannot tell; assume held
        # A terminated process whose handle is still open (a zombie) still opens by PID,
        # and reporting it as alive would strand the lock until the parent exits.
        return exit_code.value == STILL_ACTIVE
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def pid_from_lock(path: Path) -> int:
    try:
        return int(path.read_text(encoding="utf-8").split()[0])
    except (ValueError, IndexError, OSError):
        return -1


class RunLock:
    """An exclusive lock file holding `<pid> <unix time>`.

    Creation is atomic (`O_EXCL`), so two processes starting at the same instant cannot both
    believe they took it. A lock whose PID is dead is stale and is replaced; a live one is an
    error unless `force` is set.
    """

    def __init__(self, path: Path):
        self.path = path
        self._held = False

    def acquire(self, force: bool = False) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                with self.path.open("x", encoding="utf-8") as f:
                    f.write(f"{os.getpid()} {time.time()}\n")
                self._held = True
                return
            except FileExistsError:
                pid = pid_from_lock(self.path)
                if not force and pid_alive(pid):
                    raise RuntimeError(
                        f"run is locked by pid {pid}; use --force-unlock only if it is dead"
                    )
                self.path.unlink(missing_ok=True)
        raise RuntimeError(f"could not acquire lock {self.path} after clearing a stale one")

    def release(self) -> None:
        if self.path.exists() and pid_from_lock(self.path) == os.getpid():
            self.path.unlink(missing_ok=True)
        self._held = False

    def __enter__(self) -> RunLock:
        self.acquire()
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


class RunState:
    """`state.json`: stage status, input and output hashes, timestamps, PID, last error."""

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.path = run_dir / "state.json"
        run_dir.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
            if self.data.get("schema") != SCHEMA:
                raise RuntimeError(f"state schema {self.data.get('schema')} != {SCHEMA}")
        else:
            self.data = {"schema": SCHEMA, "stages": {}}
            self.save()

    def save(self) -> None:
        atomic_write_json(self.path, self.data)

    def status(self, stage: str) -> str:
        return self.data["stages"].get(stage, {}).get("status", "pending")

    def entry(self, stage: str) -> dict:
        return self.data["stages"].get(stage, {})

    def mark(self, stage: str, status: str, **fields: object) -> None:
        entry = self.data["stages"].setdefault(stage, {})
        entry.update(fields)
        entry["status"] = status
        entry["updated"] = time.time()
        if status == "running":
            entry["pid"] = os.getpid()
        self.save()

    def is_done(self, stage: str, inputs: dict[str, str]) -> bool:
        entry = self.data["stages"].get(stage, {})
        return entry.get("status") == "done" and entry.get("inputs") == inputs

    def artifact(self, stage: str, name: str) -> Path:
        path = self.run_dir / "artifacts" / stage / name
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

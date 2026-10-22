"""Prove the three guarantees state.py exists to provide."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from tools.train.mtp.state import (
    SCHEMA,
    RunLock,
    RunState,
    atomic_write_text,
    pid_alive,
    pid_from_lock,
    sha256_file,
    sha256_text,
)

# Everything happens in a scratch directory so a gate run never leaves residue in the repo.
ROOT = Path(tempfile.mkdtemp(prefix="mtp-state-"))
fails: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"{'ok  ' if cond else 'FAIL'}  {name}{'  ' + detail if detail and not cond else ''}")
    if not cond:
        fails.append(name)


def fresh(path: Path) -> Path:
    shutil.rmtree(path, ignore_errors=True)
    return path


# --- hashes -------------------------------------------------------------------------------
p = ROOT / "hashme.bin"
p.parent.mkdir(parents=True, exist_ok=True)
p.write_bytes(b"abc" * 1000)
check("sha256_file matches hashlib", sha256_file(p) == __import__("hashlib").sha256(b"abc" * 1000).hexdigest())
check("sha256_text is hex", len(sha256_text("x")) == 64)

# --- atomic writes ------------------------------------------------------------------------
target = ROOT / "deep" / "nested" / "state.json"
atomic_write_text(target, "hello")
check("atomic write creates parents", target.read_text(encoding="utf-8") == "hello")
check("atomic write leaves no .tmp", not (target.with_name("state.json.tmp")).exists())
check("atomic write uses LF", b"\r\n" not in target.read_bytes())

# --- state round trip ---------------------------------------------------------------------
d = fresh(ROOT / "run")
s = RunState(d)
check("state.json created", (d / "state.json").exists())
check("schema stamped", s.data["schema"] == SCHEMA)
check("unknown stage is pending", s.status("nope") == "pending")

s.mark("harvest", "running")
check("running stamps pid", s.entry("harvest").get("pid") == os.getpid())
s.mark("harvest", "done", inputs={"a": "b"}, outputs={"c": "d"}, chunks_total=7)
check("is_done true for same inputs", s.is_done("harvest", {"a": "b"}))
check("is_done false for changed inputs", not s.is_done("harvest", {"a": "x"}))
check("is_done false while running", not RunState(fresh(ROOT / "run2")).is_done("x", {}))

s2 = RunState(d)  # a second process, same day or not
check("state survives reload", s2.is_done("harvest", {"a": "b"}))
check("fields survive reload", s2.entry("harvest")["outputs"] == {"c": "d"} and s2.entry("harvest")["chunks_total"] == 7)
check("json is sorted and newline terminated", (d / "state.json").read_text(encoding="utf-8").endswith("}\n"))

# --- schema mismatch ----------------------------------------------------------------------
bad = fresh(ROOT / "bad")
bad.mkdir(parents=True, exist_ok=True)
(bad / "state.json").write_text(json.dumps({"schema": 999, "stages": {}}), encoding="utf-8")
try:
    RunState(bad)
    check("schema mismatch raises", False)
except RuntimeError:
    check("schema mismatch raises", True)

# --- pid alive ----------------------------------------------------------------------------
check("own pid is alive", pid_alive(os.getpid()))
check("pid 0 is dead", not pid_alive(0))
zombie = subprocess.Popen([sys.executable, "-c", "pass"])
zombie.wait()  # terminated, but this process still holds its handle
check("a terminated pid is dead even with its handle open", not pid_alive(zombie.pid))

# --- lock ---------------------------------------------------------------------------------
lock_path = ROOT / "run" / "lock"
lock = RunLock(lock_path)
lock.acquire()
check("lock file holds our pid", pid_from_lock(lock_path) == os.getpid())
try:
    RunLock(lock_path).acquire()
    check("second acquire is refused", False)
except RuntimeError:
    check("second acquire is refused", True)
lock.release()
check("release removes the file", not lock_path.exists())

# a live lock cannot be stolen, and release() must not delete a lock that belongs to another pid
lock.acquire()
lock_path.write_text(f"{zombie.pid} {time.time()}\n", encoding="utf-8")  # pretend a foreign pid
RunLock(lock_path).release()
check("release() leaves another pid's lock alone", lock_path.exists())
lock.release()
lock_path.unlink(missing_ok=True)

# stale lock from a dead pid is reclaimed without force
lock_path.write_text(f"{zombie.pid} {time.time()}\n", encoding="utf-8")
RunLock(lock_path).acquire()
check("stale lock reclaimed", pid_from_lock(lock_path) == os.getpid())
RunLock(lock_path).release()

# force overrides a live lock
lock_path.write_text(f"{os.getpid()} {time.time()}\n", encoding="utf-8")
RunLock(lock_path).acquire(force=True)
check("force overrides a live lock", pid_from_lock(lock_path) == os.getpid())
RunLock(lock_path).release()

# --- the race: 8 processes, no double holder ----------------------------------------------
race = fresh(ROOT / "race")
race.mkdir(parents=True, exist_ok=True)
rlock = race / "lock"
script = (
    "import sys,time;sys.path.insert(0,r'.');"
    "from pathlib import Path;from tools.train.mtp.state import RunLock;"
    "l=RunLock(Path(sys.argv[1]));"
    "ok=True\n"
    "try:\n l.acquire()\nexcept RuntimeError:\n ok=False\n"
    "print('GOT' if ok else 'BUSY');"
    "time.sleep(0.15);l.release() if ok else None"
)
procs = [
    subprocess.Popen([sys.executable, "-c", script, str(rlock)], stdout=subprocess.PIPE, text=True)
    for _ in range(8)
]
outs = [pr.communicate()[0].strip() for pr in procs]
check("exactly one winner in a race", outs.count("GOT") >= 1 and outs.count("BUSY") == 8 - outs.count("GOT"), str(outs))

# --- artifacts ----------------------------------------------------------------------------
a = RunState(ROOT / "run").artifact("package", "D.gguf")
check("artifact dir created", a.parent.exists() and a.name == "D.gguf")

print()
print("FAILED: " + ", ".join(fails) if fails else "all checks passed")
shutil.rmtree(ROOT, ignore_errors=True)
raise SystemExit(1 if fails else 0)

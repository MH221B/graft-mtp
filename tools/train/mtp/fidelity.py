"""G1: is our PyTorch head the same function as the runtime's MTP block?

The check is not "does the head predict well". It is: run llama.cpp's own MTP block and our
reimplementation on the same weights and the same post-output-norm hidden state, and compare
the outputs. If they differ, one of the two is wrong, and a head trained through the wrong one
is trained against a function the runtime does not run.

Three numbers:

- G1a: max |our h_out - llama.cpp's h_out|. The block's output, elementwise.
- G1b: the fraction of positions where our argmax token equals the runtime's. Discrete, and the
  one that acceptance actually depends on.
- G1c: max |our logits - the runtime's logits|. The tied head's output, elementwise.

The reference comes from `llama-imatrix --dump-mtp-out`, which creates a real MTP context
(ctx_type = MTP) and decodes a batch of (token, hidden) pairs, exactly like the speculative loop.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tomllib
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer

from tools.train.mtp.mtp_model import (
    MTPHead,
    load_head_from_gguf,
    load_lm_head,
    resolve_device,
)

EMBD = 2560
PROBE_TOKENS = 256
# Tolerances are set by the artifact's dtype, not by what looks strict. The head in B is f16,
# and ggml rounds f32 activations to f16 for an f16-weight matmul, so about 1% per matmul is
# the floor. Measured after the norm-convention fix: h_out max 1.89, logits max 0.57, token
# match 0.996. A structural error (wrong concat order, wrong offset, the missing +1 on the
# norms) produced 60 / 30 / 0.0, so these bounds keep a 20-50x margin against the failure the
# gate exists to catch while allowing the dtype's own noise.
TOL_H_OUT = 3.0
TOL_LOGITS = 1.0
G1B_MIN = 0.99
CACHE_TOL = 1e-3
CACHE_ROW_TOKENS = 512


def causal_mask(
    n: int,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    mask = torch.zeros(1, 1, n, n, dtype=dtype, device=device)
    mask.masked_fill_(
        torch.triu(torch.ones(n, n, dtype=torch.bool, device=device), diagonal=1),
        torch.finfo(dtype).min,
    )
    return mask


def _probe_ids(tokenizer, n: int) -> np.ndarray:
    text = "The quick brown fox jumps over the lazy dog. " * 500
    ids = tokenizer(text, add_special_tokens=False)["input_ids"][:n]
    if len(ids) < n:
        raise RuntimeError(f"probe text is only {len(ids)} tokens")
    return np.asarray(ids, dtype=np.int32)


def _dump(
    tool: Path,
    model: Path,
    out_dir: Path,
    ids: np.ndarray,
    chunk: int,
    mtp: bool,
    max_seq: int = 4096,
    ngl: int = 99,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ids_path = out_dir / "probe.ids.bin"
    ids.tofile(ids_path)
    cmd = [
        str(tool),
        "-m", str(model),
        "--tokens", str(ids_path),
        "--dump-nextn-dir", str(out_dir),
        "--chunk-index", str(chunk),
        "-c", str(max_seq),
        "-ngl", str(ngl),
        "--no-ppl",
    ]
    if mtp:
        cmd.append("--dump-mtp-out")
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _read(out_dir: Path, chunk: int, suffix: str) -> np.ndarray:
    path = out_dir / f"chunk-{chunk:05d}.{suffix}.bin"
    if not path.exists():
        raise RuntimeError(f"missing dump: {path}")
    return np.fromfile(path, dtype=np.float32)


def probe(
    config_path: Path,
    artifact: Path | None = None,
    ngl: int = 99,
    dtype: torch.dtype = torch.float32,
) -> dict:
    """G1a-G1c on a 256-token probe.

    Runs in f32 by default. The gate asks whether the two implementations compute the same
    function; a bf16 comparison answers a different question, because bf16's rounding moves a
    saturated softmax by whole tokens. The bf16 delta is reported separately.
    """
    cfg = tomllib.loads(config_path.read_text(encoding="utf-8"))
    run_dir = Path("runs/mtp-alignment") / cfg["run"]["run_id"]
    probe_dir = run_dir / "probe"
    model = artifact or (run_dir / "artifacts" / "B-grafted-f16.gguf")
    tool = Path(cfg["paths"]["dump_tool"])
    if not model.exists():
        raise SystemExit(f"missing artifact: {model} (build it first, or pass --artifact)")

    tokenizer = AutoTokenizer.from_pretrained(cfg["paths"]["model_id"], trust_remote_code=True)
    ids = _probe_ids(tokenizer, PROBE_TOKENS)
    _dump(tool, model, probe_dir, ids, 0, mtp=True, max_seq=512, ngl=ngl)

    n = int(ids.size)
    device = resolve_device(str(cfg.get("train", {}).get("device", "auto")))

    hidden = torch.from_numpy(_read(probe_dir, 0, "hidden")[: n * EMBD].reshape(n, EMBD))
    ref_out = torch.from_numpy(_read(probe_dir, 0, "out")[: n * EMBD].reshape(n, EMBD))
    ref_logits = torch.from_numpy(_read(probe_dir, 0, "logits").reshape(n, -1))

    config = AutoConfig.from_pretrained(cfg["paths"]["model_id"], trust_remote_code=True)
    text = getattr(config, "text_config", config)
    head = MTPHead(text).to(dtype).eval().to(device)
    load_head_from_gguf(model, head, dtype=dtype)
    lm = load_lm_head(model, dtype=dtype).to(device)

    hidden = hidden.to(dtype)[None].to(device)
    # The input embedding is a row lookup in the tied head: the same matrix, so there is no
    # second copy that could drift from it.
    embed = lm[torch.from_numpy(ids.astype(np.int64)).to(device)][None].to(dtype)

    with torch.no_grad():
        h_out = head(
            hidden,
            embed,
            causal_mask(n, device, dtype),
            torch.arange(n, device=device)[None],
        )
        # f32, not bf16: the reference is f16 weights with f32 accumulation, and a bf16
        # matmul here would add ~1e-2 of its own error to the number G1c measures.
        logits = h_out[0].float() @ lm.float().T

    ref_out = ref_out.to(device)
    ref_logits = ref_logits.to(device)

    dh = (h_out[0].float() - ref_out).abs()
    dl = (logits - ref_logits).abs()
    g1a = dh.max().item()
    g1c = dl.max().item()
    g1b = (logits.argmax(-1) == ref_logits.argmax(-1)).float().mean().item()

    return {
        "artifact": model.name,
        "probe_tokens": n,
        "probe_dtype": str(dtype).replace("torch.", ""),
        "g1a_tolerance": TOL_H_OUT,
        "g1a_max_abs_delta_h_out": g1a,
        "g1a_mean_abs_delta_h_out": dh.mean().item(),
        "g1a_p999_abs_delta_h_out": torch.quantile(dh.flatten(), 0.999).item(),
        "g1a_pass": g1a <= TOL_H_OUT,
        "g1b_draft_token_match": g1b,
        "g1b_min": G1B_MIN,
        "g1b_pass": g1b >= G1B_MIN,
        "g1c_max_abs_delta_logits": g1c,
        "g1c_mean_abs_delta_logits": dl.mean().item(),
        "g1c_p999_abs_delta_logits": torch.quantile(dl.flatten(), 0.999).item(),
        "g1c_pass": g1c <= TOL_LOGITS,
        "g1c_source": "llama.cpp MTP context (--dump-mtp-out)",
    }


def verify_cache(config_path: Path, sequences: int = 8) -> dict:
    """Re-dump the first row of a few chunks and compare with the cached hidden states.

    A whole chunk cannot be re-dumped and compared as one sequence: the cache holds one row's
    states per row, each decoded on its own. So the check re-dumps exactly one row (the first,
    capped at CACHE_ROW_TOKENS) and compares that span, which is an exact comparison.
    """
    cfg = tomllib.loads(config_path.read_text(encoding="utf-8"))
    run_dir = Path("runs/mtp-alignment") / cfg["run"]["run_id"]
    cache = run_dir / "cache"
    tool = Path(cfg["paths"]["dump_tool"])
    model = run_dir / "artifacts" / "B-grafted-f16.gguf"
    shards = sorted(cache.glob("chunk-*.done.json"))
    if not shards:
        return {"cache_check": "skipped, no cache yet"}
    worst = 0.0
    checked = 0
    for shard in shards[:sequences]:
        index = json.loads(shard.read_text(encoding="utf-8"))["index"]
        ids = np.fromfile(cache / f"chunk-{index:05d}.ids.bin", dtype=np.int32)
        seqs = np.fromfile(cache / f"chunk-{index:05d}.seqs.bin", dtype=np.int32)
        if seqs.size < 2:
            continue
        offset, length = int(seqs[0]), int(seqs[1])
        window = ids[offset : offset + min(length, CACHE_ROW_TOKENS)]
        out_dir = run_dir / "probe" / f"check-{index:05d}"
        _dump(tool, model, out_dir, window, 0, mtp=False)
        cached = np.fromfile(cache / f"chunk-{index:05d}.hidden.bin", dtype=np.float32)[
            offset * EMBD : (offset + window.size) * EMBD
        ]
        fresh = _read(out_dir, 0, "hidden")[: window.size * EMBD]
        delta = float(np.abs(cached - fresh).max())
        print(f"chunk {index} row 0 ({window.size} tokens): max abs delta {delta:.3e}")
        worst = max(worst, delta)
        checked += 1
    return {
        "cache_check": f"run on {checked} chunks",
        "cache_max_abs_delta": worst,
        "cache_pass": worst <= CACHE_TOL,
    }


def run(
    config_path: Path,
    phase: str = "all",
    artifact: Path | None = None,
    ngl: int = 99,
    dtype: torch.dtype = torch.float32,
) -> int:
    run_dir = Path("runs/mtp-alignment") / tomllib.loads(
        config_path.read_text(encoding="utf-8")
    )["run"]["run_id"]
    report: dict = {}
    failed = False
    if phase in ("all", "probe"):
        report.update(probe(config_path, artifact, ngl, dtype))
        failed = failed or not (report["g1a_pass"] and report["g1b_pass"] and report["g1c_pass"])
    if phase in ("all", "cache"):
        report.update(verify_cache(config_path))
        failed = failed or not report.get("cache_pass", True)
    print(json.dumps(report, indent=2))
    (run_dir / "eval").mkdir(parents=True, exist_ok=True)
    (run_dir / "eval" / "fidelity.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("phase", nargs="?", default="all", choices=["all", "probe", "cache"])
    parser.add_argument("--artifact", type=Path, default=None)
    parser.add_argument("--ngl", type=int, default=99, help="GPU layers for the reference dump (0 = CPU)")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "bfloat16"],
        help="dtype for the PyTorch side of the probe (default: float32, the implementation check)",
    )
    args = parser.parse_args()
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    return run(args.config, args.phase, args.artifact, args.ngl, dtype)


if __name__ == "__main__":
    sys.exit(main())

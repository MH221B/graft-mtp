"""Run the harvest: trunk hidden states for the whole corpus, one row per context.

The dump tool writes `chunk-NNNNN.hidden.bin` (f32, `tokens x n_embd`) next to the
`chunk-NNNNN.ids.bin` this driver writes. There is no `embed.bin`: the input embedding is a row
lookup in `token_embd.weight`, the same matrix the output head uses, so the trainer gathers it
from the tensor it already holds. That keeps the tied weight literally tied and halves the dump.

Each chunk is a transport unit, not a context. The rows inside it are decoded separately by the
tool (`--dump-nextn-seqs`), because a row decoded after another row would attend to it and its
hidden states would not be the ones the runtime produces.
"""

from __future__ import annotations

import json
import subprocess
import tomllib
from pathlib import Path

import numpy as np

from tools.train.mtp.state import RunState, sha256_file, sha256_text

FALLBACK_EMBD = 2560


def _chunks(rows: list[dict], chunk_tokens: int) -> list[tuple[int, int, list[tuple[int, int]]]]:
    """Split the corpus into chunks at row boundaries.

    A row is never split across two chunks, and every row keeps its own span inside
    the chunk. The harvest runs each row as its own context, so these boundaries are
    the contract with the dump tool.
    """
    out: list[tuple[int, int, list[tuple[int, int]]]] = []
    start = 0
    end = 0
    seqs: list[tuple[int, int]] = []
    for row in rows:
        offset, length = int(row["offset"]), int(row["length"])
        if end > start and end - start + length > chunk_tokens:
            out.append((start, end, seqs))
            start, end, seqs = offset, offset + length, []
        else:
            end = offset + length
        seqs.append((offset - start, length))
    if end > start:
        out.append((start, end, seqs))
    return out


def n_embd(cfg: dict) -> int:
    """The hidden width, from the model config. The dump size check depends on it."""
    path = Path(cfg["paths"]["hf_dir"]) / "config.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        text = data.get("text_config", data)
        value = text.get("hidden_size") or text.get("n_embd")
        if value:
            return int(value)
    return FALLBACK_EMBD


def _fingerprint(cfg: dict, model: Path, tool: Path, tokens: np.ndarray, width: int) -> str:
    """What a chunk's numbers depend on. A chunk is reusable only under the same value.

    `chunk_tokens` changes where the boundaries fall, so every chunk index then means a
    different token range. Skipping `done.json` files from a different split would mix two
    chunkings into one cache: each file is individually valid and the dataset is not. The
    fingerprint is stored in every `done.json` and checked on resume.
    """
    parts = [
        f"chunk_tokens={int(cfg['corpus']['chunk_tokens'])}",
        f"max_seq={int(cfg['harvest']['max_seq'])}",
        f"n_embd={width}",
        f"tokens={tokens.size}",
        f"tail={int(tokens[-1]) if tokens.size else 0}",
        f"model={sha256_file(model)}",
        f"tool={sha256_file(tool)}",
    ]
    return sha256_text("\n".join(parts))


def run(config_path: Path, fail_after: int = 0) -> None:
    cfg = tomllib.loads(config_path.read_text(encoding="utf-8"))
    run_dir = Path("runs/mtp-alignment") / cfg["run"]["run_id"]
    state = RunState(run_dir)
    tokens = np.load(run_dir / "corpus" / "tokens.npy")
    rows = [
        json.loads(line)
        for line in (run_dir / "corpus" / "rows.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    chunks = _chunks(rows, int(cfg["corpus"]["chunk_tokens"]))
    out = run_dir / "cache"
    out.mkdir(parents=True, exist_ok=True)
    tool = Path(cfg["paths"]["dump_tool"])
    # The dump tool needs block 32 to create the MTP context, so it runs against the
    # grafted file, not the bare target.
    model = run_dir / "artifacts" / "B-grafted-f16.gguf"
    if not model.exists():
        raise SystemExit(f"missing {model}; run the graft stage first")
    if not tool.exists():
        raise SystemExit(f"missing {tool}; build it in Task 6")

    width = n_embd(cfg)
    max_seq = int(cfg["harvest"]["max_seq"])
    fingerprint = _fingerprint(cfg, model, tool, tokens, width)

    for index, (start, end, seqs) in enumerate(chunks):
        done = out / f"chunk-{index:05d}.done.json"
        if done.exists():
            previous = json.loads(done.read_text(encoding="utf-8"))
            if previous.get("fingerprint") != fingerprint:
                raise SystemExit(
                    f"chunk {index} was harvested under different inputs: chunk_tokens, max_seq, "
                    f"the model or the tool changed. The chunk boundaries no longer line up with "
                    f"the corpus. Delete {out} and run the harvest again."
                )
            print(f"chunk {index}: already done")
            continue
        ids = tokens[start:end]
        ids_path = out / f"chunk-{index:05d}.ids.bin"
        np.asarray(ids, dtype=np.int32).tofile(ids_path)
        # The row boundaries: int32 (relative offset, length) pairs.
        seqs_path = out / f"chunk-{index:05d}.seqs.bin"
        np.asarray([v for pair in seqs for v in pair], dtype=np.int32).tofile(seqs_path)
        cmd = [
            str(tool),
            "-m", str(model),
            "--tokens", str(ids_path),
            "--dump-nextn-seqs", str(seqs_path),
            "--dump-nextn-dir", str(out),
            "--chunk-index", str(index),
            "-c", str(max_seq),
            "-ngl", "99",
            "--no-ppl",
        ]
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        hidden = out / f"chunk-{index:05d}.hidden.bin"
        if not hidden.exists():
            raise RuntimeError(f"missing dump: {hidden}")
        expected = int(ids.size) * width * 4
        actual = hidden.stat().st_size
        if actual != expected:
            raise RuntimeError(
                f"{hidden} is {actual} bytes, expected {expected} "
                f"({ids.size} tokens x {width} x 4). Is n_embd right?"
            )
        done.write_text(
            json.dumps(
                {
                    "index": index,
                    "tokens": int(ids.size),
                    "rows": len(seqs),
                    "n_embd": width,
                    "fingerprint": fingerprint,
                    "hidden": sha256_file(hidden),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        state.mark(
            "harvest",
            "running",
            chunks_done=index + 1,
            chunks_total=len(chunks),
            fingerprint=fingerprint,
        )
        if fail_after and index + 1 >= fail_after:
            raise RuntimeError("injected failure: harvest")
    state.mark("harvest", "done", chunks_total=len(chunks))
    print(f"harvest complete: {len(chunks)} chunks, {int(tokens.size)} tokens")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--fail-after-chunks", type=int, default=0)
    args = parser.parse_args()
    run(args.config, fail_after=args.fail_after_chunks)

"""Train the MTP head on the harvested cache.

The loss is weighted by the per-token weights the corpus builder wrote (assistant spans count
2x), and `best.pt` is selected by the held-out draft-match proxy, because acceptance - not the
loss - is the deployment metric.

Two things about this trainer are deliberate:

- The input embedding is gathered from the frozen LM head. `token_embd.weight` is the tied output
  head, so there is one matrix, not two that could drift apart, and the cache only stores the
  hidden states.
- The logits are computed in blocks of positions. A full 2048-token logits block is 2 GB in fp32
  and the iGPU has 16.9 GiB shared with everything else; 512-position blocks are both faster and
  4 GiB lighter, with identical arithmetic.
"""

from __future__ import annotations

import argparse
import json
import random
import time
import tomllib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoConfig

from tools.train.mtp.device import describe_device, resolve_device
from tools.train.mtp.mtp_model import MTPHead, hf_shards, load_lm_head, load_mtp_weights
from tools.train.mtp.state import RunLock, RunState

EMBD = 2560
LM_CHUNK = 512

# torch.load defaults to weights_only=True. The optimiser state dict carries this marker, and
# these checkpoints are written by this module into this run directory.
torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])


def _corpus(run_dir: Path) -> tuple[list[dict], np.ndarray]:
    rows = [
        json.loads(line)
        for line in (run_dir / "corpus" / "rows.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    weights = np.load(run_dir / "corpus" / "weights.npy")
    return rows, weights


def _shards(cache: Path) -> list[dict]:
    return [
        json.loads(shard.read_text(encoding="utf-8"))
        for shard in sorted(cache.glob("chunk-*.done.json"))
    ]


def _sources_in(rows: list[dict], start: int, end: int) -> set[str]:
    found = set()
    for row in rows:
        if row["offset"] < end and start < row["offset"] + row["length"]:
            found.add(row["source"])
    return found


class _Windows:
    """Window specs, plus the one chunk being read.

    The full cache is 18.1 GiB as float32 (1.9M tokens x 2560 x 4 B) and this machine has
    23.6 GiB of RAM in total, so nothing may materialise every window. What is held here is a
    list of `(chunk, start)` pairs, about 24 bytes each, and the arrays of the chunk being
    read. Windows are consumed in corpus order, so one resident chunk is enough and a step
    touches at most two.

    The cache chunks are in corpus order, so the corpus offset of a window is the sum of the
    token counts of the chunks before it. That is what lets the per-token weights and the
    source tags line up with the cache.
    """

    def __init__(
        self,
        cache: Path,
        run_dir: Path,
        seq_len: int,
        exclude: set[str],
        val_fraction: float,
    ) -> None:
        self.cache = cache
        self.seq_len = seq_len
        rows, self.weights = _corpus(run_dir)
        shards = _shards(cache)
        if not shards:
            raise RuntimeError(f"no harvested chunks in {cache}; run the harvest first")
        total = sum(int(shard["tokens"]) for shard in shards)
        holdout_start = int(total * (1.0 - val_fraction))
        self.specs: list[tuple[int, int, int, bool]] = []
        base = 0
        for shard in shards:
            index = int(shard["index"])
            t = int(shard["tokens"])
            for start in range(0, t - seq_len + 1, seq_len):
                corpus_start = base + start
                if exclude and (_sources_in(rows, corpus_start, corpus_start + seq_len) & exclude):
                    continue
                self.specs.append((index, start, corpus_start, corpus_start >= holdout_start))
            base += t
        self._index = -1
        self._h: np.ndarray | None = None
        self._ids: np.ndarray | None = None

    @property
    def train(self) -> list[tuple[int, int, int, bool]]:
        return [s for s in self.specs if not s[3]]

    @property
    def val(self) -> list[tuple[int, int, int, bool]]:
        return [s for s in self.specs if s[3]]

    def _load(self, index: int) -> None:
        if index == self._index:
            return
        h = np.fromfile(self.cache / f"chunk-{index:05d}.hidden.bin", dtype=np.float32)
        ids = np.fromfile(self.cache / f"chunk-{index:05d}.ids.bin", dtype=np.int32)
        t = min(len(ids), len(h) // EMBD)
        # A view, not a copy: the previous chunk is released when this reference drops.
        self._h = h[: t * EMBD].reshape(t, EMBD)
        self._ids = ids[:t]
        self._index = index

    def get(self, spec: tuple[int, int, int, bool]):
        index, start, corpus_start, _is_val = spec
        self._load(index)
        assert self._h is not None and self._ids is not None
        sl = slice(start, start + self.seq_len)
        return (
            torch.from_numpy(self._h[sl]),
            torch.from_numpy(self._ids[sl].astype(np.int64)),
            torch.from_numpy(self.weights[corpus_start : corpus_start + self.seq_len].astype(np.float32)),
        )


def _mask(n: int, batch: int, device: torch.device) -> torch.Tensor:
    mask = torch.zeros(batch, 1, n, n, dtype=torch.bfloat16, device=device)
    mask.masked_fill_(
        torch.triu(torch.ones(n, n, dtype=torch.bool, device=device), 1),
        torch.finfo(torch.bfloat16).min,
    )
    return mask


def _lm_loss(out, lm, ids, w, seq_len: int, chunk: int) -> torch.Tensor:
    """Weighted cross-entropy at t+2, one block of positions at a time.

    The full logits are [batch, seq_len, 248320]: 1.0 GB in bf16, 2.0 GB once upcast to fp32
    for the loss, before the softmax's own temporaries. Measured on the iGPU for one
    2048-token window: the monolithic version peaks at 10.3 GiB and takes 2.57 s; in
    512-position blocks it peaks at 6.0 GiB and takes 1.67 s. Backward reuses each block's
    saved tensors and then frees them, so the peak does not grow with the block count.

    The arithmetic is exact: every position's contribution is independent.
    """
    total = out.new_zeros(())
    for start in range(0, seq_len - 2, chunk):
        stop = min(start + chunk, seq_len - 2)
        logits = (out[:, start:stop] @ lm.T).float()
        per_token = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            ids[:, start + 2 : stop + 2].reshape(-1),
            reduction="none",
        )
        total = total + (per_token * w[:, start:stop].reshape(-1)).sum()
    return total


def _proxy(head, lm, windows: _Windows, val: list, seq_len: int, device: torch.device) -> float:
    """Greedy draft-match rate on the held-out windows. This is the selection metric.

    The logits go through the same 512-position blocks as the loss, and the matmul is bf16
    because the frozen LM head is bf16. Casting a block to float instead would copy the head
    (2.5 GB) and produce a 2 GB logits tensor for a number that only selects a checkpoint.
    """
    if not val:
        return 0.0
    hits = 0
    total = 0
    with torch.no_grad():
        for spec in val[:16]:
            h, ids, _w = windows.get(spec)
            h = h.to(torch.bfloat16)[None].to(device)
            ids = ids.to(device)
            embed = lm[torch.cat([ids[1:], ids[-1:]])][None]
            positions = torch.arange(seq_len, device=device)[None]
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                out = head(h, embed, _mask(seq_len, 1, device), positions)
            for start in range(0, seq_len - 2, LM_CHUNK):
                stop = min(start + LM_CHUNK, seq_len - 2)
                logits = out[0, start:stop].to(torch.bfloat16) @ lm.T
                target = ids[start + 2 : stop + 2]
                hits += int((logits.argmax(-1) == target).sum())
                total += int(target.numel())
    return hits / max(total, 1)


def _np_state() -> tuple:
    """The numpy RNG state in a form `torch.load(weights_only=True)` accepts.

    `np.random.get_state()` carries a 624-element uint32 ndarray, and the safe unpickler
    rejects ndarray globals. The same numbers as bytes round-trip exactly.
    """
    name, keys, pos, has_gauss, cached = np.random.get_state()
    return (name, keys.tobytes(), int(pos), int(has_gauss), float(cached))


def _np_restore(state: tuple) -> None:
    name, keys, pos, has_gauss, cached = state
    np.random.set_state(
        (name, np.frombuffer(keys, dtype=np.uint32), int(pos), int(has_gauss), float(cached))
    )


def _save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def run(
    config_path: Path,
    fail_after: int = 0,
    tag: str = "main",
    exclude: str = "",
    max_steps: int = 0,
    half: bool = False,
) -> None:
    cfg = tomllib.loads(config_path.read_text(encoding="utf-8"))
    run_dir = Path("runs/mtp-alignment") / cfg["run"]["run_id"]
    state = RunState(run_dir)
    lock = RunLock(run_dir / f"train-{tag}.lock")
    lock.acquire(force=(state.status("train") == "running"))
    exclude_sources = {s for s in exclude.split(",") if s}
    ckpt_dir = run_dir / "checkpoints" / tag
    try:
        torch.manual_seed(cfg["run"]["seed"])
        random.seed(cfg["run"]["seed"])
        np.random.seed(cfg["run"]["seed"])

        device = resolve_device(str(cfg["train"].get("device", "auto")))
        if device.type == "cuda":
            print(f"device: {device} - {torch.cuda.get_device_name(0)}")
            print(f"hip: {getattr(torch.version, 'hip', None)}  torch: {torch.__version__}")
        else:
            print(f"device: {device} - CPU fallback, no accelerator visible to torch")

        config = AutoConfig.from_pretrained(cfg["paths"]["model_id"], trust_remote_code=True)
        text = getattr(config, "text_config", config)
        # The master weights stay fp32. AdamW moves a parameter by about lr (2e-4) per
        # step; one bf16 ulp at 1.0 is 7.8e-3, so a bf16 master rounds every update to zero
        # for the tensors whose values sit near 1 or above. Measured: `shared_head_norm`,
        # whose values run 0.28-4.1, did not move by a single ulp in 216 steps, while
        # small-magnitude tensors did. bf16 is the compute dtype (autocast), not the storage
        # dtype for the thing being trained.
        head = MTPHead(text)
        hf_dir = Path(cfg["paths"]["hf_dir"]) if cfg["paths"].get("hf_dir") else None
        load_mtp_weights(hf_shards(cfg["paths"]["model_id"], hf_dir), head)
        head = head.to(device)
        # The tied output head, frozen. Also the input embedding: one matrix, gathered per batch.
        lm = load_lm_head(Path(cfg["paths"]["verifier_gguf"])).to(device)

        opt = torch.optim.AdamW(
            [p for p in head.parameters() if p.requires_grad], lr=float(cfg["train"]["lr"])
        )
        seq_len = int(cfg["train"]["seq_len"])
        val_fraction = float(cfg["corpus"].get("val_fraction", 0.05))
        windows = _Windows(run_dir / "cache", run_dir, seq_len, exclude_sources, val_fraction)
        train_windows = windows.train
        val_windows = windows.val
        if not train_windows:
            raise RuntimeError("no cached training windows; run harvest first")
        print(f"{len(train_windows)} train windows, {len(val_windows)} val windows")

        start_step = 0
        cursor = 0
        epoch = 0
        best_proxy = -1.0
        last = ckpt_dir / "last.pt"
        if last.exists():
            ckpt = torch.load(last, map_location=device)
            head.load_state_dict(ckpt["head"])
            opt.load_state_dict(ckpt["opt"])
            start_step = ckpt["step"]
            cursor = ckpt["cursor"]
            epoch = ckpt["epoch"]
            best_proxy = ckpt.get("best_proxy", -1.0)
            # The CPU generator's state must be a CPU ByteTensor; map_location put it on the GPU.
            torch.set_rng_state(ckpt["torch_rng"].cpu().to(torch.uint8))
            random.setstate(ckpt["py_rng"])
            _np_restore(ckpt["np_rng"])
            print(f"resumed at step {start_step}, epoch {epoch}, cursor {cursor}")

        batch = int(cfg["train"]["batch_size"])
        micro = max(1, int(cfg["train"].get("micro_batch", 1)))
        steps_per_epoch = max(1, len(train_windows) // batch)
        # epochs may be fractional: 0.01 is how the smoke test asks for "a step or two".
        total_steps = max(1, int(steps_per_epoch * float(cfg["train"]["epochs"])))
        ckpt_every = int(cfg["train"]["checkpoint_steps"])
        proxy_every = int(cfg["train"].get("proxy_every", 100))
        # A long run can be split: each invocation adds half of the *total* step count and
        # stops, leaving a checkpoint for the next one. From a clean start that is two
        # invocations (0 -> half -> total). After an interruption it takes one more, because a
        # resumed run still adds a full half rather than halving the remainder.
        stop_step = total_steps
        if max_steps:
            stop_step = min(stop_step, start_step + max(1, max_steps))
        if half:
            stop_step = min(stop_step, start_step + (total_steps + 1) // 2)
        if stop_step < total_steps:
            print(
                f"stopping at step {stop_step} of {total_steps}: this invocation runs "
                f"{stop_step - start_step} of the {total_steps - start_step} remaining",
                flush=True,
            )
        t0 = time.time()
        step = start_step
        while step < stop_step:
            if cursor >= len(train_windows):
                cursor = 0
                epoch += 1
            chunk = train_windows[cursor : cursor + batch]
            cursor += len(chunk)
            # One window at a time: the logits dominate memory, and gradient accumulation
            # makes up the effective batch.
            total_weight = (
                float(
                    sum(
                        float(windows.weights[c[2] : c[2] + seq_len - 2].sum())
                        for c in chunk
                    )
                )
                or 1.0
            )
            opt.zero_grad(set_to_none=True)
            loss_value = 0.0
            for micro_chunk in [chunk[i : i + micro] for i in range(0, len(chunk), micro)]:
                parts = [windows.get(spec) for spec in micro_chunk]
                h = torch.stack([p[0] for p in parts]).to(torch.bfloat16).to(device, non_blocking=True)
                ids = torch.stack([p[1] for p in parts]).to(device, non_blocking=True)
                w = torch.stack([p[2] for p in parts]).to(device, non_blocking=True)
                # The runtime feeds e_{t+1}: the head consumes h_t and the embedding of the
                # *next* token, and predicts t+2 (docs/notes/mtp-forward.md). `lm[ids]` is e_t,
                # which trains a function the runtime never runs. On one held-out window with
                # the stock head, e_{t+1} matched the t+2 token 79.4% of the time and e_t 3.7%.
                embed = lm[torch.cat([ids[:, 1:], ids[:, -1:]], dim=1)]
                positions = torch.arange(seq_len, device=device)[None].expand(len(micro_chunk), -1)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    out = head(h, embed, _mask(seq_len, len(micro_chunk), device), positions)
                    contrib = _lm_loss(out, lm, ids, w, seq_len, LM_CHUNK) / total_weight
                contrib.backward()
                loss_value += float(contrib.detach()) * len(micro_chunk) / len(chunk)
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            step += 1
            if fail_after and step >= fail_after:
                raise RuntimeError("injected failure: train")
            if step % 10 == 0 or step == stop_step:
                print(f"step {step}/{total_steps} loss {loss_value:.4f} {time.time()-t0:.0f}s", flush=True)
            if proxy_every and step % proxy_every == 0:
                proxy = _proxy(head, lm, windows, val_windows, seq_len, device)
                print(f"step {step} proxy acceptance {proxy:.4f}", flush=True)
                if proxy > best_proxy:
                    best_proxy = proxy
                    _save(
                        {"head": head.state_dict(), "step": step, "proxy": proxy, "tag": tag},
                        ckpt_dir / "best.pt",
                    )
            if step % ckpt_every == 0 or step == stop_step:
                _save(
                    {
                        "head": head.state_dict(),
                        "opt": opt.state_dict(),
                        "step": step,
                        "cursor": cursor,
                        "epoch": epoch,
                        "best_proxy": best_proxy,
                        "torch_rng": torch.get_rng_state(),
                        "py_rng": random.getstate(),
                        "np_rng": _np_state(),
                        "config": str(config_path),
                        "tag": tag,
                        "exclude": sorted(exclude_sources),
                        "device": describe_device(),
                    },
                    ckpt_dir / "last.pt",
                )
        if step >= total_steps:
            state.mark("train", "done", steps=step, tag=tag, device=describe_device())
        else:
            state.mark(
                "train",
                "partial",
                steps=step,
                total=total_steps,
                tag=tag,
                device=describe_device(),
            )
        print(f"train: tag={tag} steps={step}/{total_steps} best_proxy={best_proxy:.4f}")
    finally:
        lock.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--tag", default="main", help="checkpoint subdirectory name")
    parser.add_argument(
        "--exclude-source",
        default="",
        help="comma separated corpus sources to leave out (G8 control arm)",
    )
    parser.add_argument("--fail-after-steps", type=int, default=0)
    parser.add_argument(
        "--max-steps", type=int, default=0, help="stop after this many more steps (0: no limit)"
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="stop at half of the total run; a second invocation completes it",
    )
    args = parser.parse_args()
    run(
        args.config,
        fail_after=args.fail_after_steps,
        tag=args.tag,
        exclude=args.exclude_source,
        max_steps=args.max_steps,
        half=args.half,
    )

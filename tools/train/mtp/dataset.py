"""Build the training corpus: tokens, per-token weights, and source tags.

Three things have to survive this stage, because everything downstream depends on them:

- the token array, in the order the harvest will read it;
- a weight per token, 1 for user/system and 2 for assistant spans (spec section 7);
- a source tag per row, so G8 can train a control head from the same cache with the
  self-generated windows left out.

Rows are shuffled **as whole sequences** and never split across a row boundary, so every
recorded offset still points at its own tokens.
"""

from __future__ import annotations

import hashlib
import json
import random
import tomllib
from pathlib import Path
from typing import Iterable

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

SOURCES = {
    "nohurry": "nohurry/Opus-4.6-Reasoning-3000x-filtered",
    "roman": "Roman1111111/claude-opus-4.6-10000x",
    "jackrong": "Jackrong/Qwen3.5-reasoning-700x",
}

EOS = None  # set from the tokenizer in build()


def _hash_messages(messages: list[dict]) -> str:
    return hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()


def _require(row: dict, keys: tuple[str, ...], source: str) -> None:
    missing = [key for key in keys if key not in row]
    if missing:
        raise RuntimeError(f"{source}: row is missing {missing}; has {sorted(row)}")


def _nohurry(row: dict) -> list[dict]:
    _require(row, ("problem", "solution"), "nohurry")
    thinking = (row.get("thinking") or "").strip()
    answer = (row.get("solution") or "").strip()
    return [
        {"role": "user", "content": row["problem"]},
        {"role": "assistant", "content": (thinking + "\n\n" + answer).strip()},
    ]


def _roman(row: dict) -> list[dict]:
    _require(row, ("messages",), "roman")
    return [
        {"role": m.get("role", "user"), "content": m.get("content") or ""}
        for m in row["messages"]
    ]


def _jackrong(row: dict) -> list[dict]:
    _require(row, ("conversation",), "jackrong")
    out = []
    for turn in row["conversation"]:
        role = "user" if turn["from"] in ("human", "user") else "assistant"
        out.append({"role": role, "content": turn["value"]})
    return out


def _messages(row: dict) -> list[dict]:
    """A row that already carries chat messages. Used by the self-generated slice."""
    if "messages" in row:
        return row["messages"]
    if "conversations" in row:
        out = []
        for turn in row["conversations"]:
            role = "user" if turn.get("from") in ("human", "user") else "assistant"
            out.append({"role": role, "content": turn.get("value", "")})
        return out
    raise ValueError("row has neither messages nor conversations")


def _clean_native(messages: list[dict]) -> list[dict]:
    cleaned = []
    for m in messages:
        content = (m.get("content") or "").replace(" thinking", "").replace("</think>", "").strip()
        if not content:
            continue
        cleaned.append({"role": m.get("role", "user"), "content": content})
    return cleaned


def _render(tokenizer, messages: list[dict]) -> list[int]:
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, enable_thinking=True
        )
    except TypeError:
        # Older templates do not take enable_thinking. The rendered text is what matters;
        # a template without the kwarg is still the model's own template.
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def _render_weights(tokenizer, messages: list[dict], n_tokens: int) -> list[int]:
    """1 for user and system tokens, 2 for assistant tokens (spec section 7).

    The boundary of each assistant span is found by rendering the conversation's prefixes:
    prefix i ends where message i-1 ends. counts[-1] is the full render, by construction the
    same call that produced the token array, so the spans cannot drift off the end.

    The empty prefix is 0 tokens by definition; asking the chat template for it raises
    ("cannot apply a chat template to an empty conversation"). A prefix that the template
    rejects (a lone system message has no user query) keeps the previous boundary, which
    leaves that span at weight 1. Only assistant spans matter here, so this cannot shift
    an assistant boundary.
    """
    counts = [0]
    for i in range(1, len(messages) + 1):
        try:
            counts.append(len(_render(tokenizer, messages[:i])))
        except Exception:  # noqa: BLE001 - a template that rejects a prefix has no span
            counts.append(counts[-1])
    weights = [1] * n_tokens
    for i, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        start, end = counts[i], counts[i + 1]
        for j in range(min(start, n_tokens), min(end, n_tokens)):
            weights[j] = 2
    return weights


def build(config_path: Path) -> Path:
    cfg = tomllib.loads(config_path.read_text(encoding="utf-8"))
    corpus = cfg["corpus"]
    out = Path("runs/mtp-alignment") / cfg["run"]["run_id"] / "corpus"
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg["paths"]["model_id"], trust_remote_code=True)
    global EOS
    EOS = tokenizer.eos_token_id

    sources: list[tuple[str, float, Iterable[dict], callable]] = [
        (name, corpus[f"share_{name}"], load_dataset(dataset_id, split="train"), mapper)
        for name, mapper in (
            ("nohurry", _nohurry),
            ("roman", _roman),
            ("jackrong", _jackrong),
        )
        for dataset_id in (SOURCES[name],)
    ]

    total = int(corpus["total_tokens"])
    max_seq = int(cfg["harvest"]["max_seq"])
    rng = random.Random(cfg["run"]["seed"])
    seen: set[str] = set()
    seqs: list[tuple[list[int], list[int], dict]] = []

    def add(name: str, budget: int, rows_iter: Iterable[dict], mapper) -> int:
        used = 0
        skipped = 0
        reasons: dict[str, int] = {}

        def skip(reason: str) -> None:
            nonlocal skipped
            skipped += 1
            reasons[reason] = reasons.get(reason, 0) + 1

        for row in rows_iter:
            if used >= budget:
                break
            # Some rows are not a conversation this model can be run on: a null message body,
            # an assistant-first or tool-only thread, a template that rejects it. A corpus is
            # a sample of what the model can actually be run on, so those rows are skipped
            # and counted, not fatal.
            try:
                messages = mapper(row) if mapper is _messages else _clean_native(mapper(row))
            except Exception as exc:  # noqa: BLE001 - a malformed row is not a corpus
                skip(f"map:{type(exc).__name__}")
                continue
            if not messages or not any(m["role"] == "assistant" for m in messages):
                continue
            key = _hash_messages(messages)
            if key in seen:
                continue
            seen.add(key)
            try:
                ids = _render(tokenizer, messages)
            except Exception as exc:  # noqa: BLE001 - jinja raises TemplateError and friends
                skip(f"render:{type(exc).__name__}")
                continue
            if len(ids) < 64:
                continue
            # Truncate to the harvest's context. Attention is causal, so the hidden states
            # of a row's first max_seq tokens are the same whether or not the rest of the
            # row follows. A row longer than that is kept as its prefix. The EOS appended
            # below is part of the row, so the prefix is max_seq - 1: a row of max_seq + 1
            # would not fit the harvest's context.
            truncated = len(ids) > max_seq - 1
            ids = ids[: max_seq - 1]
            if len(ids) < 64:
                continue
            if used + len(ids) > budget:
                continue
            weights = _render_weights(tokenizer, messages, len(ids))
            seqs.append(
                (
                    ids,
                    weights,
                    {"source": name, "hash": key, "tokens": len(ids), "truncated": truncated},
                )
            )
            used += len(ids) + 1
        if skipped:
            print(f"{name}: skipped {skipped} rows ({reasons})")
        return used

    shortfalls = {}
    for name, share, ds, mapper in sources:
        budget = int(total * share)
        used = add(name, budget, ds, mapper)
        print(f"{name}: {used} tokens (budget {budget})")
        if used < budget * 0.95:
            shortfalls[name] = (used, budget)

    selfgen = Path(cfg["paths"]["selfgen_jsonl"])
    if selfgen.exists() and corpus["share_selfgen"] > 0:
        def gen_rows():
            with selfgen.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        yield json.loads(line)

        used = add("selfgen", int(total * corpus["share_selfgen"]), gen_rows(), _messages)
        print(f"selfgen: {used} tokens")
    else:
        print("selfgen: 0 tokens (missing file or share is 0; G8 runs the 0% control)")

    # Shuffle whole sequences, not rows. The row list and the token array keep the same
    # order, so every recorded offset still points at its own tokens.
    rng.shuffle(seqs)
    tokens: list[int] = []
    weights: list[int] = []
    rows: list[dict] = []
    for ids, seq_weights, meta in seqs:
        offset = len(tokens)
        tokens.extend(ids)
        tokens.append(EOS)
        weights.extend(seq_weights)
        weights.append(1)
        meta["offset"] = offset
        meta["length"] = len(ids) + 1
        rows.append(meta)
    arr = np.asarray(tokens, dtype=np.uint32)
    warr = np.asarray(weights, dtype=np.uint8)
    np.save(out / "tokens.npy", arr)
    np.save(out / "weights.npy", warr)
    with (out / "rows.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    by_source = {
        name: sum(1 for row in rows if row["source"] == name)
        for name in sorted({row["source"] for row in rows})
    }
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "total_tokens": int(arr.size),
                "rows": len(rows),
                "eos": EOS,
                "max_seq": max_seq,
                "by_source": by_source,
                "shortfalls": {k: {"used": v[0], "budget": v[1]} for k, v in shortfalls.items()},
                "sources": sorted(by_source),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {arr.size} tokens to {out}")
    if shortfalls:
        print(f"WARNING: sources short of budget: {shortfalls}")
    return out


if __name__ == "__main__":
    import sys

    build(Path(sys.argv[1]))

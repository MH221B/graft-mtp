# graft-mtp

An MTP (multi-token prediction) head grafted onto a quantized Qwen3.5-4B merge, so that
`llama.cpp` can use **built-in speculative decoding** without a second model.

**Model:** `Athena-4B-MTP.i1-Q4_K_S.gguf` - 441 tensors, 33 blocks, 2.61 GiB.

| | |
|---|---|
| Base | `xtools-at/Qwen3.5-4B-MultiDistill-3xMerge-UNCENSORED-heretic` (i1-Q4_K_S by mradermacher) |
| Head | block 32, from `RBergBauer/Qwen3.5-4B-MTP-Heretic`, grafted at f16 |
| License | Apache-2.0 (see `NOTICE`) |
| Measured | **82.61%** draft acceptance at `--spec-draft-n-max 2`, **43.5 tok/s** vs 29.1 without speculation (1.49x) |

## Status - read this first

The head is the **stock** MTP head, not a retrained one. A realignment attempt was run and
**failed its acceptance gate**: 63.18% at `n_max 2` against the required 87.6% (the stock head's
82.61% plus 5 points). It is not shipped. The graft is what makes the file work: the published
target declares 33 blocks but ships 32, so `llama.cpp` refuses it with
`tensor 'blk.32.attn_norm.weight' not found`. Block 32 is supplied here.

Two things were fixed relative to a naive graft, and they matter:

- **Norm convention.** Qwen3.5 normalises with `(1 + w)`, `llama.cpp` with `w`, so the GGUF must
  store `w + 1`. A verbatim copy makes block 32 compute a different function: measured
  0 accepted of 187 drafted.
- **The head is f16.** The 8 2D tensors are f16, the 7 norms F32. The body stays i1-Q4_K_S.

## Download

**Release:** https://github.com/MH221B/graft-mtp/releases/tag/v1.0

The GGUF is 2,805,112,832 bytes, above GitHub's 2 GiB release-asset limit, so the release ships it
in two parts. Reassemble, then verify.

**Windows**

```bat
copy /b Athena-4B-MTP.i1-Q4_K_S.gguf.part1 + Athena-4B-MTP.i1-Q4_K_S.gguf.part2 Athena-4B-MTP.i1-Q4_K_S.gguf
```

**Linux / macOS**

```bash
cat Athena-4B-MTP.i1-Q4_K_S.gguf.part1 Athena-4B-MTP.i1-Q4_K_S.gguf.part2 > Athena-4B-MTP.i1-Q4_K_S.gguf
```

**Verify**

```bash
sha256sum Athena-4B-MTP.i1-Q4_K_S.gguf
# a2544866bddd9afcee62aa97affb85594dfda63eb4614593b2dd4e68ff812b1b
```

## Run with llama.cpp

Any build with the MTP speculative path. Verified on commit `4364bf723` (build 11228), where the
flags are `--spec-type draft-mtp` and `--spec-draft-n-max`.

**Server**

```bash
llama-server \
  -m Athena-4B-MTP.i1-Q4_K_S.gguf \
  -c 4096 -ngl 99 -fa on --jinja \
  --spec-type draft-mtp --spec-draft-n-max 2 \
  --port 8080
```

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is 17 times 3?"}],"max_tokens":64}'
```

**CLI**

```bash
llama-cli \
  -m Athena-4B-MTP.i1-Q4_K_S.gguf \
  -p "What is 17 times 3?" -n 64 -ngl 99 --temp 0.7 \
  --spec-type draft-mtp --spec-draft-n-max 2
```

**Choosing `n_max`.** Measured on this GPU (Radeon 890M, 16.9 GiB shared):

| `--spec-draft-n-max` | acceptance | accepted length | tok/s |
|---|---|---|---|
| 1 | 88.80% | 1.888 | 38.7 |
| 2 | 82.61% | 2.652 | **43.5** |
| 3 | 72.47% | 3.174 | 42.3 |
| no speculation | - | - | 29.1 |

`2` is the best throughput here. On a faster GPU, `3` may win: the drafts cost less there.

**Vision.** The base model is multimodal. The projector ships with the release as
`Qwen3.5-4B-MultiDistill-3xMerge-UNCENSORED-heretic.mmproj-Q8_0.gguf` (350 MiB, sha256
`c9605ff1…`); pass `--mmproj <file>` if you need image input.

**Builds without MTP.** The file still loads; `blk.32.*` is reported as unused and decoding falls
back to normal. Nothing breaks.

## How the draft works

The head is block 32: a full-attention block, shape-identical to block 31, plus four `nextn`
tensors. It consumes the verifier's post-output-norm hidden state `h_t` and the embedding of the
**next** token `e_{t+1}`, and predicts token `t+2`. It has its own KV cache. The 32-block verifier
then accepts a prefix of each draft.

`docs/notes/mtp-forward.md` pins the exact forward from the llama.cpp source.

## Repository layout

```text
tools/train/mtp/          graft, harvest, trainer, packaging, eval harness
docs/superpowers/specs/   the design of record (11 revisions)
docs/superpowers/plans/   the 17-task implementation plan and its outcome
docs/notes/mtp-forward.md the pinned MTP forward
patches/llama.cpp/        the nextn-dump patch for llama-imatrix (harvest)
bench/mtp-alignment/      measured acceptance JSON and a summary
```

## Reproducing the measurement

The harness drives `llama-server` and reads `draft_n` / `draft_n_accepted` from the response
timings, so the numbers come from the runtime, not from a reimplementation.

```powershell
python -m tools.train.mtp.eval_accept tools/train/mtp/configs/mtp-alignment.toml \
    Athena-4B-MTP.i1-Q4_K_S.gguf bench/out.json --label B
```

100 prompts, `T=0.7`, 128 new tokens, `n_max` 1/2/3, plus a no-speculation baseline.

## Limits

- 100 prompts, one GPU, one quantization. Treat the numbers as this machine's, not universal.
- The aligned head is absent; see `docs/superpowers/specs/` revision 11 for what was tried and why
  it failed.
- The 1D head norms are F32 and the 2D head tensors f16; the head is deliberately not quantized.

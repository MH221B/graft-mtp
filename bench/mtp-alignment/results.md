# Measurements

All numbers come from `llama-server` itself: the harness reads `draft_n` and `draft_n_accepted`
from each response's `timings`, so acceptance is the runtime's own count, not a reimplementation.

Hardware: Radeon 890M (16.9 GiB shared), llama.cpp `4364bf723` (build 11228), Vulkan, `-ngl 99`,
`-fa on`. 100 prompts, `T=0.7`, 128 new tokens, `n_max` 1/2/3, plus a no-speculation baseline.

## Shipped head (artifact B, stock MTP head)

| n_max | drafted | accepted | acceptance | accepted length | tok/s |
|---|---|---|---|---|---|
| 1 | 6,608 | 5,868 | 88.80% | 1.888 | 38.73 |
| 2 | 9,355 | 7,728 | **82.61%** | 2.652 | **43.48** |
| 3 | 11,824 | 8,569 | 72.47% | 3.174 | 42.25 |
| no speculation | - | - | - | - | 29.12 |

At `n_max` 2 the draft head is worth **1.49x** over plain decoding.

Raw data: `B.json`, `B-nospec.json`.

## Rejected: the realigned head (artifact D)

Trained for 216 steps on 1.9M harvested tokens, spliced into a copy of B, measured on the same 100
prompts. It is worse on every row.

| n_max | acceptance | accepted length | tok/s |
|---|---|---|---|
| 1 | 79.95% | 1.800 | 37.75 |
| 2 | 63.18% | 2.264 | 37.54 |
| 3 | 46.63% | 2.399 | 32.58 |

The gate needed 87.6% at `n_max` 2 (B's 82.61% plus 5 points). D is 24.4 points short, and slower
than B at every `n_max`. The trainer was verified correct before this run: the input pairing
(`e_{t+1}`, not `e_t`) and fp32 master weights were both fixed, and all 15 head tensors moved.

The failure mode is worth recording: weighted CE fell from 0.674 to 0.09 while acceptance fell 19 points. The head became more confident about the same predictions - it fitted the corpus's next-next distribution rather than the verifier's behaviour on unseen prompts.

Raw data: `D-rejected.json`.

## Fidelity (G1)

`fidelity.json`: PyTorch against llama.cpp's own MTP context on a 256-token probe, f32.
`h_out` max delta 1.888 (tolerance 3.0), draft-token match 0.996 (min 0.99), logits max delta 0.574
(tolerance 1.0). The cache reconstruction matched the runtime exactly (delta 0.0).

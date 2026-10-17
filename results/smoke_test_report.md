# Distributed neuron map: smoke test report

Branch `research/distributed`. Code in `src/dist/`; how to run it is in `RUNBOOK.md`. This report fills in phase by
phase.

## Phase A: local, RTX 5090 ($0), done 2026-10-17

**Setup**
- Qwen3-8B, bf16, pinned to Hugging Face revision `b968826d9c46dd6066d109eabc6255188de91218`. The local copy's 5
  safetensors files match that revision's sha256.
- Store `D:/model-mapper-runs/smoke_5090`, a local folder.
- Tasks:
  - 2 observe shards of about 1M tokens each, from the same FineWeb documents, in the same order, as
    `data/atlas_Qwen3-8B.pt`;
  - `merge:observe`;
  - `ablate:18:0-63:ps0`, 64 neurons with 10 top + 5 bottom firings each and 10 shared random control contexts, using
    a 128-token window.

**Throughput (5090)**

| Task | Time | Tokens | Tokens/s | Output |
|---|---|---|---|---|
| observe shard (1M tokens, K=40) | 303–305 s | 998k | 3,275–3,290 | 157 MB per shard |
| merge:observe (2 shards) | 27 s | – | – | 166 MB |
| ablate, 64 neurons | 29 s | 256k | 8,850 | 6 MB |

- Model load from local disk took 17 s.
- Observe speed is close to the original `build_atlas.py` run (about 3,850 tokens/s with 6,144-token batches; here 4,096).
  Observe also records head statistics and the per-neuron mean, variance and sparsity.

**Checks**
- **Same as the existing atlas.** 181,697 neuron poles have their atlas peak inside our first 2M tokens. In every one
  (100.00%), our top firing is the same token. Values are bit-identical at the 99th percentile; the largest relative
  difference is 5%.
- **Merge is exact.** CPU, Qwen3-0.6B: 2 shards merged equals 1 shard over the same documents. The top-k sets are
  identical and the 512 ablate rows identical.
- **Ablate is right.** 8 random rows recomputed the `meaning_jobs.py` way (separate base and ablated forward passes,
  batch size 1) matched in activation and KL to 4–5 significant figures (CPU, 0.6B).
- **Kill and resume.** A task killed at 20 s and at 37 s left no `_DONE`. The rerun gave byte-identical output (29/29
  files), and a further rerun skipped the task.
- **Launcher.** Against a fake RunPod API it replaced an interrupted pod, stopped at the spend cap, and ended with zero
  pods.
- **R2 free-tier caps.** A tiny storage cap stopped the worker before its first task. Draining copied finished results
  out byte-identically, while keeping the ones a pending task still needed.

**Window benchmark** (`src/dist/window_bench.py`)

How much text before a firing does the ablate task need? We used 800 top firings: 32 random neurons × 5 firings in
each of layers 2, 10, 18, 26 and 34. For each, the neuron was knocked out with a window of W tokens, and the result
was compared with knocking it out given the whole document up to the firing token, as `meaning_jobs.py` does.

| Window | Median activation vs full | Firings keeping ≥80% | ≥50% | KL rank correlation with full | Median KL ratio | Cost vs full |
|---|---|---|---|---|---|---|
| 32 | 0.96 | 69% | 84% | 0.80 | 1.00 | 0.10 |
| 64 | 1.00 | 81% | 92% | 0.87 | 1.00 | 0.19 |
| **128** | 1.00 | 90% | 95% | 0.91 | 1.00 | 0.34 |
| 256 | 1.00 | 95% | 97% | 0.95 | 1.00 | 0.57 |
| full | 1.00 | 99%* | 99%* | 1.00 | 1.00 | 1.00 |

\* The full prefix is the reference. The 1% are neurons whose top "firing" is about 0, most likely near-silent neurons
in layer 2.

At 128 tokens, 90% of firings keep at least 80% of their strength. The knockout effect is unchanged at the median and
ranks almost the same (0.91). It costs about a third of the full prefix. 256 tokens buys 5 points more at 1.7× the
cost. **Recommendation: 128.** Each row records the activation inside its window, so firings that weaken can be
flagged.

**Knockout noise floor (8B, bf16): a design issue to decide** (`src/dist/noise_floor.py`)

These are 128 top firings in layer 18, each knocked out at its firing token with a 128-token window.

| Measurement | Median KL | 90th percentile |
|---|---|---|
| Noise only: same context unablated, run alone vs inside a batch of 32 | 3.2e-4 | 1.9e-3 |
| Single-neuron knockout at the firing token | 7.3e-4 | 8.3e-3 |

- The typical knockout effect is only about 2× the rounding noise that batch shape alone causes. 27% of knockouts
  exceed the noise's 90th percentile, and 16% exceed 10× its median.
- Paired knockouts (each next to its unablated copy in the same pass) agree perfectly across batch sizes 8 and 16
  (Spearman 1.000), and 0.89 with each pair run alone. Comparing with the `meaning_jobs.py` way (separate passes,
  batch size 1) gives only 0.58. That matches what the main session found for batched concept knockouts in map_loop.
- **Conclusion.** A single neuron zeroed at a single token is mostly below what bf16 can resolve. Per-row KLs and
  top-token lists for small effects are largely noise, and remote-vs-5090 differences will be of this size.
- **Options (James's call; this changes the mapping measure):**
  1. Add a null control to every task (an unablated context in a second batch shape), so every result carries its own
     noise floor.
  2. Aggregate per neuron over its firings, and flag neurons whose effect isn't above the floor.
  3. Use a stronger intervention, e.g. zero the neuron at every firing position in the window, not just the last.
  4. Run the layers after L in float32. This fits on the 5090 but not on 24 GB cards.

**Early full-map arithmetic** (5090 speeds; rented cards come in phase C)
- **Observe:** 10M tokens at about 3,280 tokens/s ≈ 51 min of one 5090. That's small.
- **Ablate:** all 442,368 neurons = 6,912 tasks of 64 neurons at about 29 s each ≈ 56 GPU-hours on a 5090.
  **Ablation dominates the cost.** Starting the knockout rows from the cached residual at layer L (a TODO) would
  cut this, especially for late layers.
- **Output:** about 6 MB per ablate task, so about 40 GB in total. The launcher's drain-to-local keeps the bucket under
  the free tier.
- **Attention-head knockouts:** not built yet (TODO).

**Open questions from phase A**
- **GPU memory per worker.** The 5090 showed 30 GB in use, mostly PyTorch's cache. Each task now records its peak
  memory; the 24 GB cards are measured in phase C.
- **Self-termination.** It's unverified whether a pod's own key can terminate it (RunPod's docs don't say). Phase C
  checks this. The launcher also terminates finished pods as a backstop.

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

**Knockout redesign, tested on 8B** (`ablate:18:0-63:ps1`)

Every task now carries its own noise floor. Each firing is knocked out two ways: at the firing token only (LAST),
and at every token in the window where the neuron fires at ≥ half its median top firing (PASSAGE).

| | Rows | Median KL | Share above floor p90 | Share above 10× floor median | Tokens zeroed (median) |
|---|---|---|---|---|---|
| Noise floor (32 contexts, alone vs batched) | 32 | 3.3e-4 (p90 1.8e-3) | – | – | – |
| LAST, top firings | 640 | 8.8e-4 | 31% | 18% | 1 |
| LAST, bottom firings | 320 | 9.4e-4 | 29% | 17% | 1 |
| LAST, random control tokens | 640 | 3.8e-4 | 7% | 0% | 1 |
| PASSAGE, top firings | 640 | 7.3e-4 | 33% | 18% | 1 |
| PASSAGE, random control tokens | 640 | 0 | 0% | 0% | 0 |

- **Controls behave like the floor.** Knocking out a neuron on random tokens, where it doesn't fire, gives about the
  noise floor: zeroing a near-zero value still perturbs bf16 by that much. So each neuron's control rows can serve as
  its own null.
- **PASSAGE barely differs from LAST.** In a 128-token window, the median top firing has only one token at ≥ half
  strength, so top firings are mostly single-token spikes.
- **Per neuron:** 11 of 64 (LAST) and 15 of 64 (PASSAGE) have a median top-firing KL above the task's floor p90; 4 of 64
  have one above 3× it. At layer 18, most single neurons' effect on the next token is at bf16's noise level.
- **Same estimate, two batch layouts:** a knockout measured in the task's batch and alone ranks the same (Spearman 0.90).
- **Speed and memory:** 44.8 s for 64 neurons with both modes, 9,290 tokens/s. Peak GPU memory 17.6 GB allocated
  (20.8 GB reserved), so the ablate task fits on 24 GB cards.

**Early full-map arithmetic** (5090 speeds; rented cards come in phase C)
- **Observe:** 10M tokens at about 3,280 tokens/s ≈ 51 min of one 5090. That's small.
- **Ablate:** all 442,368 neurons = 6,912 tasks of 64 neurons. That's about 29 s each with one mode (≈ 56 GPU-hours on a 5090),
  or about 45 s with both modes and the noise floor (≈ 86 GPU-hours).
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

## Phase B: real bucket (R2 `modelmapperdistributed`), 2026-10-17

- **No custom image.** Pods start from the public `pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime` image and fetch the
  committed worker code from the bucket, so no registry and no Docker are needed. `Dockerfile` remains as an option.
- **Bucket key.** Read, write, list and delete all work. The key cannot list other buckets (AccessDenied), so it is
  limited to this one bucket.
- **Worker against the bucket.** The CPU smoke test (Qwen3-0.6B: 2 observe shards, merge, 64-neuron ablate with both
  knockout modes and the noise floor) ran from the desktop against `s3://modelmapperdistributed/cpu-smoke`. It took
  4 min 38 s, and every output array is identical to the same run on a local folder.
- **Drain.** It moved all 0.039 GB to `D:/model-mapper-runs/cpu-smoke` and left 0 GB in the bucket. The run's state is
  unchanged, with no tasks pending.
- **Free-tier use so far:** 138 Class A and 85 Class B requests this month (caps 900k / 9M).
- **RunPod key.** RunPod's Cloudflare front end blocks Python's default user-agent (error 1010), so the client now
  sends its own. The REST API (`rest.runpod.io/v1`) needs the key's GraphQL permission set to Read/Write;
  `api.runpod.ai` is Serverless only and not needed.

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

## Phase C: rented GPUs (RunPod Community, on-demand), 2026-10-17

Each run repeated the phase A tasks on the same corpus with the same settings: 2 observe shards of about 1M tokens,
`merge:observe`, and `ablate:18:0-63:ps1` with both knockout modes and the noise floor.

**What changed from the plan**
- **No spot pods.** RunPod rejected interruptible creates ("Spot pods are no longer offered"), so all pods are
  on-demand. Interruption handling stays in place for crashes and stuck hosts.
- **Capacity is thin.** Often none of a given GPU type was available for 15–20 minutes. The RTX A5000 never became
  available in two 20-minute windows, and for the RTX 4090 the only machine offered was faulty (below).
- **Decision (James): no pinning to one GPU type.** The launcher takes a ranked list (`--gpu A,B,C`), RunPod starts the
  first one it has, and every result records its GPU, driver, CUDA, torch and git commit.
- **Image.** RunPod's own `runpod/pytorch:1.4.0-cu1281-torch2121-ubuntu2204` (torch 2.12.1+cu129, against 2.11+cu128 on
  the 5090). Docker Hub's `pytorch/pytorch` image never started on the first host, which was most likely stuck on the
  download.

**Runs** (`src/dist/timeline.py`)

| | Remote RTX 5090 (FR) | RTX 3090 (ES) |
|---|---|---|
| $/h | 0.69 | 0.22 |
| Pod created → container running (scheduling and image) | 67 s (image cached) | 287 s (image pulled) |
| Boot script (boto3, code, packages, imports) | 20 s | 26 s |
| Model download, 16.4 GB from Hugging Face | 44 s (377 MB/s) | 133 s (123 MB/s) |
| Observe, 1M tokens | 649–656 s, **1,530 tok/s** | 826–838 s, **1,200 tok/s** |
| Merge (2 shards) | 68 s (40 s of transfers) | 88 s |
| Ablate, 64 neurons, both modes and noise floor | 118 s | 113 s |
| Peak GPU memory (allocated) | 20.5 GB | 20.4 GB |
| Lifetime / cost | 27.3 min / **$0.31** | 39.9 min / **$0.15** |
| Share of lifetime spent on tasks | 91% | 78% |

- **The remote 5090 is half the speed of the local 5090** (1,530 vs 3,280 tok/s, and 118 s vs 45 s for ablate), with
  the GPU 100% busy. The likely causes are a power cap or the newer torch. Speed varies host to host, so plan on
  measured rates.
- **Per dollar the 3090 wins about 2.5×:** 1,200 tok/s at $0.22/h is ≈19.6M tokens per dollar, against ≈8M for the
  remote 5090.

**Consistency with the local 5090** (`src/dist/compare_runs.py`, all 36 layers, 884,736 neuron poles)

| | Remote 5090 | 3090 |
|---|---|---|
| Top-40 firings shared (set overlap) | 99.99% | 97.7% |
| Identical top-40 set | 99.6% | 34% (47% differ by 1 entry, 15% by 2) |
| Rank of a swapped-out entry | – | median 39 of 40; 96% in ranks 31–40; 0.5% in the top 10 |
| Same #1 firing | – | 96.2% |
| Order of shared entries (rank correlation) | 1.000 (97% identical order) | 0.9935 (largest move median 4 places) |
| Activation values, median relative difference | 0 | 0.6% (about 1–2 bf16 steps) |
| Attention heads, top-40 overlap | 100% | 96.5% |
| Ablate KL, Spearman | 0.96 | 0.90 |

- **The same chip family (Blackwell) reproduces the local atlas essentially exactly.** An older chip family (Ampere)
  gives the same top firings, reordered among near-ties and swapped at the 40th place, with values within about 1–2
  bf16 steps.
- **Practical rule:** for "what does this neuron respond to", any 24 GB+ card is fine. For bit-level comparisons with
  the local atlas, use a Blackwell card.

**Problems found and fixed** (each commit is on `research/distributed`)
1. **Spot pods are no longer offered.** On-demand is now the default (`--interruptible` keeps the option).
2. **RunPod's Cloudflare front end blocks Python's user-agent (error 1010).** The client sends its own.
3. **The REST API needs the key's GraphQL permission** (Read/Write); `api.runpod.ai` is Serverless only.
4. **No capacity used to crash the launcher.** It now retries each interval up to `--max-wait-min` and recognises
   both of RunPod's wordings.
5. **A container stuck before boot cost about $0.08.** Now:
   - every boot step writes a timestamped log to the bucket through a presigned URL;
   - `--boot-timeout-min` terminates a pod whose worker never starts.
6. **One faulty RTX 4090 host (Taiwan, machine `4aveufxk279b`): "CUDA unknown error".** torch fell back to CPU and a pod
   spent about $0.28 computing on CPU at 0% GPU. Now:
   - a GPU matmul check runs before the model download, and a failure terminates the pod at once;
   - the worker refuses to run on CPU on a pod;
   - the launcher terminates any pod whose boot log says FAILED, and remembers machines where the GPU check failed.
7. **Self-termination.** The REST DELETE with the pod-scoped key is refused (403), and RunPod's image has an older
   `runpodctl`. `runpodctl remove pod $RUNPOD_POD_ID` works and is now tried first. The launcher's backstop
   terminates any pod whose worker wrote `exit.json`.
8. **Hung workers.** The worker writes a heartbeat every 2 minutes, and the launcher terminates a pod after
   `--stale-min` of silence.
9. **Account backstop.** The launcher reads the RunPod balance every interval (`--min-balance` stops everything below a
   floor).

**Spend at the end of phase C:** $0.72: 4090 attempts about $0.37 (faulty host), 5090 $0.31, 3090 $0.15, A5000 $0. Zero
pods were left after each run.

**RTX 4090 (benchmark retry, healthy host)**
- 1,705–1,713 tok/s delivered, and about 2,100 tok/s of pure compute (476 s per 1M tokens). Ablate took 70 s for 64
  neurons. Lifetime 32.8 min, **$0.19**.
- This host's upload was slow (about 1.5 MB/s): 107 s per 164 MB shard result, and 175 s of the merge's 223 s went on
  transfers.
- Compute per dollar is the best of the cards tested, at about 22M tokens per dollar.
- Against the local 5090 it behaves like the 3090: top-40 overlap 97.6%, value difference 0.6%, ablate KL Spearman 0.88.
- **Hosts:** 2 of the 4 RTX 4090 hosts we were given had broken CUDA ("CUDA unknown error"). The GPU check caught both
  within a minute.

**Card summary (per dollar, this workload)**

| Card | $/h | Observe tok/s (delivered) | Ablate, 64 neurons | Tokens per $ (observe) | Matches the local 5090 bit for bit? |
|---|---|---|---|---|---|
| RTX 3090 | 0.22 | 1,200 | 113 s | ≈19.6M | no (≈0.6% values, near-tie reorders) |
| RTX 4090 | 0.34 | 1,710 (2,100 compute) | 70 s | ≈18M (22M compute) | no (same as the 3090) |
| RTX 5090, rented | 0.69 | 1,530 | 118 s | ≈8M | yes |
| RTX A5000 | 0.16 | – | – | – | none available in two 20-minute windows |

## Phase D: fan-out, 2026-10-17

**Free local test** (3 CPU workers with Qwen3-0.6B on 20 tasks in the real bucket; one killed mid-task and a
replacement started 20 s later)
1. **First run.** All 20 tasks completed, and the killed task was retaken after about 85 s. But **5 of the 20 tasks ran
   twice**: workers that finish together pick the same free task. The merge is the worst case, since every waiting
   worker takes it the moment it becomes ready.
2. **Fix: claim, pause, verify.** The bucket keeps the last write, so both racers read back the same winner. One
   duplicate remained, from a stale listing.
3. **Fix: re-read the claim just before claiming.** The final run had **0 duplicates**. The killed task was retaken by
   the replacement 76 s after the kill (test settings: 15 s heartbeat, 45 s claim expiry), and every worker exited
   cleanly when the queue emptied.
- **Also fixed:** a claim used to block a dead worker's task for 45 minutes. Claims now expire 6 minutes after their
  last refresh, and the worker's heartbeat refreshes its claim every 2 minutes.

**Paid test** (2 rented machines on 19 short 8B tasks: 8 observe shards of 150k tokens, a merge, 10 ablate tasks of 16
neurons; GPU list 3090 → 4090 → 5090; $1.50 cap and $7 balance floor)
- **Machines.** RunPod started an RTX 5090 ($0.69/h) and an RTX 4090 ($0.34/h), because no 3090 was free.
- **The 4090 host was faulty** ("CUDA unknown error"). It failed its GPU check 50 s after boot and terminated itself
  with `runpodctl remove pod`. The launcher started a replacement (an RTX 3090) within the minute.
- **Simulated interruption.** The test terminated that 3090 mid-run. The launcher kept trying to replace it (no
  capacity for about 3 minutes) while the surviving 5090 kept working.
- **The replacement landed on the known-bad Taiwan machine** and was released immediately (bad-host list). That was the
  test's 4th and last allowed launch, so the 5090 finished the rest alone.
- **Result:** all 19 tasks done, 0 run twice, leftover claims cleared, results drained to D:, and **zero pods** at the
  end. Cost **$0.42**.

**Total smoke-test spend:** $1.32 of $10 (balance $8.68). Zero pods after every run.

## Loop v2 on this runner (2026-10-17)

`map:<batch>` runs the research branch's loop v2 (`src/loop_v2/map_core.py`) on the pinned model, about 20 concepts per
task.
- **Check against the gate.** On the local 5090, the 12-concept repro set as one map task was **identical** to loop v2's
  own gate run: every v1 field, top-30 single drops, head knockouts, null drops, roles and every array, bitwise. It took
  24 s per concept, peaked at 17.6 GB, and produced 81 MB.
- **Incremental runs.** `make_run --kind map --append --watch N` adds specs as they appear, and `_PREPARE_DONE` closes
  the run. Workers and the launcher wait while a run is open. Tested on CPU: 12 specs trickled in, 6 batches were all
  done, the run closed and the worker exited.

## Projections for the full neuron map (Qwen3-8B, 442,368 neurons)

Assumptions:
- observe over 10M tokens (the atlas's corpus) as 10 shards of 1M;
- ablate as 6,912 tasks of 64 neurons, each with both knockout modes and the noise floor;
- measured per-task times from tonight, on-demand Community prices, and about 6 minutes of boot per pod.

| | RTX 3090 ($0.22/h) | RTX 4090 ($0.34/h) |
|---|---|---|
| Observe, 10M tokens | 2.3 GPU-h, $0.51 | 1.6 GPU-h, $0.55 |
| Ablate, 6,912 × 64 neurons | 217 GPU-h (113 s/task), $48 | 134 GPU-h (70 s/task), $46 |
| **Total GPU work** | **≈220 GPU-h ≈ $48** | **≈136 GPU-h ≈ $46** |
| Wall clock at N = 10 pods | ≈22 h | ≈14 h |
| N = 20 | ≈11 h | ≈7 h |
| N = 50 | ≈4.5 h | ≈3 h |

Boot overhead adds about N × 6 min of pod time (≈$0.20–1.70), and merging is small.

- **Ablation is about 98% of the cost.** The `TODO` shortcut, starting the knockout rows from the cached residual at
  layer L, would cut late-layer tasks most.
- **The cards cost about the same per unit of work.** Choose by availability; the 4090 finishes sooner.
- **Capacity is the real limit.** Tonight even 2 simultaneous machines needed retries, and 2 of 4 RTX 4090 hosts were
  faulty. 50 Community pods at once is unlikely, so 10–20 with the mixed list is realistic. Secure Cloud is 2–3×
  pricier (4090 $0.89/h).
- **R2:** about 6 MB per ablate task means about 40 GB in total, so drain-to-local keeps the bucket inside the free
  tier. Requests are about 30 per task, comfortably under 1M a month.

**Recommended task length: 15–30 minutes on the cheapest card.**
- Pods run many tasks each, so boot amortises over a pod's life, not per task. Task length then only sets how much a
  lost pod wastes and how evenly the tail finishes.
- Spot pods aren't offered, so interruptions are rare (stuck or faulty hosts) and longer tasks are fine.
- **Concretely:**
  - observe: 1M-token shards (8–14 min);
  - ablate: about 512 neurons per task (8 × 64, ≈10–15 min on a 4090), not 64 (1–2 min, too many tiny objects and
    polls);
  - loop v2: 20 concepts per task (≈8 min locally, ≈15–25 min on cloud).

**Open items and blockers for scale**
1. **Observe output is 157 MB per shard** (K=40, fp32 values). Store values as fp16 and upload in the background while
   the next task runs: on slow-upload hosts transfers were 20% of task time.
2. **The rented 5090 ran at half the local speed.** Speed varies by host, so plan on measured rates and keep timing
   per task (done: `timeline.py`).
3. **Learn bad hosts from self-terminated pods too**, not only those the launcher catches mid-boot.
4. **Attention-head knockouts aren't built** (TODO in `tasks.py`).
5. **Single-token knockouts on 8B are mostly at the bf16 noise floor** (phase A). Every task now carries its own floor,
   but the output-side measure is weak for most single neurons, and that is a research question rather than an
   infrastructure one.

## First production run: loop v2, 5,000 concepts (night of 2026-10-17/18)

- **Inputs.** Vital Articles Level 4 frame, 5,000 concepts, prepared on the desktop (Luna plus G1 on the 5090) and fed
  in as they were ready (`make_run --append --watch`).
- **Compute.** Up to 8 rented RTX 3090s (Community, on-demand, $0.22/h) plus the local 5090 on one shared queue. The
  launcher started at 3 pods and was restarted at 8 with older-driver hosts allowed. Faulty 4090 hosts were released
  automatically.
- **Result.** 250/250 batches, 5,000/5,000 concepts: **4,194 mapped, 806 not recalled, 0 failed**. The 5090 did 56
  batches.
- **Time.** About 8.5 h end to end (22:15 to 06:41).
- **Cost.** **About $12.90** of cloud. Each pod spent 92–97% of its lifetime on tasks (`timeline.py`), so boot was
  3–8%.
- **Output.** About 32 GB drained to `D:/already-clean-data/loop_v2/run1/results/`. The bucket peaked under 1 GB, and
  R2 requests reached about 37k Class A and 54k Class B for the month.
- **Teardown.** Zero pods at the end; RunPod's billing endpoint showed $12.96 for the period.

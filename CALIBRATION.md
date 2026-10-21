# Calibration of `modelmapper check`

The values `check` shows as "ours", and the runs they come from. All runs here: Windows 10, CPU only (no GPU),
Python 3.11.9, torch 2.11.0+cpu, transformers 5.20.0, float32 (what `check` picks when there is no GPU), 8 threads,
Qwen3-0.6B. Dates are 19 Oct 2026.

## Same numbers as the research code

`map_concept` in this package was run on the exact question set that the previous code (`claude/cli-easy`,
`modelmapper map-concept Bitcoin`) had used, with the same settings. Every field of the two records was compared:
the 200 candidates and their drops, attribution ranks and activations, per-question drops, the random controls, both
k curves, the 50 heads, the roles and the answer-form scores. **0 differences** (provenance, timings and the new
guards fields aside). L23:2392 came 2nd with a drop of 0.849, recall 9, in both. That run used the same torch build
as the original (2.11.0+cu128, running on the CPU). Re-run with 0.1.1 (the prompt helper, the head_dim fallback and the
post-MLP-norm reading of roles): 0 differences again.

The same comparison from a fresh clone with a normal `pip install .` and the CPU build of torch (2.11.0+cpu), through
the command (`modelmapper map-concept Bitcoin --questions <that question set>`), is not bit-identical: the two torch
builds do CPU arithmetic in slightly different orders. The same: recall (9), the mean baseline (0.26), k_mean (5)
and k_flip (1), the random-control neurons (mean drops 0.000 and 0.007, against -0.005 and -0.002), the set of top 30
neurons, and 193 of the 200 candidates. L23:2392 is 2nd in both (0.849 and 0.856); 4th and 5th swap places.
Per-neuron mean drops differ by a median of 0.017 and at most 0.078, inside the 0.13 band (FINDINGS.md, section 6:
0.02-0.04 is typical between GPU types). Single-question values differ more, by up to about 0.1. The values below are from the CPU build, which is what SETUP.md installs on a machine without
an NVIDIA GPU.

## Offline steps (no API calls; deterministic on one machine)

| Step | Result | Time |
|---|---|---|
| key1 | 3/3: Bitcoin L23:2392 rank 2 (drop 0.160), YouTube L23:2069 rank 1 (0.150), photosynthesis L23:487 rank 4 (0.211) | 101 s |
| map | recall 12; L23:2392 at position 2, drop 0.838; top 5 L27:81, L23:2392, L9:1456, L4:2581, L27:39 | 386 s |
| steer | L23:2392 held at 692.0 (8 × its peak); both replies changed; the repeat was identical | 6 s |

The key's own drops (0.155, 0.146, 0.208) are from the GPU in bfloat16; on the CPU in float32 they come out
0.003 to 0.005 higher, with the same ranks. The previous code's check gave the same CPU values (0.160 / 0.150 / 0.211).

## Live step (`--apis`; fresh labeller questions each run)

Two runs, each with fresh labeller questions for the key's three concepts. Position among the 200 candidates:

| Run | Bitcoin L23:2392 | YouTube L23:2069 | photosynthesis L23:487 | Result | Time | Labeller cost |
|---|---|---|---|---|---|---|
| live 1 | 2 | 2 | 4 | PASS, 3/3 | 1,429 s | $0.005 |
| live 2 | 2 | 3 | 4 | PASS, 3/3 | 1,297 s | $0.005 |

The step passes with 2 of 3 markers in their concept's top 20, so it has room for fresh questions to vary.

Earlier, five independent `map-concept Bitcoin` runs with fresh questions put L23:2392 at position 2, 2, 3, 2 and 2.

## Second machine: a laptop GPU in bfloat16

James's work laptop, 19 Oct 2026: Windows 11 Enterprise (build 26100), Python 3.10.11, NVIDIA RTX 1000 Ada (6 GB),
CUDA, bfloat16, labeller through an Azure OpenAI endpoint, from a fresh clone of this package. `check --apis`: PASS
in 309 s. It is the first GPU bfloat16 run of this check and the first on a second machine.

| Step | Laptop (GPU, bfloat16) | Ours (CPU, float32) |
|---|---|---|
| key1 | 3/3 at ranks 2 / 1 / 4; drops 0.163 / 0.144 / 0.216 | 3/3 at ranks 2 / 1 / 4; drops 0.160 / 0.150 / 0.211 |
| map | L23:2392 2nd, drop 0.840; top 5 L27:81, L23:2392, L9:1456, L4:2581, L27:39 | 2nd, drop 0.838; the same top 5 |
| steer | 692.0; both replies changed; repeat identical | the same |
| live | 3/3 (positions 2, 3, 2) | 3/3 in three runs |
| judge | SKIPPED (no Jev key) | PASS |

The key's own drops (0.155 / 0.146 / 0.208) came from an RTX 5090 in bfloat16; the laptop's key1 drops were the same
on its earlier runs of the previous code. Its drops differ from ours by up to 0.006 and from the key's by up to 0.008; no
rank differs.

## Labelling: `check --labels` (21 Oct 2026)

The values the labelling steps compare with, and how their bands were set. Rules: `modelmapper/labelcheck.py`. Our
runs: the same Windows 10 desktop, CPU only (16 logical cores; torch 2.11.0+cpu), float32, Python 3.11.9; labeller
gpt-6-luna, judge Jev. Shipped references (`modelmapper/data/check/`, 126 KB in all):

| File | What | Size |
|---|---|---|
| `sweep1_Qwen3-0.6B.npz` | our profiles: this package's sweep on that CPU in float32 | 25 KB |
| `sweep1_Qwen3-8B.npz` | the research sweep's stored profiles (`vocab_sweep_v2_c0/c1`, RTX 5090, bfloat16) for the same tokens | 22 KB |
| `exam1_Qwen3-0.6B.json.gz` | our gate exam of the falsifier's 20 real neurons (below): descriptions, test lists, controls, bins | 23 KB |
| `exam1_Qwen3-8B.json.gz` | 20 neurons of the research v3 gate exam (`default_rng(2026)` over its 950 records) | 55 KB |

**sweep1.** The slice is the 512 tokens `sweep.allowed_tokens` draws with its seed (2026); Qwen3-0.6B and Qwen3-8B share
the tokenizer, so it is the same slice for both. Contexts 0 and 1, slot reading, gate and up profiles of 6 markers:
Qwen3-0.6B L4:2581, L9:1456, L23:487, L23:2069, L23:2392, L27:81 (the answer key's markers and three neurons of its
Bitcoin map); Qwen3-8B the v3 check rows L0:1951, L7:11335, L25:7850, L31:10454, plus L16:11480 and L35:8545 (the
sample's first neurons in layers 16 and 35). The band, measured on Qwen3-0.6B on our CPU:

| Comparison | Max abs difference | Largest, in profile sd | Min correlation |
|---|---|---|---|
| float32, batch 64 vs batch 1024 | 0.000 | 0.00 | 1.0000 |
| bfloat16 vs float32 | 0.694 | 0.25 | 0.9992 |

PASS needs every profile within 0.5 sd and correlated at 0.995 or more (twice the bfloat16 difference); WARN down to a
correlation of 0.98; below that, or a different token slice, FAIL. `check --labels` on this machine (which picked
bfloat16 on the CPU, as a GPU is present) gave the bfloat16 row: PASS, the slice in about 10 s plus the model load. GPU against CPU was not measured (the GPU is kept for research); on Qwen3-8B the only
measurement is inside the research sweep: its four check rows, the same neurons read through the pseudo path, agree
with the sample profiles to 0.07 sd (correlation 0.999995). The Qwen3-8B step itself was not run here (it needs the
8B weights on a GPU).

**exam1.** Re-judging the shipped inputs with `exam.rejudge` (the exam's own judge prompts, merged calls, scoring and
adaptive draws; tested against `run_exam` on a synthetic sweep, where it gives identical results):

| Model | Same final bin | Our bins | Judge cost | Time |
|---|---|---|---|---|
| Qwen3-0.6B | 20/20 | 2 intelligible, 3 list-only, 15 no-generalisation | $0.020 | 8 s |
| Qwen3-8B | 19/20 | 11 intelligible, 3 list-only, 5 no-generalisation, 1 decoys-too-easy | $0.032 | 11 s |

The one Qwen3-8B difference: L11:289 came out `intelligible`, ours `decoys-too-easy (intelligible)` (its control
AUC sits at the 0.65 line). Jev's ratings vary a little between runs (AUCs within about 0.03 here), so the step
passes at 18 of 20 and otherwise WARNs.

**falsifier** on Qwen3-0.6B (CPU, float32): 20 random real neurons (`default_rng(2026)`) and 20 fake ones (pseudo.py's up
set, seeds 2026 / 2027) labelled by `label.label_run` on a 3,000-token sweep:

| | Gate | + side | − side |
|---|---|---|---|
| Fakes verified (exam) | 0/20 | 1/20 | 0/20 |
| Fakes verified + generative check | 0/20 | 1/20 | 0/20 |
| Real neurons verified (reported only) | 10% | 10% | 0% |
| Control AUC median, real neurons | 0.478 | 0.506 | 0.485 |

PASS (at most 1/20 fakes per part, control median 0.488 in 0.40-0.60). Spend: labeller $0.24, judge $0.26. Time:
about 12 minutes: the sweep 517 s (8 torch threads, nothing else running), the labelling 175 s (API calls, then scoring the generative check's words on the model), the fake-neuron set and model load under 20 s. The real neurons' rates are far below Qwen3-8B's (44% / 30% / 31%): a small model and a 3,000-token
sweep; no bar is set for them. For Qwen3-8B the falsifier was not run by this check; the v3 run's own falsifier
(100 fakes per part, the full sweep) gave 1% / 0% / 1% verified + generative and a control median of 0.50.

**Second machine** (laptop trial, 21 Oct 2026): a fresh clone on James's work laptop (Windows 11 Enterprise, Python
3.10.11, RTX 1000 Ada 6 GB, CUDA 12.8, behind a corporate proxy that blocks huggingface.co, so the model came from a
local folder via `MODEL_MAPPER_MODELS`). `check --labels --model Qwen/Qwen3-0.6B`: PASS in 245 s. sweep1 min
correlation 0.9990, largest difference 0.17 sd (GPU, bfloat16, against our CPU float32); exam1 20/20; falsifier 0/20 on
every part, control median 0.486. Spend $0.55. A 30-neuron `label` run with 20 fakes then took 44 minutes (the two
sweeps 1,427 s and 870 s; API about 6 minutes) and $1.32; its fakes passed verified + generative 2/20 (gate), 0/20
(+ side) and 3/20 (− side). That is why `label --with-fakes` now gives the same PASS / WARN / FAIL verdict as this
check, and why the plan's cost is a range (this run's judge cost was about 4 times v3's per neuron).

## Not calibrated here

- The hardware band. Qwen3-8B's 0.13 is the research side's measurement (METHOD.md, "In, uncertain or out"); Qwen3-0.6B has
  none yet: paired runs of one concept on two machines are not enough to set one.

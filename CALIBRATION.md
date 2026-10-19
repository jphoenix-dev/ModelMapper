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
as the original (2.11.0+cu128, running on the CPU).

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

## Not calibrated here

- The hardware band. Qwen3-8B's 0.13 is the research side's measurement (METHOD.md, "In, uncertain or out"); Qwen3-0.6B has
  none yet: paired runs of one concept on two machines are not enough to set one.

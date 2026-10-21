# Labelling neurons

How `modelmapper` labels what each part of a neuron does, and how every label is checked. It is the method of the v3
labelling run on Qwen3-8B (1,000 neurons, October 2026). This package reproduces that run's outputs (see "Proof"
below). Explainer: *Reading neurons directly, fast*.

## The neuron, in three parts

Each MLP neuron in a SiLU-gated model computes, for every token, activation = SiLU(gate · x) × (up · x), then adds
activation × a fixed `down` vector to the residual stream. We label three things separately:

- **Gate**: what opens the neuron. The gate reading decides how much gets through. It's soft: below zero it lets a
  little through, with the sign flipped.
- **Up, two sides**: once open, the sign of the up reading picks a side. The **+ side** writes +down; the **− side**
  writes −down. The two sides can respond to unrelated things, so each gets its own label.
- **Launcher (down)**: fixed by the weights. Its job type per side, the words it pushes the model towards, and what
  breaks when the neuron is knocked out.

This picture applies to dense models with SiLU-gated MLP neurons, like Qwen3-8B.

## The pipeline

| Step | Command | What it does | Needs |
|---|---|---|---|
| 1 | `modelmapper sweep` | Every token in the vocabulary (~151k for Qwen3) goes into 12 template sentences, and every neuron's gate and up readings are recorded in one pass. Per neuron, it keeps the top 100 tokens by gate, by + side activation and by − side activation. Then `hub_tokens` finds glitch tokens: favourites of very many unrelated neurons. | GPU for 8B (about 30 GB of output) |
| 2 | `modelmapper pseudo` | Fake neurons that go through every later step like real ones: random mixtures of real gate rows, or random directions, with random up rows. This is the falsifier. Fixed once written. | CPU |
| 3 | `modelmapper exam` | **Describe:** the labeller (Luna) sees half of the favourites and writes a short description; "no pattern" is allowed. **Exam:** a calibrated judge (Jev) sees only the description and must pick the held-out favourites out from surface-matched decoys. The same test with 5 other neurons' descriptions is the control. A **list exam** scores the favourites themselves. | Labeller + judge APIs |
| 4 | `modelmapper gencheck` | For exam-verified labels: Luna writes 30 new words from the label alone, and 30 from a description of only the tokens' surface (spelling, script, length). The neuron must fire more on the first set. | Labeller API, then the model |
| 5 | `modelmapper jobtype` | From the weights, per side: the cosine between what the side reads (unit(gate) ± unit(up), through the norm the MLP reads) and what it writes (±down). The scorecard classifies a side as an amplifier at ≥ 0.2, an eraser at ≤ −0.2, otherwise a redirector (the v3 rule). The file also keeps a per-layer null threshold, from pairing each neuron with a random other neuron's write. | CPU |
| 6 | scorecard (the last step of `label`; `modelmapper.scorecard` in Python) | Combines everything into one record per neuron, with a triage and a summary. | CPU |
| — | `modelmapper label` | Runs steps 3, 4 and 6 for a list of neurons. It can resume after a stop, overlaps describing the next chunk with judging the current one, and stops at spend caps. | as above |

Knockouts (which concepts break when a neuron is switched off) come from `map-concept` / `map-batch` (METHOD.md). The
scorecard can read them from a ledger to cross the triage with knockout effects.

## Verdicts

- **Label exam passes**: AUC ≥ 0.75 *and* AUC − control ≥ 0.2, over repeated draws of decoys (a second draw when the
  first is close). The bin is `intelligible` when both the label exam and the list exam pass; this is what "verified"
  means.
- **Generative check passes**: AUC ≥ 0.75 for own-label words against surface-label words. It runs only on verified
  labels.
- **Verified at the edge** (bin `decoys-too-easy (intelligible)`): both exams passed, but other neurons' labels or
  token lists also scored above 0.65 on the same test, so the test was at the edge of being too easy to certify the
  label. These count as verified, with lower confidence (`at_edge` in the records; `lookup` and `label` mark them).
  This was decided after the runs: the rule set before them left these labels unverified, but on real text (the corpus
  exam) they pass nearly as often as other verified labels (67% against 81% on same-word decoys; 30% for labels that
  failed the exam). About 2% of gates and 1% of each side. As in the runs, the generative check runs only on labels
  in the bin `intelligible`, so these have none yet.
- **Triage**: *clear* (gate verified), *one-sided* (no gate label, but a side verified), *pattern-not-named* (no
  verified label, but the gate's list exam passes), *unexplained*.

## Results on Qwen3-8B

Two runs with this method (October 2026):

- **v3: 950 random neurons** (plus 50 shared with an earlier run). This is the fair estimate for a whole model.
- **v3x: 10,000 knockout-linked neurons**, chosen because switching them off changes the model's answers to concept
  questions. They're expected to be more readable than average, and they are.

| Part | v3 random: label exam | … and generative check | v3x knockout-linked: label exam | … and generative check |
|---|---|---|---|---|
| Gate | 45.6% | 33.4% | 69.9% | 55.5% |
| + side | 30.6% | 12.4% | 45.8% | 19.9% |
| − side | 31.3% | 13.9% | 47.2% | 20.6% |

The label-exam rates include labels verified at the edge (above): 1.6 / 0.7 / 0.7% in v3 and 2.0 / 1.0 / 1.1% in
v3x. **Fake neurons** (100 per part in each run) were verified 1 / 1 / 1% in v3 and 3 / 4 / 4% in v3x. For both checks
together, the figures are 1 / 0 / 1% and 1 / 2 / 0%. **Controls** (other neurons' descriptions on the same test)
score at chance, with a median AUC of about 0.50. Middle layers are the hardest to label in both runs.

**Cost:** v3 took about 37 minutes and $7.29 of API calls for 1,000 neurons. v3x took 2 h 04 min and about $78 for
10,000, every part: about $0.009 per neuron.

## How `label` matches the research runs

`label` runs the production flow of the v3x runner:

- **Pools of about 1,000.** Each exam's controls (other neurons' labels) and surface-matched decoys (other neurons'
  favourites) come from its pool, so the pool size matters. As in v3 and v3x, a large run is split into interleaved
  pools of about 1,000 neurons spanning the layers. A small run is filled up to 1,000 with random neurons drawn from
  all layers (seeded), which are described once but not judged. The sweep always covers every layer.
- **Describe once, judge in two slices, merge**, as in v3x (`--batches 2`): the same verdicts as one judging pass, up
  to judge noise.
- **Hedged describe requests** after 45 s (v3x `--hedge-s 45`): a slow describe request gets a duplicate and the first
  answer is used. Faster, with a small known selection effect.
- **Coverage**: a neuron that already has a label for a part isn't judged again for it. Where the research runner read
  its ledger, `label` uses the published labels and your earlier runs in `--out`; `--relabel` judges everything again.
- **Spend caps**: the per-service caps of the research runner, plus `--budget` (by default twice the plan's upper
  estimate).

## Experimental: checking labels on real text

`modelmapper realtext` packages the research code for two further checks. Both ran on Qwen3-8B on 21 October 2026. They
are **experimental** here: the code is the research run's, with its paths and model made arguments, and it is tested
offline on Qwen3-0.6B (`tests/test_realtext.py`). It has not been re-run or calibrated in this package.

**Readings** (`realtext readings`, GPU for big models). One forward pass over a corpus reads every neuron's gate and up
values at every position. Per neuron it keeps the top 100 positions by gate, by + side activation and by − side
activation. The research run used three corpora, about 9.3 million tokens:
- web pages: a FineWeb shard, 5.0 million tokens, 44 minutes on an RTX 5090;
- chat-format recall questions;
- Wikipedia intros.

Give a `.parquet` (FineWeb's `text` column), `.jsonl` or `.txt` file of documents. The research run's questions and
intros were its own files and don't ship.

**The corpus exam** (`realtext corpus-exam`). Does a gate label pick out where the neuron fires in ordinary text it never
saw? Per neuron, the judge sees only the label and rates short passages that end at a marked token:
- **positives:** the neuron's 10 strongest moments in the corpus (at most 2 per document);
- **A, same token** (the main test): the same token in another passage where the neuron's gate stays shut, which holds
  the word fixed;
- **B, same topic:** strong moments of the 5 neurons with the nearest labels (by Qwen3-Embedding-0.6B);
- **C, random:** strong moments of 5 random other labelled neurons.

The same items are rated with 5 other neurons' labels as controls. Each contrast passes at AUC ≥ 0.75 and ≥ 0.20 above
the controls. A neuron with fewer than 10 quiet same-token passages fires on its token almost everywhere: it is
reported as *token-locked*, not failed. A judge-free keyword baseline (label words found in the passage) runs on A.

The research run covered 12,244 labelled neurons and 200 fakes, and all eight pre-registered predictions were met:

| | A same token | C random | B same topic |
|---|---|---|---|
| verified gate labels | 81% | 86% | 37% |
| all gate labels | 64% | 71% | 31% |
| fake neurons | 2.2% | 4.5% | 2.5% |

Other labels scored 0.49–0.51 on every contrast. Cost: $14.81 of judge calls ($0.0012 per neuron), plus the readings
and a 22-minute gather pass on an RTX 5090.

**Graded simulation** (`realtext simulate`). From the label alone, the judge rates how strongly the neuron fires at 60
web positions: its 12 strongest moments, plus 48 shared positions sampled across its own activation range (12 each
below the median, median to 90th percentile, 90th to 99th, above the 99th). The score is the rank correlation with the
real value, for the gate, + side or − side. A first pass stores every neuron's values at 4,000 shared positions, so no
neuron needs a pass of its own.

The research run scored every labelled Qwen3-8B neuron's gate and side labels (12,244 gates, 12,018 per side) and 200
fakes per part, on web text. All three pre-registered bars were met. Median rank correlation:

| | Gate | + side | − side |
|---|---|---|---|
| verified labels | 0.65 | 0.55 | 0.55 |
| without the 12 strongest moments | 0.42 | 0.25 | 0.24 |
| labels that weren't verified | 0.41 | 0.33 | 0.34 |
| other neurons' labels (controls) | 0.09 | 0.09 | 0.10 |
| fake neurons | 0.17 | 0.13 | 0.17 |

A label describes a neuron's peaks best; much of its weaker activity isn't captured. Cost: $36 of judge calls for about
36,600 labels, about $0.001 per label, after a 73-second GPU pass that stored every neuron at 4,000 web positions. One
neuron's score from 60 positions is noisy, about ±0.1.

```bash
modelmapper realtext readings --corpus web --source fineweb_000_00000.parquet --max-tokens 5000000
modelmapper realtext corpus-exam --corpus web --tag t1 --labels labels/Qwen3-8B --smoke 30 --yes
modelmapper realtext simulate --corpus web --tag s1 --labels labels/Qwen3-8B --n 50 --yes
```

`--published` examines the published Qwen3-8B labels. `--pseudo-set` and `--pseudo-exams` add a label run's fake
neurons. Every step resumes, and the judge steps need `--yes`.

## Proof that this code is the research code

`tests/test_v3_repro.py` and `tests/test_v3_scorecard.py` (marked `repro`; set `MODEL_MAPPER_V3_REF` to the research
run's folder) check this package against the research run's own files:

- pseudo-neuron sets: bit for bit;
- job types: bit for bit;
- scorecard: all 1,000 records and the triage counts.

The sweep, exam and generative-check checks are described in CALIBRATION.md.

On your own machine and model, `modelmapper check --labels [--model M]` runs three of them against shipped values: a
sweep slice (`sweep1`), 20 exams re-judged (`exam1`), and real and fake neurons labelled for real (`falsifier`). Only
labeller gpt-6-luna with the Jev judge, on Qwen3-8B and Qwen3-0.6B, is tested; other models get the falsifier only and
are reported as uncalibrated (CALIBRATION.md, "Labelling").

## Prior work this builds on

Almost every ingredient here has been published before. What is new is putting them together cheaply enough to label
neurons at scale.

- Describe-then-score with an LLM: Bills et al. 2023 (*Language models can explain neurons in language models*);
  Paulo et al. 2024 ([arXiv 2410.13928](https://arxiv.org/abs/2410.13928)); Transluce neuron descriptions.
- Scoring the examples themselves (our list exam): Paulo & Belrose 2025 ([arXiv 2507.08473](https://arxiv.org/abs/2507.08473)).
- Random units as a falsifier: Bau et al. 2017, Network Dissection ([arXiv 1704.05796](https://arxiv.org/abs/1704.05796));
  SAEBench 2025 ([arXiv 2503.09532](https://arxiv.org/abs/2503.09532)).
- Every input through a module, then summarise the strongest (our sweep): Singh et al. 2023, SASC
  ([arXiv 2305.09863](https://arxiv.org/abs/2305.09863)).
- The gated MLP and its sign split: Shazeer 2020 ([arXiv 2002.05202](https://arxiv.org/abs/2002.05202)); Shabalin &
  Piotrowski 2026 ([arXiv 2610.24521](https://arxiv.org/abs/2610.24521)).
- Amplifier / eraser neurons from the weights: Gerstner & Schütze 2025 ([arXiv 2505.17936](https://arxiv.org/abs/2505.17936)).
- A neuron's output as the words it pushes: Geva et al. 2021 ([arXiv 2012.14913](https://arxiv.org/abs/2012.14913)) and
  2022 ([arXiv 2203.14680](https://arxiv.org/abs/2203.14680)).

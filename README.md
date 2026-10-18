# Model Mapper

**Model Mapper** maps what the individual MLP neurons of an open-weights language model do. It finds which neurons
a concept depends on, tests how few are enough, labels neurons with one LLM and grades the labels with another, and
steers the model with them. Every stage has controls. The code runs on one consumer GPU, and the model, the
labeller and the judge are all configurable ([Use it on another model](#use-it-on-another-model-or-endpoint)).

> **See it in action:** [**Eiffel Tower Qwen**](https://eiffel-tower-qwen.vercel.app/) is an interactive explainer for
> lay readers. Turn a handful of the model's Eiffel Tower neurons up and watch the tower take over its answers,
> switch them off and watch it forget, then try any of the 138 concepts we tested. Source: [explainer/](explainer/).

> **Installing it?** Follow [SETUP.md](SETUP.md). It is a step-by-step install and self-check that a person or a
> coding agent can follow ("clone this repo and follow SETUP.md"). It ends with a short report of what passed.

Its first use, and everything reported below, is an independent, roughly 24-hour test of Nick Cammarata's essay
[*Neural Networks Are Already Clean*](https://already-clean.goodfire.com/) (Goodfire, 2026). It was run on one
consumer GPU, with an AI agent doing the experimental work under human direction.

The essay claims that most individual MLP neurons in a large language model do one recognisable job, and that the
model builds concepts out of such neurons. We tested this on **Qwen3-8B** (36 layers × 12,288 SwiGLU neurons =
442,368 neurons), with a **Qwen3-0.6B** comparison.

> **Scope.** Only two small models from one family were tested: Qwen3-8B, plus Qwen3-0.6B. The essay's audits are of
> Qwen3-32B. No other model families or sizes were tested. Most results are exploratory. The pre-registered
> experiments are CR, CEN, H-DEPTH and the confirmatory CR-3 ([PREREG.md](PREREG.md)).

> **Since the first release (17-18 Oct 2026; exploratory, work in progress).** The project moved from testing the
> essay to building a reusable map of what Qwen3-8B's neurons and heads do. None of this changes the headline results
> below; each item says how strongly it is supported.
> - **Mapping loop v2, run 1:** 5,000 Wikipedia Vital Articles concepts mapped to neurons and heads, with controls
>   ([LOOP_V2.md](LOOP_V2.md), [PREREG_MAP.md](PREREG_MAP.md)).
> - **Layers and jobs:** what can and cannot be said about the job of each layer band, graded by evidence
>   ([LAYERS_AND_JOBS.md](LAYERS_AND_JOBS.md)).
> - **The ledger:** every measurement about every neuron / head / concept in one queryable store, with a format for
>   third-party contributions ([LEDGER.md](LEDGER.md), [src/ledger/](src/ledger/)).
> - **Loop v3 (in progress):** an enrichment pass over every mapped concept, adding knowledge neurons acting at the
>   concept's name, answer formats, missed candidates and stronger nulls ([LOOP_V3.md](LOOP_V3.md)).
> - **Details and caveats:** the dated entries in [RESULTS.md](RESULTS.md); the current picture in
>   [RESEARCH_STATE.md](RESEARCH_STATE.md).
>
> Several runs were still going at the time of this snapshot (links between key neurons, a larger J-lens estimate,
> attribution graphs, loop v3). Their results will follow. Research scripts read their data from `AC_DATA_DIR` (default
> `D:/already-clean-data`), `LEDGER_DIR`, `AC_REPO_RESULTS` and `MODEL_MAPPER_MODELS`; the bulk data (tens of GB) is not
> in this repository.

## Headline results (Qwen3-8B)

The full account is in [REPORT.md](REPORT.md) (the write-up) and [RESULTS.md](RESULTS.md) (chronological log).

**1. Recall depends on a handful of neurons.**
- Switching off at most 5 neurons lowers the answer's log-odds by ≥ 3 (odds ~20× lower) in **157/158**
  question-format concept runs. In 142/158 runs, 1–2 neurons are enough, and all runs need at most 10.
  The median prompt gives the same result.
- Completion format ("The tallest mountain is Mount"): 8/11 runs, all within 20.
- Actually flipping the answer takes ≤ 5 neurons in 137/158 runs. Near-certain facts (the Moon, Tokyo, DNA,
  Bitcoin) need 20–50, because they start with a much bigger confidence margin.
- In plain probability, the top five neurons cut P(answer) by 61–82%. Five random same-layer neurons change nothing.
- **Caveats.** This shows the neurons are *necessary* for recall, not that the knowledge is stored in them.
  The network compensates when neurons are removed (known self-repair). A concept's exact top-10 neurons change with
  the prompts (4.8/10 overlap), though the shared *parts* are stable.

**2. Concepts share readable parts.**
- Top neurons have readable labels (gold, bridge, landmark, mountain/山, coffee/咖啡, democracy/民主, Paris).
- Related pairs (coffee/tea, the Moon / the Moon landing, red/blue, …) share **2.6** specific top-50 neurons on
  average, against **0.13** for random pairs (~20×).
- Lexical control: 0 of the 26 shared neurons is a word-form detector. Some shared neurons fire on punctuation, so we
  claim "non-lexical", not "all meaning".

**3. About half of neurons verifiably do one job.**
- In a calibrated census of 1,000 random neurons (954 live), **~50% of positive poles and ~47% of negative poles**
  have a *verified* single job. The bar is set so that it passes only 8% / 4% of known two-job dossiers.
- This is closer to a floor than a point estimate.
- The labeller alone says ~81%, but it also calls half of the known two-job dossiers "one job".
- The essay reports ~90% for the 4× larger Qwen3-32B, from in-depth audits.
- Robustness checks:
  - Relabelling from scratch gives the same rate.
  - A second grader agrees (correlation 0.81).
  - Longer grading context changes nothing.
- 4.5% of all neurons are near-silent (mostly layers 1–3).

**4. Structure by depth (pre-registered H-DEPTH, replicated).**
- Three stages: recognise the question (layers 6–12) → concept parts (22–32) → shared "say the answer" machinery
  (33–35).
- An independent repeat agreed on 91% of 550 classifications.

**5. Meaning fades gradually below each neuron's top firings.**
- At 100/50/25/10% of peak, the label fits 66% → 51% → 41% → 33% of firings (random tokens: 20%).

**6. Steering works, modestly.**
- Nudging 2–6 of these neurons steers free-form chat towards the concept. Random neurons and unsteered replies score
  ≈ 0 throughout.
- Choosing the neuron set per concept: held-out "on-topic and fluent" score **0.16 vs 0.09** for a single fixed rule
  (59 concepts, Jev judge on both sides). Better in 31 concepts, worse in 7.
- Caveats:
  - Scores roughly halve from the prompts a setting was chosen on to fresh prompts.
  - A strict "main topic" judge roughly halves them again. Typical steering makes the concept *intrude* rather than
    take over.
  - People are hard to steer fluently. On person-friendly prompts, 4 of 5 hard cases reach 0.25–0.55.

**7. Coverage.**
- **139 distinct concepts recalled**, including three waves of 40 brand-new concepts.
- Waves 3–4 ran with no judge API at all and reproduced the findings.
- Every real concept we first marked "not recalled" turned out to be a fault in our measurement, with one exception
  ("the robot", a limit of the prompt design).

**8. A pre-registered confirmatory replication (CR-3) passes 6 of 7 predictions.**
- Setup: 40 never-used concepts plus 4 made-up controls, the final pipeline, and predictions and scoring frozen
  before any data.
- Passed:
  - 40/40 recalled, 0/4 made-up;
  - ≤ 5 neurons for 98%;
  - answers flip with ≤ 5 neurons for 85%;
  - the most resistant facts are the most confident (p = 0.005);
  - per-concept steering is 2.44× the fixed rule, random 0.006, and the same direction under Jev.
- **Failed:** P5 (people steerable, but less than other concepts). The people claim is revised (REPORT §4.4, §4.6).

**Model size (Qwen3-0.6B vs 8B, same 60 concepts and prompts; exploratory).**
- **Recall:** the 0.6B model recalls only 7/60 in question form (8B: all).
- **Concentration:** where it does recall, a handful of neurons again carry it (6/7 with ≤ 5).
- **Flips:** its answers flip with just 1–3 neurons.
- **Steering:** weak, 0.057 against 0.226.
- **Scope:** two sizes from one family, so no claim about trends with scale (REPORT §4.5).

**Where this sits in the literature.**
- Most individual findings have precedents: knowledge neurons, entity cells (including in Qwen3-8B), expert-neuron
  steering, staged recall.
- What looks new is narrower: the scale and consistency of the "few neurons" effect, a calibrated census, raw-neuron
  steering measured like a benchmark, and part sharing that survives a lexical control.
- See [LITERATURE.md](LITERATURE.md) (80 citations, each independently verified).

**Pitfalls.**
- Many of the night's findings were measurement bugs, caught by controls (random neurons, made-up concepts, spliced
  dossiers, re-runs with independently written prompts, held-out prompts).
- Each is logged with its fix in [ISSUES.md](ISSUES.md) and [CAPS.md](CAPS.md).

## Documents in this repo

| File | What it is |
|---|---|
| [REPORT.md](REPORT.md) | Start here: the technical write-up |
| [RESULTS.md](RESULTS.md) | Chronological results log, every experiment with its numbers |
| [PREREG.md](PREREG.md) | Pre-registrations (CR, CEN, H-DEPTH, and the confirmatory CR-3, frozen before data and scored: 6/7 pass) and change log |
| [ISSUES.md](ISSUES.md) | Every setup flaw found, and its fix |
| [CAPS.md](CAPS.md) | Every cap or truncation in the code, and why |
| [LITERATURE.md](LITERATURE.md) | Related work, with what is known vs new |
| [GLOSSARY.md](GLOSSARY.md) | Plain-language glossary of the ML and measurement terms |
| [SETUP.md](SETUP.md) | Install and self-check, step by step (for people and coding agents) |
| [ARCHITECTURE.md](ARCHITECTURE.md) | How the pipeline and code are organised |
| [LABELLING.md](LABELLING.md) | The blind-labelling protocol |

## Hardware and cost

- One NVIDIA RTX 5090 (32 GB) in a Windows 10 desktop. Qwen3-8B runs in bf16. The code also runs on Linux.
- Total wall-clock: about one day.
- The atlas pass (10M tokens through Qwen3-8B) takes ~44 min. A concept-pipeline run takes ~45 s per concept.
- API spend: roughly $6–6.5 of Jev and a few dollars of Luna.

**External models used as tools (all swappable; see below):**
- The **labeller**, "Luna" (OpenAI, `gpt-6-luna`), writes neuron labels and recall prompts.
- The **judge**, "Jev" (TypeSafe's "System One"), returns calibrated yes/no, choice and rubric probabilities. It
  grades labels and steered replies.
- A free local fluency check, validated against Jev, replaces the judge in the judge-free waves.

## Reproduce

### Setup
The short version is below; [SETUP.md](SETUP.md) has the full walk-through, with checks and troubleshooting.
```bash
python -m venv .venv
source .venv/bin/activate          # Windows (Git Bash): source .venv/Scripts/activate
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128   # pick the wheel for your CUDA
pip install -r requirements.txt
cp .env.example .env               # then fill in what you need
bash scripts/download_models.sh    # Qwen3-0.6B and Qwen3-8B into $MODEL_MAPPER_MODELS (default ./models)
bash scripts/download_corpus.sh    # one FineWeb shard (2.1 GB) into $MODEL_MAPPER_CORPUS (default ./data)
```
We used Python 3.11.9, torch 2.11.0+cu128 and transformers 5.20.0 ([requirements.txt](requirements.txt)).
Without an NVIDIA GPU, install the CPU wheel (`--index-url https://download.pytorch.org/whl/cpu`, or the default
index on a Mac); the code picks CUDA, Apple MPS or CPU automatically.

### Check your install (about 5–20 minutes)
```bash
python scripts/smoke_e2e.py                                # offline, Qwen3-0.6B, no API keys needed
python scripts/smoke_e2e.py --model google/gemma-3-1b-it   # another model family
python scripts/smoke_e2e.py --apis                         # also one round of labeller + judge calls
```
The script runs every stage at toy scale in a throwaway copy of the repo, using this repo's own text as the corpus:
1. neuron ranges;
2. the atlas and blind dossiers;
3. recall, attribution, knockout and log-odds;
4. a steering grid.

It checks that each stage writes its outputs, not that the numbers mean anything. Nothing in your `data/` or
`results/` is touched. On CPU, setting `MODEL_MAPPER_DTYPE=float32` is several times faster.

**Tested so far:**

| Setup | What ran |
|---|---|
| Windows 10, RTX 5090, Qwen3-8B and Qwen3-0.6B | Every reported result |
| Windows 10, CPU only, fresh clone and fresh virtual environment, Qwen3-0.6B | Smoke test passed: all stages, ~12 min |
| A second machine: Windows 11, Python 3.10, RTX 1000 Ada (6 GB), behind a corporate proxy; set up by a coding agent following SETUP.md | Smoke test passed: all stages, for Qwen3-0.6B (~4 min) and SmolLM2-1.7B-Instruct, a Llama-architecture model (~4 min). Qwen3-0.6B answer key with Luna (via an Azure OpenAI endpoint): tier 1 3/3, ranks identical to ours, drops within ±0.01, control 0/3; tier 2 2/3 and 3/3; tier 3 2/3, keyword control 0/6 |

| The same machine with local endpoints only: Gemma 4 E2B (GGUF) on llama-server as labeller and judge (`JUDGE_BACKEND=llm`) | Smoke test with APIs: 8/8. Qwen3-0.6B answer key, tiers 2 and 3: 1/3 each. Gemma wrote fewer, harder questions (two concepts not recalled), and vaguer labels that missed the keywords. That's a labeller limit; with Luna the same machine passes |

Not yet tested: Linux or macOS, and running Gemma as the model under study.

**Windows:** avoid very deep folders for the clone and its `.venv`. Paths inside the repo are at most 100
characters, but pip's packages can exceed Windows' 260-character limit in deeply nested locations unless long-path
support is on. A path like `C:\Users\you\Documents\GitHub\model-mapper` is fine.

### Check it measures the same thing (answer keys)
The smoke test shows the code runs; an answer key shows it finds what we found.
[answer_keys/](answer_keys/) holds two keys, one for Qwen3-8B and one for Qwen3-0.6B. Each lists a few **marker
neurons**: for a concept the model recalls, one neuron that passed three tests in every independent prompt set we
used:
- it is in the concept's top 5 by attribution;
- knocking it out alone cuts P(answer) by at least 0.1;
- its blind label (written from its top firings on web text) names the concept. For example, Qwen3-8B's democracy
  neuron L31 N11943 is labelled "democracy-related political vocabulary"; it ranks 1st, and knocking it out cuts
  P(answer) by 0.79.

```bash
python scripts/verify_answer_key.py answer_keys/Qwen3-0.6B.json --tier 1   # the key's exact inputs; no API calls
python scripts/verify_answer_key.py answer_keys/Qwen3-0.6B.json --tier 2   # new prompts from your labeller
python scripts/verify_answer_key.py answer_keys/Qwen3-0.6B.json --tier 3   # new blind labels from your own atlas
```
Tier 1 feeds the exact recall questions and answer wording pinned in the key. Re-applying the recall bar instead
let a second machine's rounding change which borderline questions counted, and that moved one marker's rank. Each key
records its own calibration (`"calibration"`): what these checks scored on our machine. That includes
negative controls, which look for each marker in a different concept and should find almost nothing.

| Check (our machine) | Qwen3-8B (10 markers) | Qwen3-0.6B (3 markers) |
|---|---|---|
| Tier 1: the key's prompts | 10/10, same ranks | 3/3, same ranks |
| Control: marker looked for in the wrong concept | 0/10 | 0/3 |
| Tier 2: two fresh prompt sets | 8/10, 8/10 | 2/3, 3/3 |
| Tier 3: fresh labels from a FineWeb shard never used before | 7/10 (keyword control: 2/90) | 3/3 (keyword control: 0/6) |

**Pass marks:** tier 1 needs 90% of markers; tiers 2 and 3 need 60%.

**What the keys show beyond the install check:**
- **Most markers reproduce.** The concept → neuron association mostly belongs to the model, not to our wording or
  text:
  - with new wording, about 80% of markers come back (16/20 for 8B, 5/6 for 0.6B);
  - on new text, the neuron's label names the concept again for 7/10 (8B) and 3/3 (0.6B).

  The weakest 8B marker, Napoleon's, missed in three of the five checks; the key keeps it and records that.
- **Model size matters.** Qwen3-0.6B yields only 3 markers from 30 concepts. Its top recall neurons are mostly generic
  (prepositions of place, list structure, "called / known as") rather than concept-specific.
- **How it was built.** `scripts/make_answer_key.py` rebuilds both keys. The marker rules were fixed before the
  0.6B runs, and the keywords were committed before its second batch ran.

### Configuration (environment variables)

Each variable is read from the environment first, then from a gitignored `.env` file at the repo root. See
[.env.example](.env.example) for the full list.

| Variable | Used for | Default |
|---|---|---|
| `MODEL_MAPPER_MODEL` | The model under study | `Qwen/Qwen3-8B` |
| `MODEL_MAPPER_MODELS` | Folder of local model copies (`<dir>/Qwen3-8B/config.json`) | `./models`, else the Hugging Face hub |
| `MODEL_MAPPER_CORPUS` | Web-text shard for the atlas, scans and random-snippet controls | `./data/fineweb_000_00000.parquet` |
| `MODEL_MAPPER_DEVICE` | `cuda`, `cuda:1`, `mps` or `cpu` | CUDA, else Apple MPS, else CPU |
| `MODEL_MAPPER_DTYPE` | `bfloat16`, `float16` or `float32` (fastest on CPU) | `bfloat16` |
| `MODEL_MAPPER_ATLAS_BATCH`, `MODEL_MAPPER_GEN_BATCH` | Atlas tokens per pass; steering replies per batch | `6144`, `100` (`2048`, `25` on GPUs under 12 GB) |
| `OPENAI_BASE_URL`, `OAI_MODEL`, `OPENAI_API_KEY` | Labeller endpoint (`src/oai.py`) | OpenAI, `gpt-6-luna` |
| `JUDGE_BACKEND` | Judge: `jev` or `llm` (`src/jev.py`) | `jev` |
| `JEV_API_KEY` | Jev judge key (`TYPESAFE_API_KEY` also works) | – |
| `JUDGE_BASE_URL`, `JUDGE_MODEL`, `JUDGE_API_KEY`, `JUDGE_EXTRA_BODY` | Judge endpoint when `JUDGE_BACKEND=llm` | OpenAI |

### Main pipelines
All scripts run from the repo root. Large derived data (the atlas, neuron ranges) goes to the gitignored `data/`
folder; results go to `results/`. Model names below can be any model you have configured.

```bash
# 1. Neuron ranges and the activation atlas (GPU; ~7 min and ~45 min for 8B)
python src/neuron_stats.py Qwen/Qwen3-8B                # -> data/stats_Qwen3-8B.pt
python src/build_atlas.py Qwen/Qwen3-8B 10e6 40         # -> data/atlas_Qwen3-8B.pt (+ _docs.npz)

# 2. Census of random neurons (labeller labels, judge grades), then calibrate the "clean" bar on spliced two-job dossiers
python src/label_batch.py Qwen/Qwen3-8B random:1000 results/census_Qwen3-8B.json
python src/regrade.py results/census_Qwen3-8B.json results/census_regrade_rubric.json
python src/spliced_control.py results/census_Qwen3-8B.json spliced 100       # -> results/spliced_scan.json
python src/label_batch.py Qwen/Qwen3-8B scan results/spliced_labels.json --scan results/spliced_scan.json
python src/regrade.py results/spliced_labels.json results/spliced_regrade.json --scan results/spliced_scan.json
python src/calibrate_clean.py results/census_regrade_rubric.json results/spliced_regrade.json

# 3. Concept pipeline: recall prompts -> attribution -> ablation -> steering grid (question format, reply-only)
RECALL_VARIANT=V3 CONCEPT_OUT_DIR=results/concepts_v3 \
  python src/pipeline_fast.py Qwen/Qwen3-8B --reply-only "coffee" "Eiffel Tower"
RECALL_VARIANT=V3 CONCEPT_OUT_DIR=results/concepts_v3 \
  python src/logodds_check.py my_tag "coffee" "Eiffel Tower"        # log-odds concentration (k_mean, k_flip)

# 4. Judge-free wave: recall + concentration + auto-selected steering, no judge API
CONCEPT_OUT_DIR=results/concepts_w3 python src/wave_free.py prompts/concepts_wave3.json

# 5. Interactive steering "dial" (~20 s to load)
python src/dial.py
```

The exact commands behind each result are in RESULTS.md. Generation is greedy, so steering
re-runs reproduce (mean difference 0.013).

## Use it on another model or endpoint

- **Another model.** Set `MODEL_MAPPER_MODEL` (and pass the same id to the scripts that take one). The hooks need a
  Hugging Face decoder with gated MLPs, laid out as `model.model.layers[i].mlp.{gate_proj, up_proj, down_proj}`
  plus `self_attn.o_proj`, `model.model.norm` and `lm_head`. Qwen2/3, Llama and Mistral fit this layout.
  - Recall and steering prompts go through the tokenizer's chat template, so instruction-tuned chat models are
    expected.
  - Build that model's `neuron_stats` and atlas first; every later stage reads them.
  - Only Qwen3-8B and Qwen3-0.6B have actually been run. Nothing here has been tested on another family.
- **Another labeller.** Point `OPENAI_BASE_URL` / `OAI_MODEL` at any OpenAI-compatible endpoint that supports JSON
  mode, for example vLLM, a llama.cpp server or OpenRouter.
- **Another judge.** `JUDGE_BACKEND=llm` answers the judge's yes/no, choice and rubric questions with any
  OpenAI-compatible endpoint that returns `top_logprobs`. It reads the first answer token's probabilities and
  renormalises them over the valid answers.
  - Use a non-reasoning model, or turn thinking off via `JUDGE_EXTRA_BODY`.
  - This backend is tested offline only. Its probabilities are not calibrated like Jev's, so re-run the controls
    (spliced dossiers, random snippets, shuffled judge) before comparing numbers with the ones here.

## Data

- **Web text.** All web text comes from [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) (Hugging
  Face; ODC-By 1.0). We used one shard, `sample/10BT/000_00000.parquet`, fetched by `scripts/download_corpus.sh`.
  The Golden Gate scans also used the FineWeb documents that mention "Golden Gate".
- **What is not in this repository.** The repository does not redistribute FineWeb text in bulk. These files are
  left out and are regenerated from the corpus by the listed scripts:
  - the per-neuron example scans (`results/scan_*.json`; `scan.py`, `merge_scans.py`);
  - the blind labelling dossiers (`results/**/dossiers_*/`; `make_dossiers.py`, `Atlas.write_dossiers`);
  - the middle-of-range samples (`results/strata_Qwen3-8B.json`; `strata_scan.py`);
  - the spliced-dossier texts (`results/spliced_scan.json`; `spliced_control.py`);
- **What is in it.** Labels, judge scores, ablation curves, steering replies, logs and all numbers are included.
  Short excerpts quoted in the documents and logs are FineWeb text, under its licence.
- **Large derived data.** The atlas and neuron ranges (~270 MB for Qwen3-8B) are not included; rebuild them with step 1.

## Credit

- **James Phoenix** set the questions and the standards: no silent caps, assume the set-up is broken, pre-register,
  test for noise. James made the calls throughout.
- **Claude** (Anthropic's Claude Code, an AI coding agent) designed and ran the experiments, controls and analyses,
  and wrote the code and the logs.
- Thanks to Nick Cammarata and Goodfire for a clear, testable essay.

This is independent work. It is not affiliated with Anthropic, Goodfire, OpenAI, TypeSafe or Alibaba (Qwen).

## Licence

[MIT](LICENSE), for the code, prompts, documents and results. Third-party material keeps its own terms:
- FineWeb excerpts: ODC-By 1.0;
- Qwen model outputs: the Qwen3 licence (Apache 2.0).

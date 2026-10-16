# Already clean? Testing "Neural Networks Are Already Clean" on Qwen3-8B

An independent, roughly 24-hour test of Nick Cammarata's essay
[*Neural Networks Are Already Clean*](https://already-clean.goodfire.com/) (Goodfire, 2026). It was run on one
consumer GPU, with an AI agent doing the experimental work under human direction.

The essay claims that most individual MLP neurons in a large language model do one recognisable job, and that the
model builds concepts out of such neurons. We tested this on **Qwen3-8B** (36 layers × 12,288 SwiGLU neurons =
442,368 neurons), with a **Qwen3-0.6B** comparison.

> **Scope.** Only two small models from one family were tested: Qwen3-8B, plus Qwen3-0.6B. The essay's audits are of
> Qwen3-32B. No other model families or sizes were tested. Most results are exploratory. The pre-registered
> experiments are CR, CEN and H-DEPTH ([PREREG.md](PREREG.md)).

## Headline results (Qwen3-8B)

The full account is in [MORNING.md](MORNING.md) (summary), [RESULTS.md](RESULTS.md) (chronological log) and
[REPORT.md](REPORT.md) (draft write-up).

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

**Model size (Qwen3-0.6B vs 8B):** in progress at the time of writing; see `results/scale_*` and RESULTS.md.

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
| [MORNING.md](MORNING.md) | Start here: summary of all findings, checks, negative results, spend |
| [REPORT.md](REPORT.md) | Technical write-up (draft) |
| [RESULTS.md](RESULTS.md) | Chronological results log, every experiment with its numbers |
| [PREREG.md](PREREG.md) | Pre-registrations (CR, CEN, H-DEPTH, draft CR-3) and change log |
| [ISSUES.md](ISSUES.md) | Every setup flaw found, and its fix |
| [CAPS.md](CAPS.md) | Every cap or truncation in the code, and why |
| [LITERATURE.md](LITERATURE.md) | Related work, with what is known vs new |
| [GLOSSARY.md](GLOSSARY.md) | Plain-language glossary of the ML and measurement terms |
| [ARCHITECTURE.md](ARCHITECTURE.md) | How the pipeline and code are organised |
| [HANDOVER.md](HANDOVER.md), [PLAN.md](PLAN.md), [ESSAY_NOTES.md](ESSAY_NOTES.md), [LABELLING.md](LABELLING.md) | Original brief, plan, essay notes, labelling protocol |

## Hardware and cost

- One NVIDIA RTX 5090 (32 GB) in a Windows 10 desktop. Qwen3-8B runs in bf16.
- Total wall-clock: about one day.
- The atlas pass (10M tokens through Qwen3-8B) takes ~44 min. A concept-pipeline run takes ~45 s per concept.
- API spend: roughly $6–6.5 of Jev and a few dollars of Luna (MORNING.md, "Spend").

**External models used as tools:**
- **Luna** (OpenAI, `gpt-6-luna`) writes neuron labels and recall prompts.
- **Jev** (TypeSafe's "System One") returns calibrated yes/no and rubric probabilities. It grades labels and steered
  replies.
- A free local fluency check, validated against Jev, replaces Jev in the judge-free waves.

## Reproduce

### Setup
```bash
python -m venv .venv
.venv/Scripts/pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128   # Linux: .venv/bin/pip
.venv/Scripts/pip install -r requirements.txt
cp .env.example .env    # then fill in paths and keys
```
We used Python 3.11.9, torch 2.11.0+cu128 and transformers 5.20.0 ([requirements.txt](requirements.txt)).

### Configuration (environment variables)

Each variable is read from the environment first, then from a gitignored `.env` file at the repo root (see
[.env.example](.env.example)). The API keys can also come from the legacy files `OAI.env` and `jevkey.env`.

| Variable | Used for | Default |
|---|---|---|
| `ALREADY_CLEAN_MODELS` | Folder of local model copies (`<dir>/Qwen3-8B/config.json`) | `<repo>/models`, else the Hugging Face hub |
| `ALREADY_CLEAN_CORPUS` | FineWeb parquet shard for the atlas and scans | `<repo>/data/fineweb_000_00000.parquet` |
| `OPENAI_API_KEY` | Luna labeller (`src/oai.py`) | fallback file `OAI.env` |
| `JEV_API_KEY` (or `TYPESAFE_API_KEY`) | Jev judge (`src/jev.py`) | fallback file `jevkey.env` |
| `PYTHON` | Interpreter used by `scripts/*.sh` | `.venv` python, else `python` |

**Corpus.** We used one FineWeb shard, `sample/10BT/000_00000.parquet` from
[HuggingFaceFW/fineweb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) (2.1 GB).

### Main pipelines
All scripts run from the repo root. Large derived data (the atlas, neuron ranges) goes to the gitignored `data/`
folder; results go to `results/`.

```bash
bash scripts/download_models.sh                         # Qwen3-0.6B and Qwen3-8B into $ALREADY_CLEAN_MODELS

# 1. Neuron ranges and the activation atlas (GPU; ~7 min and ~45 min for 8B)
python src/neuron_stats.py Qwen/Qwen3-8B                # -> data/stats_Qwen3-8B.pt
python src/build_atlas.py Qwen/Qwen3-8B 10e6 40         # -> data/atlas_Qwen3-8B.pt (+ _docs.npz)

# 2. Census of random neurons (Luna labels, Jev grades), then calibrate the "clean" bar on spliced two-job dossiers
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

# 4. Judge-free wave: recall + concentration + auto-selected steering, no Jev
CONCEPT_OUT_DIR=results/concepts_w3 python src/wave_free.py prompts/concepts_wave3.json

# 5. Interactive steering "dial" (needs the GPU free; ~20 s to load)
python src/dial.py
```

The exact commands behind each result are in RESULTS.md and `scripts/*.sh`. Generation is greedy, so steering
re-runs reproduce (mean difference 0.013).

## Credit

- **James** set the questions and the standards: no silent caps, assume the set-up is broken, pre-register, test for
  noise. He made the calls throughout.
- **Claude** (Anthropic's Claude Code, an AI coding agent) designed and ran the experiments, controls and analyses,
  and wrote the code and the logs.
- Thanks to Nick Cammarata and Goodfire for a clear, testable essay.

This is independent work. It is not affiliated with Anthropic, Goodfire, OpenAI, TypeSafe or Alibaba (Qwen).

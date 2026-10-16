# Architecture: approach and code

This file explains how the experiments are put together: the pipeline stages, which script does what, where the
data goes, and which scripts are superseded or exploratory. For the results, see [MORNING.md](MORNING.md) and
[RESULTS.md](RESULTS.md). For terms, see [GLOSSARY.md](GLOSSARY.md).

## The approach in one paragraph

**A "neuron"** is one entry of the input to an MLP layer's `down_proj` in Qwen3: `SiLU(gate_proj(x)) * up_proj(x)`.
Each neuron's positive and negative ends ("poles") are judged separately, and the gate sign is kept so the SwiGLU
"double-negative" regime can be separated out.

The project then asks three questions:
1. **Census.** Do random neurons have one job? An LLM labeller reads each neuron's top firings and writes a label.
   A second, independent LLM grades that label on held-out examples and on random text. The pass bar is calibrated
   on known two-job dossiers.
2. **Concepts.** For a concept the model recalls (e.g. "Eiffel Tower"), which neurons is recall *necessary* on, and
   how few are enough? This uses attribution, then knockout, measured on a non-saturating log-odds scale.
3. **Steering.** Do those neurons *steer* free-form chat towards the concept?

Every stage has controls: random neurons, made-up concepts, spliced two-job dossiers, held-out prompts and
independently rewritten prompt sets.

## Components

```
            FineWeb shard ($ALREADY_CLEAN_CORPUS)          Qwen3-8B / 0.6B ($ALREADY_CLEAN_MODELS)
                       │                                              │
                       └──────────────► neurons.py ◄──────────────────┘   load, capture hooks, steer hooks
                                           │
        ┌──────────────────────────────────┼─────────────────────────────────────┐
        ▼                                  ▼                                      ▼
  neuron_stats.py                    build_atlas.py                          scan.py
  data/stats_<m>.pt                  data/atlas_<m>.pt (+_docs.npz)          results/scan_*.json
  (each neuron's max/min)            top-40 firings per pole, every neuron   (chosen units, with context)
        │                                  │
        │                                atlas.py ── renders examples / blind dossiers
        │                                  │
        │            ┌─────────────────────┴───────────────┐
        │            ▼                                     ▼
        │   CENSUS: label_batch.py (Luna labels,     CONCEPTS: concept_pipeline.py (Pipeline)
        │     Jev grades) → regrade.py (rubric)            └► pipeline_fast.py (FastPipeline)
        │     → spliced_control.py → calibrate_clean.py         recall → attribution → ablation → steering
        │                                                           │
        └──────────── steering strengths are multiples ─────────────┤
                      of each neuron's own peak (stats)             ▼
                                         logodds_check.py · steer_select.py · auto_steer_scale.py · wave_free.py · dial.py
```

### External tools
- **`oai.py`, Luna** (OpenAI `gpt-6-luna`): writes neuron explanations and verdicts (ONE / TWO_RELATED / MIXTURE /
  NONE). It also writes recall prompts and checks answer wordings.
- **`jev.py`, Jev** (TypeSafe "System One"): returns calibrated probabilities, not text. It grades label coverage
  (yes/no or a 3-level rubric) and judges steered replies ("on-topic?", "fluent?"). Billing is per input token.
- **`cache.py`:** every API result is appended to `<out>.cache.jsonl` as it arrives, so long runs can resume. These
  caches are also the raw judge outputs behind every number.
- **`config.py`:** paths and keys come from environment variables or `.env` (see README, "Configuration").

## Pipeline stages

### 0. Infrastructure: model hooks, ranges, atlas
| Script | Does | Writes |
|---|---|---|
| `neurons.py` | Loads a model. `capture()` hooks the neuron activations (and gate/value). `steer()` clamps or adds to chosen neurons. | – |
| `neuron_stats.py` | Each neuron's max and min over 2M tokens (its "own peak", used to scale steering) | `data/stats_<model>.pt` |
| `build_atlas.py` | One pass over 10M FineWeb tokens: top-40 positive and negative firings for all 442,368 neurons, with the gate sign | `data/atlas_<model>.pt`, `data/atlas_<model>_docs.npz` |
| `atlas.py` | Looks up a neuron's top firings in the atlas; writes blind dossiers; `output_words` (direct vocabulary effect) | – |
| `scan.py` | Streams web text for a chosen set of units (neurons, planted mixtures, random directions), with context | `results/scan_*.json` |

### 1. Atlas and census labelling (Luna + Jev)
| Script | Does |
|---|---|
| `make_dossiers.py` | Blind markdown dossiers, one per unit, identified only by a random code (key kept separately) |
| `label.py` | Blind labelling: Luna explains each pole, Jev grades coverage per example. Used by Experiment GG and by `concept_pipeline` |
| `label_batch.py` | The census labeller. Same judgement as `label.py`, heavily parallel, with a batched random-snippet check. `random:N` samples neurons |
| `regrade.py` | Re-grades existing explanations with the validated 3-level rubric, keeping soft scores (`coverage_soft`, `random_soft`) |
| `grader_validation.py` | Checks the coverage grader against known answers: own examples vs another neuron's (AUC per question wording) |
| `spliced_control.py` | Two-job dossiers by construction: 10 examples from neuron A plus 10 from an unrelated neuron B |
| `calibrate_clean.py` | Picks the "clean" bar so that ≤ 10% of spliced dossiers pass, then restates the census with bootstrap CIs |
| `strata_scan.py` → `strata_grade.py` | Does a label still fit at 50/25/10% of peak? Also builds planted activation mixtures |
| `intruder_test.py` | Explanation-free check: can Jev spot a random snippet among a unit's top firings? Also provides `random_snippets` |
| `grader2.py` | Second grader: Luna scores the same rubric; correlation with Jev |
| `ctx_compare.py`, `effort_compare.py` | Robustness: grading context length; Luna reasoning effort |
| `spotcheck_build.py`, `spotcheck_score.py` | Blind human spot-check set (`site/spotcheck.html`) and its scoring |

### 2. Concept pipeline: recall prompts and answer discovery
`concept_pipeline.py` defines `Pipeline`, which runs one concept end to end. `pipeline_fast.py` subclasses it as
`FastPipeline`, with the same measurements, batched GPU use and one metric fix. The steps are:

1. **Recall prompts.**
   - Luna writes recall prompts that end right before the concept's name, plus control prompts with other answers.
   - A leak filter drops prompts that give the answer away.
   - The spec is saved as `<concept>_spec.json`.
2. **Recall format** (`recall_variants.py`). V0 is completion ("… is"); V3 is a chat question ("Answer with just the
   name."). V3 is the standard from CR-2 on. Select it with `RECALL_VARIANT=V3`.
3. **Answer discovery** (`answer_discovery.py`, built into `pipeline_fast`).
   - The model answers each prompt first, and Luna checks that the answer names the concept.
   - The model's own wordings are then scored as `answer_forms_extra`.
   - Made-up concepts (`prompts/fake_concepts.json`) guard against faked recall (ISSUES #31).
4. **Recall** = P(whole answer) ≥ `CONCEPT_MIN_P` on enough prompts. It is teacher-forced, so multi-token answers
   count.

Concept lists are in `prompts/`: `golden_gate.json`, `concepts_wave{2,3,4}.json`, `scale_set.json`,
`fake_concepts.json`. The neutral steering prompts are `PROMPTS` in `steer_test.py`, plus
`prompts/neutral_heldout.json` and `prompts/neutral_person_friendly.json`.

### 3. Attribution
- **`Pipeline.attribute`.** Ranks every neuron (and attention head) by activation × gradient of log P(answer), summed
  over positions and averaged over prompts. Keeps the top 200, the bottom 50, per-prompt and per-position maps.
- **Re-ranking by measured knockout.** The top 200 are re-ranked by ablating each neuron alone (`k50_check.py`, and
  inside `logodds_check.py` / `wave_free.py`), because attribution alone mis-ranks.
- `says_name.py` and `attribution_diag.py` read each neuron's *direct* effect on the vocabulary: output weights →
  final norm → unembedding.

### 4. Ablation and log-odds
- `Pipeline.ablation` zeroes the top-k neurons (k = 1…50) against random same-layer neurons, and reports `k50`
  (fewest neurons to halve P). **That metric saturates** near P ≈ 1 (ISSUES #27).
- **`logodds_check.py`** is the replacement. Neurons are taken in measured order, and the script reports:
  - `k_mean`: fewest neurons to lower mean log-odds by ≥ 3;
  - `k_median`: the same for the median prompt;
  - `k_flip`: fewest neurons to push P below 0.5 on half the prompts the model had right.
- `logodds_summary.py` and `logodds_compare.py` summarise the runs.
- Follow-ups:
  - `backup_routes.py`: iterative knockout (why near-certain facts resist);
  - `h_depth.py`: pre-registered depth stages;
  - `shared_parts.py` and `shared_parts_lexical.py`: part sharing between related concepts, plus its lexical control;
  - `lexical_check.py`: prompt-word confounds.

### 5. Steering and auto-selection
- **`Pipeline.steering`.** A grid over the number of neurons (2/4/6) and strength (0.5–2× each neuron's own peak), in
  clamp or add mode. Steering applies to the reply only (`--reply-only`) or to the prompt as well. Replies are judged
  by Jev (on-topic × fluent on the same reply). A random-neuron control runs at the best setting.
- **Selection rules** (`steer_select.py`):
  - R1: late-layer attribution;
  - R2: any layer;
  - R3: measured knockout;
  - R4: direct effect on the answer token;
  - R5: label-matched (`label_rule.py`, `label_match_luna.py`).
- **Held-out and stricter judging:**
  - `steer_retest.py` / `steer_retest_summary.py`: re-run on fresh prompts (winner's curse);
  - `strict_judge.py`: "main topic" wording;
  - `shuffled_judge.py`: is the judge too lenient?
- **People:** `people_prompts.py`, `people_rules.py`, `people_fluency.py`, `single_neuron_steer.py`.
- **Auto-selection:**
  - `auto_steer_scale.py` picks the best rule × count × strength per concept on the selection prompts, then scores
    it on held-out prompts;
  - `default_heldout_regen.py` and `auto_steer_jev_check.py` judge both sides with Jev.
- **`dial.py`:** an interactive version. Pick a concept, then `auto` selects the neurons, and you chat with the
  steered model.

### 6. Judge-free wave pipeline
`wave_free.py` runs a whole wave with no Jev:
1. V3 recall plus answer discovery;
2. log-odds concentration (`k_mean`, `k_median`, `k_flip`) against random neurons;
3. auto-selected steering (R1/R2/R4 × 2/4 neurons × 0.5–2×, add, reply-only), chosen on the selection prompts and
   scored on held-out prompts.

The steering score is "a distinctive answer word appears AND the reply passes `fluency_proxy.py`". That proxy is a
free local check (mean log-prob, repetition, non-ASCII share), validated against ~12k Jev verdicts (89% agreement).
`wave_summary.py` summarises a wave by category. The model-size comparison runs `wave_free.py` with `MODEL=` and
`prompts/scale_set.json`, writing to `results/scale_<model>/`.

### 7. Controls (where they live)
| Control | Guards against | Where |
|---|---|---|
| Random same-layer neurons | "any k neurons would do this" | every ablation and steering run |
| Unsteered replies | judge says yes to anything | steering runs, `people_prompts.py` |
| Made-up concepts | faked recall, leaky prompts | `prompts/fake_concepts.json`, all concept runs |
| Spliced two-job dossiers | a lenient labeller or grader | `spliced_control.py` → `calibrate_clean.py` |
| Planted activation mixtures | the same (weaker; ISSUES #25) | `make_units.py`, `strata_scan.py` |
| Random snippets | vague labels that fit anything | `label_batch.py`, `regrade.py` (`random_soft`) |
| Shuffled judge | lenient on-topic judge | `shuffled_judge.py` |
| Held-out prompts | winner's curse | `steer_retest.py`, `auto_steer_scale.py`, `wave_free.py` |
| Rewritten prompt sets (CR-2b) | prompt-set noise | `replicate_compare.py` |
| Test-retest / independent repeat | labelling noise | census redo, `h_depth.py` `_rep2` |
| Lexical checks | word-form detectors posing as concepts | `lexical_check.py`, `shared_parts_lexical.py` |

## Script status

**Pre-registered** (frozen before running; PREREG.md):
- CR: `concept_pipeline.py`, run by `scripts/run_replication.sh`;
- CEN: the census (`label_batch.py random:1000`);
- H-DEPTH: `h_depth.py`.

Everything else is exploratory.

**Superseded** (kept because results in the log came from them):
| Script | Superseded by | Note |
|---|---|---|
| `concept_pipeline.py` (as a CLI) | `pipeline_fast.py` | Still the base class; the CR results came from it |
| `Pipeline.ablation`'s `k50` | `logodds_check.py` | Saturates at high confidence (ISSUES #27) |
| `attribute.py`, `attribution_diag.py` | `Pipeline.attribute` | Golden Gate-only precursors (Experiment GG-2) |
| `gg_search.py`, `gg_recognition.py`, `make_units.py`, `merge_scans.py`, `summarise_gg.py`, `steer_test.py` (as a script) | the concept pipeline | Experiment GG, the first exploratory pass. `steer_test.py` still supplies `PROMPTS` and `generate` |
| `label.py` (for large sets) | `label_batch.py` | `label.py` is still used by `concept_pipeline` |
| `summarise_concepts.py` | `logodds_summary.py`, `wave_summary.py` | Summarises the v1 `concepts/` runs |
| `scripts/patch_pipeline_once.py` | – | A one-off code patch, already applied; kept for the record |

**Development and checks:** `smoke_test.py` (loads 0.6B, checks hooks), `jev_test.py` (three live API calls).

**Queue scripts** (`scripts/*.sh`) chained GPU jobs overnight. They wait on log files from earlier jobs and record
the exact commands behind several results. `download_models.sh` fetches the models with resumable curl.

## Data and results layout

| Path | Contents | In git? |
|---|---|---|
| `data/` | Atlas (`atlas_<m>.pt`, `_docs.npz`) and neuron ranges (`stats_<m>.pt`). Derived, large; rebuild with `build_atlas.py` / `neuron_stats.py` | no |
| `models/` (or `$ALREADY_CLEAN_MODELS`) | Local model weights | no |
| `prompts/` | Concept lists and neutral and held-out prompt sets | yes |
| `results/census_*`, `spliced_*`, `strata_*`, `effort_*`, `ctx*`, `grader*` | Census and its calibration and robustness checks | yes |
| `results/concepts/` | CR (pre-registered, v1 pipeline, completion format) | yes |
| `results/concepts_v3/`, `concepts_v3b/` | CR-2 and CR-2b (question format; independently written prompts) | yes |
| `results/concepts_w2/`, `_w3/`, `_w4/` | Coverage waves 2–4 (40 new concepts each; 3–4 judge-free) | yes |
| `results/concepts_disc*/`, `concepts_test/`, `concepts_dial/` | Answer-discovery debugging, pipeline tests, the dial's saved concepts | yes |
| `results/scale_Qwen3-0.6B/`, `scale_Qwen3-8B/` | Model-size comparison | yes |
| `results/logodds_*`, `steer_*`, `auto_steer_*`, `people_*`, `shared_parts*`, `hdepth_*`, `says_name*` | Per-experiment outputs and `.txt` summaries | yes |
| `results/dossiers_*/`, `*_KEY.json` | Blind dossiers and their answer keys | yes |
| `results/*.cache.jsonl` | Raw API responses (Luna and Jev), one JSON line per call | yes |
| `results/*.log` | Run logs | yes |

Each concept run writes `<concept>_spec.json` (prompts, answer forms) and `<concept>__<variant>.json`. The variant
is the steering setting (`clamp`, `clamp_reply`, `add`, `add_reply`). That file holds recall, attribution, ablation,
the steering grid with every reply and per-reply judge scores, and provenance (code hashes, git HEAD, library
versions).

## Known rough edges
- **Script style.** Many experiment scripts run their work at import time (there is no `main()`), and they read
  `sys.argv` directly. Run them as `python src/<script>.py …` from the repo root.
- **Tokenizer loading.** `atlas.py`, `label_batch.py`, `regrade.py`, `intruder_test.py` and a few others load the
  tokenizer from `$ALREADY_CLEAN_MODELS/<model>`. Download the models first (`scripts/download_models.sh`).
  `neurons.load()` falls back to the Hugging Face hub; tokenizer loads in those scripts do not.
- **Hard-coded model.** Several scripts assume Qwen3-8B (the atlas, census and steering prompts). The 0.6B comparison
  runs only through `wave_free.py` (`MODEL=Qwen/Qwen3-0.6B`).

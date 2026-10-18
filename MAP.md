# The Qwen3-8B map

A static, growing map of what Qwen3-8B's MLP neurons and attention heads do, built on top of the Eiffel Tower Qwen
explainer. It lives at `explainer/site/map/` (served next to the explainer, so `/map/` on the deployed site) and is
built from saved results by `explainer/gen/map/`. Nothing on the page runs a model.

## Layout of this work (release 2026-10-18)

- `explainer/site/map/` is the **released** page, served at `/map/`: the neuron map (`index.html`, `neurons.js`,
  `graphs.js`) and the data it reads (`data/neuronmap.json`, `concepts.json`, `n/`, `push/`, and `graphs/` once
  attribution graphs exist). Only those data files are committed (see `explainer/site/map/.gitignore`).
- `explainer/gen/map/lookup_site/` holds the **lookup-table pages** (concepts, layers, heads, findings, method and API
  in the original darker style). They aren't released yet; their data (c/, findings, tokens...) is generated but not
  committed. `make_artifact.py <out>` still packages them for private review.
- `explainer/gen/map/` holds every build and experiment script, the hand-written findings, and the method notes
  (JOB_NAMES.md, SCALES.md, LINKS.md, ATTRIB_GRAPHS.md).

## The neuron map (`map/index.html`)

The page we're converging on: the explainer's Figure 5 (layers as rows of dots, question in at the top) grown into
a map of every neuron we have tied to a concept, written for a lay reader in the explainer's style.

- **Layout.** 36 rows, one per layer. Layers 20-35 are grouped into columns by the Wikipedia section of the concepts
  a neuron mostly serves ("Shared" when it serves several). Layers 0-19 are not grouped: their neurons react to many
  things and their jobs are much less clear, so they sit in their real order. The grouping is ours, not the model's.
- **A concept** lights up its 30 key neurons as a sweep moves down the layers; a bar per row shows the answer's
  logit-lens rank. Tap a subject to zoom into its column (concept picks switch to that subject). Heads are an
  optional overlay.
- **Hover / tap** gives a plain-English description: how much less likely the right answer gets without the neuron,
  a **checked convenience name** where one exists, or "job not yet clear", the words it pushes (flagged as
  unreliable before layer 20), blind web-text labels, other findings, and the concepts that need it.
- **Convenience names** come from `explainer/gen/map/job_names.py` (method and pass rule, fixed before the run, in
  `explainer/gen/map/JOB_NAMES.md`): Luna names a neuron without seeing one of its concepts; Jev, given only the
  name, must pick that held-out concept out of five; a different neuron's name is the control. Only names with
  Luna confidence "high" and Jev p >= 0.5 are shown, always labelled as convenience names, not true meanings.
- Build: `python -I -X utf8 explainer/gen/map/build_neuronmap.py` (after `build_map.py`), which writes
  `data/neuronmap.json`. Package for a claude.ai artifact with `make_artifact.py <out> --page neurons`.

## What the lookup pages show (`explainer/gen/map/lookup_site/`, not released yet)

| View | URL | What's on it |
|---|---|---|
| Whole model | `#/` | Per layer: how often its neurons are among a concept's top 5, by role; its 32 heads; graded depth bands (LAYERS_AND_JOBS); pre-registered tests; sources |
| Concepts | `#/concepts` | Every concept tried (mapped or not recalled), filterable by Vital Articles section |
| Concept | `#/concept/<slug>` | Key numbers (k, k_flip, top drop, noise floor); **how it runs through the network** (key neurons by layer and drop, heads, and the logit-lens rank of the answer per depth); **what lights up** (attribution over all 442,368 neurons, 36 × 96 blocks, key neurons outlined); cumulative knockout vs random; the 30 key neurons with role and pushed words; questions; related concepts; second measurement where one exists |
| Layer | `#/layer/<L>` | Depth-band findings, where the mapped neurons sit, heads, every mapped neuron |
| Neuron | `#/neuron/L<l>:<n>` | Pushed words (weights), roles, blind labels (input side) with calibrated one-job verdicts, causal findings (entity / relation / fact jobs, web-firing judge), every concept that needs it with its drop against that concept's noise floor |
| Heads | `#/heads`, `#/head/L<l>H<h>` | 36 × 32 grid of how often each head matters; heads with a written-up role; concepts that lean on a head most |
| Findings | `#/findings` | Every claim from every source, searchable |
| How we know | `#/about` | Each source's method, status, reproduce command and caveats; the data API |

Every claim has a **how we know** button: the source's method, status (pre-registered or exploratory), reproduce
command, caveats and write-up.

## Build

```bash
# 1. once (reads ~30 GB of run-1 arrays, ~3 min on the 5090; resumable, cached in explainer/gen/map/cache/)
python -I -X utf8 explainer/gen/map/arrays_pass.py
# 2. every time anything changes (~10 s)
python -I -X utf8 explainer/gen/map/build_map.py
# 3. look at it
python -m http.server 8765 --directory explainer/site     # then open http://localhost:8765/map/
```

`arrays_pass.py` needs torch, numpy and safetensors (the project's `.venv`); `build_map.py` needs only the standard
library. Input paths default to this machine's layout and can be overridden: `MAP_RUN1`, `MAP_FRAME`, `MAP_EARLIER`,
`MAP_RESEARCH_DATA`, `MODEL_MAPPER_MODELS`, `MAP_OUT` (see `explainer/gen/map/common.py`).

The generated `explainer/site/map/data/` (~65 MB) is gitignored for now; deploying the map means building it first.

## Sources (one module each in `explainer/gen/map/sources/`)

| Module | What it adds |
|---|---|
| `loop_v2_run1` | 5,000 Vital Articles concepts from loop v2 run 1 (4,194 mapped): top 30, nulls, roles, pushed words, heads, provenance |
| `earlier_maps` | 914 maps from before loop v2 (CR-2, CR-3, waves 2-4, fame samples, coverage-3/4); roles inferred from run 1 and flagged |
| `labels` | Census (1,000 random neurons) and H-DEPTH blind labels, with the calibrated one-job bar |
| `web_firings` | Luna's judgement of concept neurons' web-text firings (about / related / unrelated) |
| `facts` | Facts-grid classes (entity / relation / fact-specific) and direct entity neurons |
| `curated` | Hand-written findings in `explainer/gen/map/findings/*.jsonl` (depth bands, head roles, pre-registered tests) |

A concept measured by two runs (78 today) is shown with both maps and their top-10 overlap.

## Adding findings

The map is meant to accrete. Two ways in:

1. **One-off claims:** append a line to a file in `explainer/gen/map/findings/`:
   ```json
   {"id": "X1", "unit": "L30:8291", "kind": "fact job", "title": "Relation neuron: 'century built'",
    "detail": "Costs ~3.2 log-odds on the century fact of every landmark tested ...", "strength": "moderate",
    "status": "exploratory", "docs": ["LAYERS_AND_JOBS.md#moderately-supported"]}
   ```
   `unit` is a neuron (`L30:8291`), a head (`L34H28`), a layer range (`layers:28-35`), a concept (`concept:<slug>`)
   or `model`. Optional: `metric` (numbers), `concept` (slug it is about), `source` (defaults to `syntheses`).
2. **A whole experiment:** add a module to `sources/` with a `SOURCE` dict (title, kind, status, date, method,
   reproduce, docs, caveats: this is what the "how we know" popover shows) and a `load(ctx)` that calls
   `ctx.add_finding(...)` (see `common.finding`) or `ctx.add_map(...)`; register it in `build_map.SOURCES`.

**Planned:** the research session is building a ledger (Parquet packs -> DuckDB, spec in `LEDGER.md` on
research/attention) with a JSONL export (`experiments`, `concepts`, `units`, `evidence/<experiment>`, `claims`,
`views/`). Once that export exists, a single `ledger` source here will read it and the hand-written adapters above
will be retired; the viewer does not change.

## Data API (read-only static JSON next to the page)

| Path | Contents |
|---|---|
| `data/meta.json` | Model, revision, counts, sources (method, status, reproduce, caveats), build time and commit |
| `data/concepts.json` | `{cols, rows}`: slug, name, answer, section, tier, kind, type, status, k_mean, k_flip, top1, sources, noise, baseline |
| `data/c/<slug>.json` | One concept: maps (top 30 as `top_cols`, drops, random_drops, null, heads, prompts, provenance, `lens`, `light`), related, findings, replication |
| `data/n/L<layer>.json` | Every mapped neuron of a layer keyed by index: `uses` = [concept row, source code, rank, drop], role(s), pushed words, hub flag, findings |
| `data/heads.json` | Per head: n concepts (top 50), n with knockout >= 0.5, median, max, top concepts, findings |
| `data/layers.json` | Per-layer aggregates and depth-band findings |
| `data/findings.jsonl` | Every finding, one per line |
| `data/tokens.json` | Pushed word -> neurons (say / attribute neurons) |

Ids follow the loop v2 join keys: neurons `L<layer>:<index>`, heads `L<layer>H<head>`.

## Known limits

- Concept maps cover one behaviour (naming an entity in answer to a question); a neuron's job here is not its only one.
- Ranking by effect on the answer favours late layers; early neurons are under-represented in maps.
- Single-neuron drops near the per-concept noise floor are bf16 noise; the views grey them out (within 2x the floor).
- Logit lens is a readout of the residual stream, not a causal measurement; early-layer ranks are noisy.
- Doc links point at `main` on GitHub; research-branch documents (LAYERS_AND_JOBS, PREREG_MAP, RESEARCH_STATE) will
  404 until they are published.

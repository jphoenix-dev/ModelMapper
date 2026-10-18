# The neuron / concept ledger (spec v1, 18 Oct 2026)

One place that records **everything we have measured about which units (MLP neurons, attention heads, later other
features) relate to which concepts, in which model, by which method, how strongly, against which noise floor**. It is the
backend for the map visualiser. Design goals, from James:

1. **Lossless over time** - every relevant measurement goes in; nothing is summarised away; dense tensors stay
   reachable.
2. **Scalable** - millions of rows today (run 1 alone is ~2M), billions later (more concepts, more models).
3. **Salient on demand** - the front end shows only what matters, by querying derived views; salience is a versioned
   rule, never baked into the data.
4. **Easy to extend** - by us with findings we have not imagined yet, and by a third party, without schema changes.

## Architecture

```
packs/<pack_id>/                 immutable, append-only contributions (one per importer run or contributor)
    pack.json                    who / what / when / method / code commit / source files + sha256 / schema version
    evidence.parquet             the measurements (long format, one row per measured fact)
    concepts.parquet             concepts this pack mentions (optional)
    experiments.parquet          experiment metadata (optional; else from pack.json)
    artifacts.parquet            pointers to raw / dense files (optional)
    claims.parquet               curated statements built on evidence (optional)
ledger.duckdb                    BUILT from all packs: views over the Parquet + derived summary tables. Never hand-edited;
                                 delete and rebuild any time (python src/ledger/build.py).
views/                           exports for the front end (derived; see "Front end")
```

- **Packs are the source of truth.** A pack is never modified. Each importer's CURRENT pack is its most recently created
  (or re-activated) one; older packs stay on disk for the record and the build ignores them. A pack may also explicitly
  `supersede` packs of other importers (pack.json).
- **Pack ids are deterministic**: `<importer>__<sha256 of its inputs (path, size, mtime), its code and core.py, 12 hex>`,
  so re-running an importer on unchanged inputs is a no-op (it re-activates that pack).
- **Row keys** include the source record (`file#locator`), so two source spellings of one concept (e.g. 'Odyssey' and
  'The Odyssey') keep separate rows under one concept id.
- **Third parties** contribute a pack as JSON lines (`evidence.jsonl`, ...) plus `pack.json`;
  `python src/ledger/pack.py from-jsonl <dir>` validates and converts it to Parquet (see `src/ledger/pack.py` for the
  format; `src/ledger/contrib/research_claims/` is a working example - our own curated claims enter this way).
- Location: `D:/already-clean-data/ledger/` (override with `LEDGER_DIR`). Code: `src/ledger/`.

## Identifiers

| thing | id format | examples |
|---|---|---|
| model | short slug | `qwen3-8b` |
| unit | `<model>/L<layer>/<kind letter><index>` | `qwen3-8b/L31/N8930` (MLP neuron), `qwen3-8b/L23/H10` (attention head) |
| concept | `<kind>:<slug>` (slug = accent-folded lowercase ASCII, `-` between words; things drop a leading the/a/an) | `thing:eiffel-tower`, `era:century-19`, `relation:year-completed`, `digit:7` |
| experiment | `<pack-local name>` unique within the ledger | `loop_v2_run1`, `era_formats` |
| evidence | sha256 of (pack_id, row key), 16 hex | |

Unit kinds and concept kinds are a controlled but extensible vocabulary (`src/ledger/vocab.yaml`). Concepts merge
across packs by id; `aliases` (other labels, e.g. the answer form 'Xinhai Revolution' for `thing:1911-revolution`) help
matching.

## Tables

**evidence** (the core; long format - a new kind of finding is a new `evidence_type`, not a new column)

| column | type | meaning |
|---|---|---|
| evidence_id | str | deterministic id |
| pack_id, experiment_id | str | provenance |
| model_id | str | |
| unit_id | str, nullable | the unit measured (null for concept-level facts, e.g. a concept's recall) |
| unit_kind, layer | str, int | denormalised from unit_id for fast filtering |
| unit2_id | str, nullable | second unit, for unit-to-unit evidence (feeds, chains, fan-out) |
| concept_id | str, nullable | the concept the measurement is about |
| concept2_id | str, nullable | second concept (sharing, cross-entity specificity) |
| context | str, nullable | conditions, `key=value;...` (e.g. `format=year`, `position=name_tokens`, `prompt_set=recall_v3`) |
| evidence_type | str | what kind of fact (vocabulary) - e.g. `knockout_drop`, `attribution_rank`, `activation_auc`, `role`, `readout_tokens` |
| metric | str | how it was scored - e.g. `mean_logodds_drop`, `rank`, `auc`, `relative_activation_drop` |
| value | float64, nullable | the number |
| value_text | str, nullable | categorical value (a role, a class, a token) |
| uom | str, nullable | unit of measure (`logodds`, `nats_per_token`, `fraction`, `digits`) |
| n | int, nullable | sample size behind the value (prompts, entities) |
| noise_floor | float64, nullable | the null it should be read against (e.g. random-neuron 99th percentile) |
| threshold | float64, nullable | the cut used for `passes` |
| passes | bool, nullable | value beyond threshold in the meaningful direction |
| rank | int, nullable | rank within the experiment's list for this concept (1 = strongest) |
| details | JSON str | everything else - per-prompt values, per-position values, controls, raw sub-results |
| source | str | where it came from: `<path relative to data root>#<record locator>` |

**concepts**: `concept_id, label, kind, aliases (list), attributes (JSON: wikidata, frame section, fame tier, named /
common, year...), pack_id`.

**experiments**: `experiment_id, pack_id, title, method (prose), status (long), status_class (pre-registered |
exploratory | invalid), date, script, reproduce (command), code_commit, data_files, caveats, docs (doc#section title),
params (JSON: thresholds, noise floors, predictions and outcomes), description`.

**artifacts**: `artifact_id, pack_id, experiment_id, path, bytes, sha256 (nullable for very large files), format,
keys (JSON: e.g. arrays inside an .npz and their shapes), concept_id, unit_id, description`. Dense data (run 1's
per-concept attribution / activation / residual / attention arrays, 32 GB) is registered here, not copied.

**claims** (curated, small): `claim_id, kind (finding | prediction | caution), scope ('L31:8930', 'layers:28-35',
'model', ...), unit, concept, category (layer band, head role, unit summary, ...), text, detail, grade (strong |
moderate | not established | failed), status, metric (JSON), evidence_sql (a query over evidence returning the supporting
rows), evidence_ids, docs, pack_id`. This is where statements like "L31:8930 carries the 1800s across formats" live, linked to their evidence.

## Derived views (in ledger.duckdb; rebuilt; versioned in `src/ledger/salience.sql`)

- `units` - every unit with any evidence: layer, kind, # concepts with passing evidence, evidence types present.
- `unit_concept` - one row per (unit, concept): the strongest passing evidence per evidence type, an overall strength
  (effect / noise floor where both exist), and the experiments that support it.
- `unit_summary` - per unit: top-k concepts by strength, role(s), readout tokens, sharing count, claims mentioning it.
- `concept_summary` - per concept: top-k units by strength, layers, heads, recall, claims.
- `layer_summary` - per (model, layer): counts by evidence type / role / concept kind.

**Salience is a rule, not data:** the default rule keeps evidence with `passes = true` (or, where no threshold exists,
the top-ranked rows), orders by effect relative to its noise floor, and is versioned (`salience_version` in every view).
Changing what the front end shows = editing `salience.sql`, never the packs.

## Writing an importer (us or a third party)

```python
from ledger.pack import PackWriter
with PackWriter("my_experiment", inputs=[...source files...], code=__file__) as pw:      # deterministic pack id
    pw.experiment("my_experiment", method="...", description="...", params={...}, results_doc="...")
    pw.concept("thing:eiffel-tower", "Eiffel Tower", "thing", attributes={...})   # or core.thing(pw, "Eiffel Tower")
    pw.evidence(unit=("qwen3-8b", 31, "N", 8930), concept="era:century-19", evidence_type="activation_auc",
                metric="auc", value=0.96, n=280, threshold=0.8, passes=True, context="format=year",
                details={...}, source="era/era_rows.json#L31:8930")
```

Rules: (1) one row per measured fact, never pre-aggregated away (aggregates may be added as extra rows with
`metric` naming them); (2) always fill `noise_floor` / `threshold` when the experiment has one; (3) unknown
`evidence_type` / concept kinds are allowed but must be added to `vocab.yaml` with a definition (the validator warns);
(4) informal readings (e.g. readout tokens) are evidence of type `readout_tokens`, never labels on the unit.

## Commands

| command | what |
|---|---|
| `python src/ledger/import_all.py` | run every importer (no-op when unchanged), build, verify, export |
| `python src/ledger/importers/<name>.py [--force]` | one importer -> one pack |
| `python src/ledger/pack.py validate|from-jsonl <dir>` | third-party JSON-lines contribution -> pack |
| `python src/ledger/build.py [--check]` | rebuild ledger.duckdb from the current packs (+ vocabulary / id checks) |
| `python src/ledger/verify.py` | lossless check: recount source records and compare values with the ledger |
| `python src/ledger/export.py` | the map visualiser's export (below) |
| `python src/ledger/query.py unit L31:8930` | everything known about a unit; also `concept`, `find`, `shared`, `experiments`, `sql` |

Importers (src/ledger/importers/): run1, earlier_maps, facts (facts + facts2: grid, classes, position, entity maps,
specificity, entity sharing, chains), era, numeric (+ magnitude), shared_neurons (+ feeders), forced_readout (pilot + fast),
fanout, specificity, concept_search, search_verify, concept_atlas, labels (census + H-DEPTH), web_firings, curated_claims
(the map's findings/*.jsonl). Dense data (run 1 npz arrays, atlas activations, concept-search scores for all 442,368
neurons) are artifacts, not rows.

## Scaling notes

- Today: ~2M evidence rows, 380 MB of packs, build ~40 s, export ~90 s. Packs are columnar Parquet (zstd) read by DuckDB
  directly, so 100x more rows is fine on one machine; the build only materialises the derived tables.
- The export shards evidence per experiment into gzipped parts of <= 100,000 rows; consumers can read only what changed
  (manifest.json has sha256 per file).
- A new model is a new `model_id` in the unit ids (`<model>/L../N..`); nothing else changes.
- If per-neuron x per-concept DENSE data (e.g. run 1's full attribution maps, 1.8B values) ever needs querying, convert
  the registered npz artifacts to a chunked store (Zarr / Parquet by layer) as a derived table - the artifacts already
  point to them.

## Front end

Agreed with the map visualiser session (see "Interface" below, filled in once agreed). The ledger provides: the DuckDB
file + Parquet packs (queryable directly, including from the browser via DuckDB-wasm), a Python query API
(`src/ledger/query.py`), and exports of the derived views per zoom level under `views/`.

## Interface (agreed with the map visualiser, 18 Oct)

The map's static build (`explainer/gen/map/sources/ledger.py` on branch research/map-page) reads
`D:/already-clean-data/ledger/export/`:
- `manifest.json` (files with rows / bytes / sha256, active and superseded packs, schema and salience versions),
- `experiments.jsonl` (one map "source" per experiment: title, method, status, status_class, reproduce, caveats, docs),
- `concepts.jsonl` (concept_id, name, slug = the map's own slug, qid, kind, aliases, attributes), `units.jsonl`,
- `evidence/<experiment_id>/part-NNNN.jsonl.gz` (one row per measured fact with the short unit key 'L31:8930' / 'L23H10',
  a rendered one-sentence `claim`, `strength` = value / noise floor, `passes_threshold`, `context`, `details`, `source`),
- `claims.jsonl`, `artifacts.jsonl`, `views/{unit,concept,layer,head}_summary.jsonl`.
Controls carry `control=...` in `context`. Ownership: this repo owns the ledger and importers; the map owns the viewer
and its single ledger adapter.

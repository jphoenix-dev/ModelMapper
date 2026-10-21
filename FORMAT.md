# Formats

Two formats: the **record** that `map-concept` and `map-batch` write for each concept, and the **published map
summary** that `lookup` reads. The record is the same one the Qwen Neuron Map is built from (schema `loop_v2.0`), so
records from any install can be pooled with the published ones. Everything below is plain JSON.

In both, a neuron is written `L<layer>:<index>` (`L23:2392`, 0-based) and a head `L<layer>H<index>` (`L19H29`).
Drops are in log-odds of the answer (METHOD.md), larger = the answer needs it more.

## The record: `maps/<model>/<concept slug>.json`

`map-batch` also appends each record, on one line, to `maps/<model>/records.jsonl`. In Python, `MapResult.load(path)`
reads one, and `r.record` is the dict below.

```
{
 "concept": "Bitcoin",
 "row": {                                   # the summary row
   "status": "ok" | "not_recalled" | "failed",
   "detail", "traceback"                    # only when failed
   "answer": "Bitcoin", "answer_form": "Bitcoin",        # the form that scored best
   "recall": 9,                             # questions answered correctly (P >= 0.2); 5 needed
   "discovered": [...],                     # extra answer forms the labeller accepted
   "answer_form_scores": {form: [P per question]},
   "recall_prompts": [chat-formatted questions that were kept],
   "baseline_mean_logodds": 0.26, "base_per_prompt": [...],
   "k_mean": 5, "k_median": 5, "k_flip": 1,              # null when even 50 neurons don't reach it
   "drops":        [mean drop with the top k off, for k = 1, 2, 3, 5, 10, 20, 50],
   "random_drops": [the same for random neurons]
 },
 "rank": {                                  # null unless status is ok
   "answer_token": " Bitcoin", "answer_ids": [...], "recall": 9, "k_mean": 5, "k_flip": 1,
   "baseline_mean_logodds": 0.26, "base_per_prompt": [...], "prompts": [...],
   "candidates_all200": [                   # by drop, largest first
     {"neuron": "L27:81", "single_drop": 1.366, "attr_rank": 0, "act_answer_pos": -226.544,
      "margin": 1.098, "call": "uncalibrated"}, ...],   # margin and call: added by the guards (below)
   "top": [the first 30 of candidates_all200],
   "per_prompt_drop_all200": {neuron: [drop per question]},
   "heads_top50_by_attr": ["L19H29", ...],
   "head_knockout_per_prompt": {head: [drop per question]}, "head_knockout_mean": {head: drop},
   "matched_original": false, "mismatch": null           # kept for compatibility with run 1
 },
 "v2": {
   "provenance": {"schema": "loop_v2.0", "model", "dtype", "torch", "transformers", "python", "gpu", "code_commit",
                  "started_utc", "finished_utc",
                  "package", "device", "cuda", "cudnn", "numpy", "os", "machine", "gpu_memory_gb", "model_revision",
                  "prompt_format"},       # "chat" (every published map) or "plain" (0.1.1)
                                            # "unknown" when a field can't be found; "none" when it doesn't apply
   "timing_s": {step: seconds}, "prepare_timing_s": {labeller step: seconds},
   "null_single_drops": {random neuron: [drop per question]},   # 2 random same-layer neurons
   "roles_top30": [{"neuron", "answer_rank", "answer_z", "role", "job_tokens": [30 words it pushes],
                    "approximate": true}],   # only on models with a post-MLP norm (0.1.1; METHOD.md step 9)
   "arrays": "bitcoin.npz",                 # in arrays/ next to the record
   "map_one_notes": [...]                   # only if something was worth noting
 },
 "guards": {                                # added by this package (records from run 1 don't have it)
   "band": {"value": null, "source": "no calibrated band for Qwen3-0.6B: calls are uncalibrated"},
                                            # Qwen3-8B: {"value": 0.13, "source": "Qwen3-8B, 3090 vs 5090, ..."};
                                            # or the override used
   "bar_rank": 30, "bar": 0.268,            # the 30th candidate's drop, in this run
   "n_in": null, "n_uncertain": null,       # calls among the top 30 (numbers once there is a band)
   "n_out_but_close": null                  # candidates ranked 31+ within the band of the bar
 }

A call is "in", "uncertain", "out", or "uncalibrated" (the model has no band).
}
```

New in 0.1.0: the provenance fields after `started_utc` (`prompt_format` in 0.1.1), each candidate's `margin` and `call`, and the `guards`
block (METHOD.md, "In, uncertain or out"). Everything else is exactly as run 1 wrote it.

`arrays/<slug>.npz` (float16): `attr` and `last_act` [layers, neurons], `pmax` / `pmin` (mean of each neuron's max /
min over the prompt), `top200`, `head_attr`, `head_out_answer_pos`, `resid_answer_pos` [questions, layers + 1,
d_model], and `att<i>`: question i's last-position attention rows [layers, heads, tokens].

`spec_cache/<slug>_spec.json` is the question set (`answer`, `recall` fragments, `controls`, `recall_v2`, `recall_v3`
questions, `discovery_replies`, `answer_forms_extra`). Pass it back with `--questions` to repeat a run on the same
questions.

## The published map summary

`lookup` reads it from `MODEL_MAPPER_MAP_URL` (default `https://eiffel-tower-qwen.vercel.app/map/data`) and caches it
in `~/.cache/modelmapper/map/`. It is the Qwen Neuron Map's own data, for Qwen3-8B.

**`concepts.json`**: one row per concept, as `{"cols": [...], "rows": [[...], ...]}` with cols
`slug, name, answer, section, tier, kind, type, status, k_mean, k_flip, top1, sources, noise, baseline`.
A concept's position in `rows` is its index below.

**`n/L<layer>.json`**: one file per layer, `{neuron index: entry}` for every neuron in some concept's top 30.

```
{"2392": {
   "uses": [[concept index, source, rank, drop], ...],   # source 0 = run 1, 1 = earlier maps; rank is 1-based
   "n_uses": 14,
   "role": "attribute", "roles": {"attribute": 9, ...},  # its usual role, and how often it had each role
   "jobs": [words it pushes],                            # from the use with the largest drop
   "az": 2.4,                                            # its largest answer z-score across its uses
   "top10": 3,                                           # how many concepts have it in their top 10
   "hub": false,                                         # in the top 10 of very many concepts
   "findings": [...]}}                                   # other experiments' results on it (titles and sources)
```

`lookup --json` prints the decoded result: for a neuron, its uses with concept names; for a concept, its neurons
across all layers.

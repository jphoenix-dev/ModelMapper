# Loop v2 plan (draft for James to mark up, 17 Oct 2026)

**What v2 is:** the proven concept-first mapping loop (`map_loop.py`), unchanged in its science, with the "regret
list" built in before we scale it. **What v2 is not:** the concept atlas / search-then-verify (parked as a side
tool), bottom-up profiling of every neuron (the distributed session's pipeline), or new task types beyond naming an
entity (a later Phase 1 repeat).

## 1. The core loop (unchanged)
Per concept:
1. **Pick** a named concept from the sampling frame (section 4).
2. **Luna writes ~10 recall questions** whose answer is the name, without giving it away. Qwen's own free answers
   catch alternative forms of the name, and Luna judges which of them name the concept.
3. **Recall filter:** keep the questions Qwen recalls (P >= 0.2, leak filter); need >= 5, otherwise "not recalled".
4. **Shortlist:** attribution (gradient x activation) over all 442k neurons; top 200 candidates.
5. **Rank:** each candidate zeroed alone at every position; drop in exact log-odds per question; top 30 = the map.
6. **k curves:** top 1..50 zeroed cumulatively, giving k_mean, k_median and k_flip; the same curve for random neurons
   as a control.
7. **Heads:** per-head attribution; the 50 most-attributed heads each zeroed alone; head outputs and attention
   patterns at the answer position.

Validated: reproduces the original pipeline exactly on 12 concepts; P1-P5 all hold on its output; 914 concepts mapped.

## 2. Efficiencies carried over (keep all of these)
| Lesson | Evidence | Keep as |
|---|---|---|
| Luna and GPU as separate queues meeting through cached spec files; the GPU takes whatever is ready and logs its idle time | Coverage-4: 376 concepts, 21 s each, GPU idle 0.3%; the old loop sat idle waiting on Luna (0.64 min per non-recalled concept) | unchanged |
| Separate Luna pool for judging, so judges don't queue behind spec writing | Judge starvation bug, fixed 17 Oct | unchanged |
| Cached specs reused, so Luna is skipped on reruns | - | unchanged |
| Rank once; no steering in coverage mode | Steering was costly and ~0 for Wikipedia titles | unchanged |
| Append-only per-concept logs (jsonl); full JSON written once at the end | Rewriting a growing JSON per concept idled the GPU (James's GPU-Z trace) | unchanged |
| Activation maps from one prompt-only sdpa pass; attention patterns from a separate eager pass | Maps match cross-activation (64,836 values, max diff 0.125) | unchanged |
| Exact float64 log-odds (no cap) | ISSUES #35 | unchanged |
| Per-step wall time and Luna tokens/cost per concept; peak GPU memory in the run meta | - | unchanged, plus GPU busy % |
| `.detach()` on every saved tensor | Memory-mapped tensor bug | unchanged |
| **Don't** batch single knockouts across candidates | Failed its gate twice; only 1.1x faster (the GPU is already full) | stays unbatched |
| **Don't** shrink the 200-candidate shortlist | Coverage-4: the top-5 lies fully inside the first 150 candidates for only 94% of concepts (100: 87%) | stays 200 |
| Batch and overlap where batches are tiny (questions, recall checks) | verify_fast: GPU 100% busy once Luna was overlapped and names batched | apply to any new side step |

**Where the time goes now** (coverage-4 timings): single knockouts ~14 s per concept, head knockouts ~3.4 s, the
rest ~4 s. So scaling comes from more GPUs running concepts in parallel (linear; the distributed session's runner),
not from squeezing one GPU further.

## 3. Additions before scaling (the regret list)
| # | Addition | Cost per concept (estimate) | Why |
|---|---|---|---|
| a | **Schema + provenance:** one versioned record per concept; required fields always present; model name and weights hash, code git SHA, dtype, batch shapes, seeds, Luna model, timestamps | ~0 | The 359 older maps lack prompts; only 304 have head data |
| b | **Output on D:** a single setting (`AC_DATA_DIR`), default `D:/already-clean-data/loop_v2/<run>/` | 0 | James, 17 Oct |
| c | **A null in every measurement:** 2 random same-layer neurons zeroed alongside the ranking, as a per-concept noise floor (the random k-curve stays) | ~0.2 s | bf16: any change that flips a rounding moves results by a similar amount |
| d | **Checks verified from records:** every gate counts from the saved rows at the end of a run, not from in-loop counters | 0 | An edit once silently disabled live counters (interference run) |
| e | **Self-checks every batch:** P1-style, the top-5 must break recall on Wikipedia cloze prompts too (cheap, ~4 prompts); P4-style, family neurons predict the batch's new concepts (CPU, at the end of a batch); report pass/fail per batch | ~1 s + a CPU step | The loop checks itself as it scales |
| f | **Raw ingredients for later units:** the residual stream at the answer position for every recall question (36 x 4096 fp16 = ~0.3 MB per question, ~3 MB per concept) | ~0.5 s, ~3 MB | Lets us switch to SAE/transcoder features or new analyses without rerunning |
| g | **A deliberate sampling frame** (section 4), with the draw log saved | 0 | Otherwise the atlas is biased by construction |
| h | **Join keys with the bottom-up pipeline:** neuron ids `L<layer>:<index>`, head ids `L<layer>H<head>`, document/position ids in the same format as `src/dist` | 0 | So the two directions link up |
| i | **Nothing Qwen-specific:** layer and width counts from the model config; no hard-coded 36 / 12288 | 0 | Goal: any model |
| j | *Optional, ~free:* each top-30 neuron's role from its weights (say / attribute / other) and its job tokens (the 30 tokens it pushes most) | ~0.1 s | Makes maps more queryable (today's findings) |

Expected cost after additions: ~22-23 s per concept on the 5090, ~$0.002 of Luna; storage ~5-8 MB per concept on D:.

## 4. Sampling frame (decision needed)
Proposal: draw concepts from English Wikipedia, **stratified by fame** (language-edition count, as in the fame
sample) **and by Wikidata type** (P31: person, place, organisation, work, event, species, substance, idea...), with
fixed seeds and a saved draw log, so the atlas covers what the model knows and not mostly famous people. The recall
filter still drops what Qwen doesn't know (recorded as data, not discarded). Concepts that aren't named entities
(relations, grammar, arithmetic) are a later Phase 1 repeat, not v2.

## 5. Gates before the first big run
1. **Reproduction:** the 12-concept repro set gives identical status, recall, answer form, k values and top-10 sets
   to v1 (as for v1 itself).
2. **Schema check:** every new field present for every concept; provenance filled in.
3. **Self-checks run** on one batch of ~50 new concepts and report.
4. **Timing:** per-step times and GPU busy % within ~10% of the estimate above.

## 6a. James's decisions (17 Oct, evening)
1. Sampling frame: fame x Wikidata type, as proposed (he had the same point: broaden away from famous names).
2. Residuals at the answer position: yes.
3. Weights-based roles and job tokens (j): yes.
4. First run ~1,000 concepts: yes, built with the gpu-fan-out session so that the same task code runs on the 5090 or on
   rented GPUs. Goal: a frictionless choice between local and cloud for bulk work, under cost caps. Split: PREPARE on
   the desktop (Luna + light GPU), MAP as an idempotent `map:<batch>` task in the src/dist runner (pure GPU; module
   src/loop_v2/map_core.py), CHECK on the desktop.
Next question after v2 (James): is "named thing" too narrow - would the loop work for "blue", "117", "bicycle",
"table"?

## 6. Decisions for James
1. Sampling frame: fame x Wikidata type as proposed, or something else?
2. Save residuals at the answer position (~3 MB per concept)? Recommended.
3. Include the optional role/job enrichment (j)?
4. Size of the first v2 run (e.g. 1,000 concepts, ~6 hours on the 5090, or spread across the distributed GPUs)?

## Parked
Concept atlas / search-then-verify (works but only a 16% hit rate; a possible way to choose names later); bottom-up
profiling of all neurons (the distributed pipeline); quantisation; new task types; a Qwen3 size curve; a cross-family
model check.

## Run 1 (launched 17 Oct ~22:15)
- **Frame:** Wikipedia Vital Articles Level 4 (10,021 articles, 11 sections), stratified section x fame (language
  editions: 128+, 64-127, 32-63, <32), seed 71: 5,000 concepts (~470 per section; all 298 eligible Mathematics);
  401 already-mapped concepts and 3 bad titles excluded (frame.py; frame.json committed under results/loop_v2/run1/).
  Named / common tags (intro capitalisation, else Wikidata P279) added by `frame.py tag` in parallel, for analysis only.
- **Prepare** (prepare.py, desktop): map_loop's Luna steps unchanged, 32 Luna workers, Luna cap $12 (balance $21.83);
  G1 on the local 5090. Trial on 3 concepts: $0.0017 per concept, 0.5 min.
- **Map:** src/dist runner (gpu-fan-out session): store s3://modelmapperdistributed/loop_v2_run1, batches of 20
  appended every 2 min as specs appear; 3 rented pods (3090 / 4090 / A5000), --max-spend 7.5, --min-balance 1.5,
  approved by James in that session; results drained to D:/already-clean-data/loop_v2/run1/results; the local 5090
  joins after prepare.
- Expected: ~3,500 recalled concepts mapped. Next: self-checks per batch and a morning report (coverage, named vs
  common, overlap, timings, costs).

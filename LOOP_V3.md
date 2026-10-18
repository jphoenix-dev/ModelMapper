# Loop v3 = an ENRICHMENT pass (plan, 18 Oct 2026)

James (18 Oct): "enrich our existing neurons/concepts first so they are all up to parity; maybe v3 covers all fields we've
identified and can be run over existing data to enrich missing info." So v3 is one per-concept task that asks the ledger
what the concept already has and runs only the missing stages (idempotent; a field invented later = a new stage, and the
same task backfills it everywhere). New concepts later run the same task from scratch, so they reach parity by
construction. Everything writes to the ledger; every v2 efficiency carries over (LOOP_V2.md section 2, incl. the 18 Oct
5090 lessons).

## Parity today (ledger, 18 Oct; 5,028 mapped concepts)

| field | run 1 (4,194) | earlier maps (913; some overlap run 1) |
|---|---|---|
| neuron knockouts (all 200 candidates) | all | all have top 30; all 200 only fame3 onwards |
| random nulls | all (2 per concept) | only where also in run 1 |
| head knockouts | all | 548 |
| roles + pushed tokens | all | 79 |
| k curves | all | 548 |
| exact float64 log-odds | yes | not in pre-ISSUES #35 files |
| subject side (knowledge neurons at the name) | 67 (facts / era entities) | 46 |
| multiple answer formats | 56 (era entities) | 42 |
| links between key neurons | top 30: running (gpu-fan-out, links_run1) | top 5: 555 (indirect_path) |
| neuron-level: J-lens readout, job names, labels | global (all neurons; J-lens running; names for neurons used by >= 2 concepts) | same |

## The enrichment task `enrich:<concept>` (stages run only if missing)

| stage | what | GPU / concept | applies to |
|---|---|---|---|
| E1 v2 core | re-run map_core v2 from the concept's existing spec (exact log-odds, all 200, nulls, heads, roles, k curves, residuals) | ~70 s | earlier-only concepts (~780) |
| E2 formats | recall prompts in 2 extra answer formats: activation of all 200 candidates at the answer position (format-free vs format-bound) + top 30 zeroed in each format | ~10 s | all |
| E3 nulls | 6 layer-matched random neurons per concept (pooled per layer band in the ledger) | ~2 s | all |
| E4 subject pass | Luna: 6 questions ABOUT the concept; recall filter; name-token attribution (log-odds), top 100 zeroed at the name; 6 nulls | ~30 s + Luna | all (pilot on the 12 gate concepts running now) |
| E5 J-lens read | each concept's saved answer-position residuals read through the J-lens per layer ("what is held" by depth) | ~0 (saved arrays) | run 1 + E1 concepts |
| E6 links | top-30 links (LINKS.md rule) | (gpu-fan-out) | all |
Neuron-level passes (global, after the concept passes): J-lens readouts (done once), job names re-run for newly
eligible neurons, labels.

**Cost (estimate):** E2-E4 ~42 s x 5,028 concepts ~ 59 GPU-hours, + E1 ~15 GPU-hours -> ~75 GPU-hours ~ $16-20 on
rented RTX 3090s (+ boot overhead), or ~3 days on the 5090; Luna for E4 ~ 6 short questions per concept (12-concept pilot:
7.3k output tokens). Spend needs James's approval in the gpu-fan-out session's chat.

## Order
1. J-lens pilot (running) -> E5's exact form.
2. Ranking check (queued) -> keep or switch the v2 attribution objective (E1 / new concepts).
3. Subject-pass pilot on the 12 gate concepts (queued) -> E4 go / no-go.
4. Build `enrich` as a src/dist task kind reading the ledger for what is missing. **Skip anything we already have**
   (James, 18 Oct): no stage re-runs for a concept that has it, and within a stage existing measurements are reused (e.g.
   run 1's 200 candidates and recall prompts are inputs, never recomputed). Gate on the 12 gate concepts (v2 numbers
   reproduce exactly; new stages timed; peak memory ~25 GB; GPU busy %).
4b. **Smoke test, then re-assess** (James: "75 hours feels like a lot"): run enrich on a small sample (e.g. 50 concepts
   across run 1 and earlier maps) on the 5090; measure real per-stage time and value (how much each stage adds that the
   ledger lacked); bring James a revised cost and a stage-by-stage keep / cut / cheapen proposal before any scale-up.
5. Ledger importer for enrich records; rebuild / verify / export.
6. Scale-up: cost estimate to James -> run on rented GPUs via gpu-fan-out.

## Predictions to state before each stage
In each script's docstring before it runs (as for every experiment since 16 Oct).

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
| E0 missed candidates | knock out each concept's log-odds-only candidates (~86; ranking_check.py stage 2: median 4.5 per concept would enter run 1's top 30) - skip the 200 already measured | ~20 s | all |
| E1 v2 core | re-run map_core v2 from the concept's existing spec (exact log-odds, all 200, nulls, heads, roles, k curves, residuals) | ~70 s | earlier-only concepts (~780) |
| E2 formats | recall prompts in 2 extra answer formats: activation of all 200 candidates at the answer position (format-free vs format-bound) + top 30 zeroed in each format | ~10 s | all |
| E3 nulls | 6 layer-matched random neurons per concept (pooled per layer band in the ledger) | ~2 s | all |
| E4 subject pass | Luna: 6 questions ABOUT the concept; recall filter; name-token attribution (log-odds), top 100 zeroed at the name; 6 nulls | ~30 s + Luna | all (pilot on the 12 gate concepts running now) |
| E5 J-lens read | each concept's saved answer-position residuals read through the J-lens per layer ("what is held" by depth) - ON HOLD until the larger J-lens estimate improves early-layer reliability | ~0 (saved arrays) | run 1 + E1 concepts |
| E6 links | top-30 links (LINKS.md rule) | (gpu-fan-out) | all |
| E7 inhibitors (added 18 Oct, MECHANISMS.md H1) | run 1's saved attribution array -> the 30 most NEGATIVE-attribution neurons (run 1 only ever knocked out the top 200 positive), each zeroed alone; 4 layer-matched nulls; change in the strongest rival token | ~6 s (34 knockouts x 2 passes at ~0.09 s, from the smoke test's E0/E3 timings) | run 1 (arrays on disk, checked for all 4,194); earlier-only concepts recompute attribution (+ a few s) |
Neuron-level passes (global, after the concept passes): J-lens readouts (done once), job names re-run for newly
eligible neurons, labels.

**Cost (estimate):** E2-E4 ~42 s x 5,028 concepts ~ 59 GPU-hours, + E1 ~15 GPU-hours -> ~75 GPU-hours ~ $16-20 on
rented RTX 3090s (+ boot overhead), or ~3 days on the 5090; Luna for E4 ~ 6 short questions per concept (12-concept pilot:
7.3k output tokens). Spend needs James's approval in the gpu-fan-out session's chat.

## Order
1. J-lens pilot: DONE - useful early, noisy at 32 windows -> larger estimate via gpu-fan-out (with a cross-GPU repro check); E5 on hold.
2. Ranking check: DONE - the log-prob search misses strong neurons (median 4.5 per concept would enter the top 30) -> E0, and v3 searches the union of both top-200 lists.
3. Subject-pass pilot: DONE - S1-S3 hold (median layer 7.5, 0% overlap with answer-side maps, 10x null) -> E4 go.
4. Build `enrich` as a src/dist task kind reading the ledger for what is missing. **Skip anything we already have**
   (James, 18 Oct): no stage re-runs for a concept that has it, and within a stage existing measurements are reused (e.g.
   run 1's 200 candidates and recall prompts are inputs, never recomputed). Gate on the 12 gate concepts (v2 numbers
   reproduce exactly; new stages timed; peak memory ~25 GB; GPU busy %).
4b. **Smoke test, then re-assess** (James: "75 hours feels like a lot"): run enrich on a small sample (e.g. 50 concepts
   across run 1 and earlier maps) on the 5090; measure real per-stage time and value (how much each stage adds that the
   ledger lacked); bring James a revised cost and a stage-by-stage keep / cut / cheapen proposal before any scale-up.
5. Ledger importer for enrich records; rebuild / verify / export.
6. Scale-up: cost estimate to James -> run on rented GPUs via gpu-fan-out.
7. **E7 inhibitors pilot (built 18 Oct, NOT run; needs a free GPU and James's go-ahead).** Code: `enrich.py` stage E7
   (opt-in), importer rows in `src/ledger/importers/loop_v3.py` (experiment `loop_v3_e7_pilot`, `knockout_drop` with
   `context=stage=E7;selection=bottom_attr;...`, negative = the answer went UP). Checks done on the CPU: (a) run 1's arrays
   load for all 4,194 concepts, and each array's top 200 reproduces that concept's 200 candidates (exact in 3,910; the
   other 284 differ by 1-4 float16 ties at rank 201-208), so the right file is read; the bottom 30 are always negative and
   never among the candidates; (b) the whole stage ran on Qwen3-0.6B for 2 concepts (attribution recomputed, since the
   arrays are 8B-shaped): 59 of 60 bottom neurons raise the answer beyond 0.10, all 8 nulls stay below it (median 0.003),
   56 of 59 lower the rival - consistent with MECHANISMS.md; (c) the refactored importer reproduces all 40,384 existing
   loop_v3_enrich rows. Commands (on the 5090 once the current enrich run has finished, or a rented 24 GB+ card):
   `ENRICH_NAME=e7_pilot ENRICH_ALL=1 ENRICH_SAMPLE_N=500 python src/loop_v2/enrich.py sample` then
   `ENRICH_NAME=e7_pilot ENRICH_STAGES=E7 python src/loop_v2/enrich.py run` then `... summary` (prints I1-I5).
   **Cost:** ~6 s per concept -> ~50 min on the 5090 for 500 concepts (a rented RTX 3090 is ~1.5-2x slower: ~1.5 h, roughly $0.3-0.5 incl. boot);
   the rest of run 1 (3,694) ~6 h; all 5,028 ~8-9 h. Halve it by dropping the rival readout (one extra pass per
   knockout) if the pilot shows it adds little. Predictions I1-I5 are in enrich.py's docstring.

## Predictions to state before each stage
In each script's docstring before it runs (as for every experiment since 16 Oct).

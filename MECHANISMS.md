# New kinds of neuron job: hypotheses and cheap tests (18 Oct 2026)

James (18 Oct): "have a session hypothesise mechanisms to identify new categories of jobs. Ideally testable/derived from
existing data and checkable cheaply and quickly from CPU on 0.6b. Idea is to find new mechanisms to loop into later loops
that we can scale out later on the GPUs."

**What this is.** A ranked list of job categories we have NOT yet looked for, each with the data that hints at it, a
cheap test (with a control and a pass rule), a prediction written down before the test runs, its CPU cost on
Qwen3-0.6B, and how it would enter the enrichment loop (LOOP_V3.md, `src/loop_v2/enrich.py`) at Qwen3-8B scale.
Results on 0.6B are cheap evidence that a mechanism exists in this model family; they are not findings about 8B.
Names like "inhibitor" are informal readings, not labels.

**Order of work.** This file was committed with predictions only (section 1) before any test ran; results were added
afterwards (section 2) without editing the predictions.

Known job categories so far (for contrast): 'say' / 'attribute' / 'late-other' / 'early' roles; entity neurons at a name
(median layer 8 in 8B); relation neurons; answer-value neurons; era-concept vs format-bound digit neurons; two-ended
scales (north <-> south); shared answer-format machinery (layers 16-24); shared neurons and their feeders; links between
key neurons; brake / accelerator HEADS at layer 28 and one suppressor head (L35H27).

A gap that shapes the list: every 8B concept map (run 1, 4,194 concepts) knocked out the top 200 neurons by POSITIVE
attribution (`concept_pipeline.attribute`, `top = argsort(descending)[:200]`). Neurons whose attribution is negative,
i.e. that push AGAINST the answer, were never knocked out. So anything that holds an answer back is unmapped in 8B, and
run 1 already saved the full attribution arrays (`arrays/*.npz`, key `attr`, 36 x 12,288) that would find them for free.

## 1. Hypotheses (ranked by value x cheapness)

Costs: "CPU" = this PC, Qwen3-0.6B in float32. "8B GPU" = rough RTX 5090 time; per concept unless marked global.

### H1. Inhibitor neurons: neurons whose knockout RAISES the answer
- **Job:** hold an answer back - dampen confidence, or push a rival. Unmapped (see the gap above).
- **Hints in existing data:** head-level brakes exist (L28H7, L25H14; L35H27 suppresses rivals; RESULTS.md 16 Oct
  attention study). Among run 1's positive-attribution candidates some knockouts already go negative (5th percentile
  of 1,048,500 drops is +0.002, minimum -2.95; null knockouts' 0.1st percentile is -0.145):
  `SELECT count(*) FROM evidence WHERE experiment_id='loop_v2_run1' AND evidence_type='knockout_drop' AND value < -0.5`.
- **Test (0.6B, `src/mechanisms/inhibitors.py`):** reuse run 1's recall prompts and answers; keep concepts 0.6B
  recalls. Per concept, attribution (activation x gradient of the answer's log-prob) on half the prompts; on the OTHER
  half, knock out alone: the 20 most negative-attribution neurons ("bottom"), the 20 most positive ("top"), and 20
  random neurons (null). Score = change in mean exact log-odds of the answer (positive = answer went up).
- **Pass rule:** an inhibitor is a bottom neuron whose knockout raises the held-out answer by more than the 99th
  percentile of the random neurons' rise (pooled over concepts). H1 holds if >= 30% of concepts have an inhibitor.
- **Predictions:** (P1a) >= 30% of concepts have >= 1 inhibitor; (P1b) inhibitors sit mostly in the last quarter of
  layers (>= 50% in layers 21-27 of 28); (P1c) some inhibitors are shared: >= 3 neurons are inhibitors for >= 10% of
  concepts (generic dampeners), the rest concept-specific; (P1d) the median rise of a concept's best inhibitor is
  >= 0.3 log-odds.
- **CPU cost:** ~10-20 min for ~40 concepts.
- **Into the loop:** new enrich stage **E7 inhibitors**: read run 1's saved `attr` array (no recompute), knock out the
  bottom 30 alone. New evidence_type `knockout_raise` (or `knockout_drop` with `context=selection=bottom_attr`, negative
  values meaning a raise). **8B GPU:** ~8-10 s per concept -> ~12-14 GPU-hours for 5,028 concepts.

### H2. Confidence (entropy) neurons: late neurons that change how sure the model is, not what it says
- **Job:** write into directions the output vocabulary ignores (the unembedding's near-null space); the effect arrives
  through the final normalisation, scaling every logit at once. Known in GPT-2 / Pythia / Llama (Stolfo et al. 2024,
  "Confidence regulation neurons"); never looked for here.
- **Hints:** the layer-28 'gain dial' reading was tested on heads and rejected (RESULTS.md 16 Oct); nobody tested
  neurons. 'Late-other' role neurons (62,669 rows: late, but do not raise the answer) are where such neurons would hide:
  `SELECT count(DISTINCT unit_id) FROM evidence WHERE evidence_type='role' AND value_text='late-other'`.
- **Test (0.6B, weights then causal, `src/mechanisms/entropy_neurons.py`):** for every neuron in the last 3 layers,
  null fraction = share of its output vector (times the final norm's weights) lying in the bottom 2% of the unembedding's
  singular directions, and its norm. Candidates = top 10 by norm x null fraction. Causal: on 64 web-text windows of 128
  tokens, set the neuron to its mean activation (mean ablation) and measure the change in output entropy and in loss,
  total vs "direct" (with the final norm's scale frozen at its clean value). Null: 30 random last-3-layer neurons.
- **Pass rule:** a confidence neuron changes mean |entropy| by > 5x the random 95th percentile AND its total effect on
  entropy is >= 2x its direct effect (it works through the normalisation).
- **Predictions:** (P2a) >= 1 candidate in 0.6B's last layer passes; (P2b) for passing neurons, the null fraction is
  >= 10x the median neuron's; (P2c) mean-ablating them changes entropy much more than it changes the top token (top-1
  agreement with the clean run >= 95%).
- **CPU cost:** weights ~1 min; causal ~5 min.
- **Into the loop:** a **global neuron pass** (once per model): weights-only null-fraction score for all 8B neurons
  (CPU, minutes), causal check of the top ~50 (~0.5 GPU-hour). Per concept: free - read their activation at the answer
  position from E2's saved activations. New evidence_type `confidence_effect` (metric `entropy_change_total_vs_direct`).

### H3. Repeat / induction neurons: neurons that fire when the text repeats, and help copy
- **Job:** detect "this token sequence has occurred before" and support predicting what came next last time.
  Copying is a core job for names in context (and for the turn-boundary copier heads L34H28 / L35H22).
- **Hints:** cross-position path: ~1/5 of the assembly moves between words through a few heads (S3); copier heads at
  the turn boundary. No neuron-level test exists.
- **Test (0.6B, `src/mechanisms/repeat_neurons.py`):** 60 sequences of 40 random common tokens repeated twice, vs
  position-matched control sequences of 80 random tokens with no repeat. Per neuron, activation in the second copy
  minus the control at the same positions (effect size d). Select on seeds 0-29, confirm on 30-59. Causal: zero the top
  20 confirmed repeat neurons in the second copy and measure loss on the second copy; null = 20 random neurons from the
  same layers.
- **Pass rule:** >= 20 neurons with held-out d >= 2; zeroing the top 20 raises second-copy loss by >= 3x the random
  set's rise.
- **Predictions:** (P3a) repeat neurons exist (>= 20 at d >= 2 held-out); (P3b) they sit mid-network (median layer
  between 8 and 20 of 28); (P3c) zeroing the top 20 raises second-copy loss >= 3x random and >= 0.05 nats/token.
- **CPU cost:** ~3 min.
- **Into the loop:** global neuron pass (~1 GPU-hour on 8B); per concept, a context-copy stage (the concept's name
  given earlier in the prompt, asked for again) would test whether recall uses memory or copying. New evidence_type
  `repeat_selectivity`.

### H4. Language-identity neurons: neurons that track which language is being written
- **Job:** keep the reply in the right language; content-free. Era neurons already write cross-lingual history words
  (M2c), so content and language separate somewhere.
- **Hints:** J-lens / logit-lens readouts mix scripts (L28:8912: 'Bolshevik', 'Lenin', 民国); 442,368 readouts in
  `jlens_qwen3_8b` could be scanned for single-script neurons.
- **Test (0.6B, `src/mechanisms/language_neurons.py`):** 40 parallel sentences (same meaning) in English, French,
  German, Spanish and Chinese. Per neuron and language, AUC of mean activation (over tokens) for that language vs the
  rest; select on sentences 0-19, confirm on 20-39. Causal: zero a language's top 20 confirmed neurons and measure loss
  per token on held-out text in each language; null = 20 random same-layer neurons.
- **Pass rule:** for >= 3 of 5 languages, zeroing its neurons raises loss on its own language >= 3x more than on the
  other languages and >= 3x the random set.
- **Predictions:** (P4a) every language has >= 10 neurons with held-out AUC >= 0.95; (P4b) they concentrate in early
  layers (median layer <= 10) because language is visible in the tokens; (P4c) the causal rule passes for >= 3 of 5.
- **CPU cost:** ~3 min.
- **Into the loop:** global pass (~0.5 GPU-hour on 8B); per concept, asking each recall prompt in a second language
  (one extra format in E2) separates concept neurons from English-answer neurons: ~10 s per concept -> ~14 GPU-hours.
  New evidence_types `language_auc`, `language_necessity`.

### H5. Position and attention-sink neurons
- **Job:** (a) position neurons encode how far into the text we are; (b) massive-activation neurons fire hugely on the
  first token and make it an "attention sink" the heads park on (Sun et al. 2024).
- **Hints:** activations in `era_formats` reach -153 / +237 (typical late neurons peak at ~20-30):
  `SELECT unit_id, max(abs(value)) FROM evidence WHERE experiment_id='era_formats' AND metric='activation' GROUP BY 1 ORDER BY 2 DESC LIMIT 5`.
  Always-on shared neurons (L20:7473) might be sink machinery rather than answer-format machinery.
- **Test (0.6B, `src/mechanisms/position_sink.py`):** 64 web-text windows of 256 tokens. (a) For each neuron, share
  of activation variance explained by position (positions 1-255, 16 bins) on windows 0-31; confirm on 32-63 (profile
  correlation). (b) Massive neurons: |activation| at position 0 > 20x the layer's 99.9th percentile elsewhere; causal:
  zero them at position 0 only, measure loss over positions 1-255 vs 20 random neurons zeroed at position 0.
- **Pass rule:** (a) >= 10 neurons with variance share >= 0.3 and held-out profile r >= 0.9; (b) zeroing the massive
  neurons at one position raises loss >= 10x the random set.
- **Predictions:** (P5a) position neurons exist, mostly in layers 0-5; (P5b) 1-10 massive neurons, in early-mid layers,
  and zeroing them at position 0 alone damages the whole text (loss up >= 1 nat/token).
- **CPU cost:** ~5 min.
- **Into the loop:** global neuron pass (~0.5 GPU-hour); per concept: tag shared 'always-on' neurons that are sink or
  position neurons so they are not read as concept machinery (free, from the global pass). New evidence_types
  `position_share`, `massive_activation`.

### H6. Contrastive 'say' neurons: push the answer AND pull down its nearest rivals
- **Job:** discriminate between close alternatives (Paris vs Lyon) rather than just raise one word.
- **Hints:** run 1's `direct_logit_top30` readouts list only the tokens pushed UP (125,820 rows); the push-down side was
  never read. L35H27 suppresses rivals at head level.
- **Test (weights only; 0.6B and 8B):** for each 'say' neuron, direct push on every token (output vector x final norm x
  unembedding). Take its top token t and t's 20 nearest tokens by unembedding cosine (excluding case/space variants of
  t). Score = mean z of those neighbours in the neuron's push. Null: the same procedure on random late neurons (their own
  top token and its neighbours).
- **Pass rule:** a contrastive neuron has neighbour z < 0 (rivals pushed below the vocabulary mean) while t is in its
  top 5. H6 holds if say neurons are contrastive >= 3x as often as random late neurons.
- **Predictions:** (P6a) most say neurons push neighbours UP (semantic smoothness): median z > 0; (P6b) a minority
  (>= 10%) is contrastive, >= 3x the random rate.
- **CPU cost:** minutes (8B: 1.2 GB unembedding + selected rows of down_proj; no forward pass).
- **Into the loop:** free per concept from weights; new evidence_type `rival_push`. Global, CPU.

### H7. Negation neurons
- **Job:** flip a statement ('is not'). **Hints:** none in our data (all prompts are positive questions) - a blind spot.
- **Test:** minimal pairs "X is in Y" / "X is not in Y" / "X is also in Y" (control word); neurons with held-out AUC
  >= 0.9 for 'not' vs 'also' at the tokens after it; causal: zero them and ask a negated question ("Which country is
  Paris NOT in: France or Spain?") - answer flips towards the un-negated reading vs random. **Prediction:** detectable
  in mid layers; causal flip weak (negation is mostly attention-mediated). **CPU:** ~5 min. **Loop:** per concept, one
  negated recall prompt in E2 (~2 s) to see whether concept neurons respect negation.

### H8. Question-type neurons (relation detected) vs answer neurons
- **Job:** recognise WHAT is asked (a year, a place, a person) at the question words, before any answer forms.
- **Hints:** relation neurons act at the answer position (M2, 255 neurons); category-reader head L26H17. Whether they
  also fire when the model does not know the answer is untested.
- **Test:** facts.py's 5 relations x entities the model does NOT know (made-up names) vs known: a question-type neuron
  fires for its relation equally for real and made-up entities; an answer neuron only for real ones. **Prediction:**
  most relation neurons in layers 20-27 are question-type (fire for made-up entities, AUC >= 0.9). **CPU:** ~5 min.
  **Loop:** per concept, 2 made-up-entity prompts per relation (~2 s).

### H9. Answer-format neurons (format-bound machinery)
- **Job:** produce the answer in the asked form (just the name / a sentence / a year). M3's always-on neurons
  (layers 16-24) are defined by sharing only.
- **Hints:** E2 of the enrich smoke test already saved activations of each concept's 200 candidates in 3 formats
  (`loop_v3_enrich`, `activation_answer_position`, 31,635 rows).
- **Test (ledger only):** neurons whose activation ratio F1/F3 is far from 1 consistently across concepts (format-
  bound for all) vs concept-specific. **Prediction:** the shared always-on neurons are the format-bound ones.
  **CPU:** seconds. **Loop:** already in E2 - add a per-neuron summary view.

### H10. Ordinal / counting neurons
- **Job:** track item number in a list ("first, second, ..."). **Test:** comma-separated lists of 2-12 items; neurons
  whose activation at each comma tracks the item index (Spearman >= 0.8 held-out) and not raw position (control: same
  lists with long items). **Prediction:** a few, mid layers. **CPU:** ~3 min. **Loop:** global pass.

### H11. Antonym scales beyond compass directions
- **Job:** one direction, two opposite meanings (hot <-> cold, big <-> small, early <-> late).
- **Hints:** scales study: 1,567 neurons matter on both signs; 3 strict scales causal at k = 1 (`scales_steer`).
- **Test:** 20 antonym pairs in matched sentences; neurons whose activation sign separates the two ends (AUC >= 0.9 and
  opposite signs) across >= 3 pairs. **Prediction:** a few multi-pair 'polarity' neurons (good/bad-like) exist.
  **CPU:** ~3 min. **Loop:** global pass + per concept for two-ended concepts.

### H12. Attention-gating neurons
- **Job:** decide what later heads look at (a neuron whose output changes a head's query or key).
- **Hints:** feeders and links data (loop_v2_feeders, links_run1) are neuron -> neuron; head -> neuron routes exist
  (brake L28H7 -> entry neuron). Neuron -> head is untested.
- **Test:** weights-only composition score of each neuron's output with each later head's query/key weights vs random;
  then causal on the top pairs (attention pattern change). **Prediction:** a few neurons strongly gate the copier
  heads. **CPU:** ~10 min. **Loop:** per concept, a stage measuring key heads' attention with the top-5 neurons zeroed
  (~3 s).

### H13. Word-stitching (detokenisation) neurons
- **Job:** recognise a multi-token word as one unit at its last piece; the first step towards early entity neurons
  (which act at the name tokens, S4). **Test:** multi-token vs single-token words; neurons firing at the last piece of
  specific multi-token words only. **Prediction:** layers 0-3, highly specific. **CPU:** ~5 min. **Loop:** subject
  pass (E4) could record whether a concept's entity neurons are stitching neurons for its name (free).

## 2. Results (filled in after the tests ran, 18 Oct evening)

Seven tests ran: five on Qwen3-0.6B (CPU) and two on Qwen3-8B data already on disk (no forward pass). All outputs are in
`D:/already-clean-data/mechanisms/` with every per-item number. The predictions above were not edited.
Two of the runs are **exploratory**, i.e. designed after seeing a pre-registered result, and are labelled as such.

| # | category | verdict | headline number | scale to 8B? |
|---|---|---|---|---|
| H1 | inhibitor neurons (knockout raises the answer) | **held** (P1a, c, d; P1b failed); **replicated** | every one of 40 concepts has them; median best +0.73 log-odds held-out (re-test +0.65 on 8 new concepts) | **yes - first** |
| H1-8B | sign flips among run 1's helpers | partly held | 0.36% of 1.05M knockouts; 40% of concepts; L31:5473 in 174 | (comes with H1) |
| H4 | language-identity neurons | **held** (P4a, c; P4b failed) | zeroing 20 neurons: own-language loss +0.28 to +0.71 nats, others ~0, for fr/de/es/zh | **yes** |
| H3 | repeat / induction neurons | mixed (P3a, b held; P3c failed) | 3,688 neurons respond to repetition; zeroing 20 (or 200) does not specifically hurt copying | no (descriptive tag at most) |
| H2 | confidence (entropy) neurons | **failed** as pre-registered; 1 neuron in an exploratory re-run | v1: nothing above 0.02 nats; v2: L27:47 (17x random) | low priority |
| H9 | shared neurons are format-bound | **failed** | 35% of neurons are format-bound, but shared ones no more than private ones | no (but E2 data is useful) |
| H6 | contrastive 'say' neurons | **failed** as tested | say neurons push near-spellings UP; 3% contrastive vs 3.5% random | re-design first |

### H1. Inhibitor neurons - HELD (the strongest new category)
`src/mechanisms/inhibitors.py`; `D:/.../mechanisms/inhibitors/`. 40 run-1 concepts within 0.6B's reach (1,559 screened;
0.6B has very few of the 8B concepts in reach). Neurons chosen on half the prompts, measured on the other half.
- **P1a held, far beyond the bar:** 40 of 40 concepts (100%) have at least one inhibitor; 645 of the 800 most negative-
  attribution neurons raise the held-out answer by more than the null's 99th percentile (+0.049 log-odds; 1,600 random
  and layer-matched neurons). Control on the counting: random neurons cross the same bar in 17.5% of concepts, as expected
  by chance for 20 draws at the 99th percentile.
- **P1d held:** the median concept's best inhibitor raises the answer by **+0.73** log-odds (its best helper costs -1.58).
  Median bottom neuron +0.21, median random neuron 0.00.
- **P1b failed:** inhibitors are spread over all layers (41% in layers 21-27, not >= 50%); layer 27 has the most (71),
  then 19 (54), 26 (46), 17 (44), and a cluster at layers 0-1 (56).
- **P1c held:** some are generic dampeners: L17:1763 inhibits in 23 of 40 concepts, L11:1708 in 22, L14:941 in 20, L19:96
  in 16, L3:733 in 15.
- **Re-test inside the run:** the sign of a bottom neuron's effect agrees between the selection half and the held-out half
  in 92% of cases.
- **How they act (informal):** removing an inhibitor lowers the strongest rival token in 91% of cases (median -0.14 log
  prob), and changes the entropy little (-0.06 nats). The rivals are of two kinds: *format rivals* - starting a sentence
  instead of giving the bare name ('The', 'A', 'In'; e.g. L27:2498 for 'Fashion' +2.70, L18:102 for 'Baku' +1.95) or
  lower-case 'gender' for 'Gender' (L19:96, +2.03) - and *content rivals* ('Reflection' for 'Refraction', L26:1559 +1.32;
  'secondary' for 'University'). So "inhibitor" covers 'hold the answer back in favour of another form or another word'.
- **Caveats:** 0.6B, run 1's prompts (written for 8B); the held-out half is small (median 4 prompts); answers 0.6B only
  half-knows (the screen accepted P > 0.12), where raising is easiest; zero-ablation only.
- **Re-test (seed 1, `inhibitors_seed1/`): replicated.** 20 concepts, of which only 8 are new (the seed reshuffles the
  same small pool of concepts 0.6B can answer, so 12 repeat run 1 - an honest limit of this re-test). All 20 have
  inhibitors (median best +0.75; null 99th percentile +0.048; random neurons cross it in 25% of concepts). On the **8 new
  concepts alone** (Arabic, Chi-squared test, Mahabharata, Autumn, National park, Great Depression, Jordan River,
  Physical cosmology): 8 of 8 have inhibitors, 133 of 160 bottom neurons pass, median best +0.65. The same shared
  dampeners return: L11:1708 (12 of 20), L3:733 (11), L17:1763 (10), L19:96 and L14:941 (7 each).

**H1 on 8B from the ledger (free; `src/mechanisms/ledger_checks.py`).** Run 1 knocked out only positive-attribution
candidates, yet 3,770 of 1,048,500 knockouts (0.36%) raise the answer beyond run 1's own null (0.1st percentile, -0.145),
in 1,666 of 4,194 concepts (40%); 312 (0.03%, 5.7% of concepts) raise it by > 0.5. Spread over all layers (and heads);
not concentrated in 28-35 (25% there), so that part of the prediction failed. The standout is **L31:5473**: a widely used
'late-other' neuron whose direct push is on function words ('–', 'to', 'in', 'as', 'at', 'of', ','); it helps in most of
its 1,594 maps (mean drop +0.32) but RAISES the answer when removed in 174 concepts. It reads as a 'keep writing the
phrase' vs 'give the bare name' arbiter - the same format-rival pattern as in 0.6B. The real inhibitors (negative
attribution) were never measured on 8B: that is the gap the E7 stage would fill.

### H4. Language-identity neurons - HELD (late, not early)
`src/mechanisms/language_neurons.py` + `parallel_sentences.json`; `D:/.../mechanisms/language/`.
- **P4a held:** neurons with held-out AUC >= 0.95 for one language on content-matched sentences: en 71, fr 37, de 146,
  es 44, zh 275.
- **P4b failed - the interesting part:** they are LATE, not early: median layer en 21, fr 25, de 24, es 25, zh 21 (of
  28), with only a few in layers 0-3. Language is mostly held where the next word is written, not where tokens are read.
- **P4c held (4 of 5):** zeroing a language's top 20 raises loss on that language and almost nothing else - fr +0.45
  nats/token (other languages <= 0.005; random set 0.005), de +0.71 (others <= 0.08; random 0.05), es +0.28 (others <=
  0.02; random 0.01), zh +0.08 (others negative; random 0.00). **English fails:** zeroing its 20 neurons changes nothing
  (-0.001). Reading: English is the default; the other languages need neurons to stay switched on.
- **Caveats:** 40 sentences per language written by Claude; zh's effect is small in nats (Chinese loss is high to start
  with, 5.35); one neuron set per language.

### H3. Repeat / induction neurons - MIXED
`src/mechanisms/repeat_neurons.py`; `D:/.../mechanisms/repeat/`. Sanity check passed: first halves identical (max
difference 0.0).
- **P3a held (too easily):** 3,688 neurons fire more on the repeat (held-out d >= 2) and 3,844 fire less - about 9% of all
  neurons. Repetition changes the model's state broadly, so "responds to repetition" is not a sparse category at d >= 2.
- **P3b held:** median layer 15 of 28 (peak at layers 13-17). The cleanest are near-silent on new text and strongly on
  for repeats, e.g. L23:45 (14.5 vs -0.4), L22:20 (13.4 vs 0.1), L12:796 (5.4 vs 0.1).
- **P3c failed:** zeroing the top 20 raises second-copy loss by only 0.015 nats/token (random -0.0004; clean loss 0.21);
  on the control it does nothing (0.003). Specific, but far below the 0.05 bar: copying survives losing 20 of them.
- **Exploratory follow-up (K=200, `REPEAT_K=200`, `repeat_k200/`):** zeroing the top 200 raises second-copy loss by
  0.099 nats (random 200 from the same layers: 0.011) - but it raises loss on the NO-repeat control just as much (+0.116;
  random -0.011). So the damage is general, not to copying. Reading: these neurons *register* repetition, but copying
  does not depend on them (in transformers it is mostly done by attention 'induction' heads). Not a causal job category.

### H2. Confidence (entropy) neurons - FAILED as pre-registered
`src/mechanisms/entropy_neurons.py`; `D:/.../mechanisms/entropy/`, `entropy_v2/`.
- **v1 (pre-registered): failed.** No candidate passed; no neuron of 50 changed entropy by more than 0.019 nats. The
  weights rule picked neurons that hardly ever fire (activation sd 0.0-0.6). A structural reason: 0.6B's unembedding has
  no real null space (smallest singular value 3.6 vs median 10.3; its embeddings are tied), unlike the GPT-2 / Pythia
  models where these neurons were found.
- **v2 (EXPLORATORY, rule changed after v1):** ranking by activation sd x |w| x null fraction, one neuron passes:
  **L27:47** - entropy change 0.104 nats (17x the random 95th percentile), 2.4x its direct effect (it works through the
  final normalisation), top word unchanged 98% of the time, null fraction 13x the median. L27:81 is close (0.118, but
  only 1.7x direct). Both are huge-variance neurons (sd 60-72). Several v2 'candidates' only look active because they fire
  at positions we did not score; mean ablation then shifts every position (an artefact to remember).
- Verdict: weak evidence for one or two confidence neurons in 0.6B. Worth one cheap weights-only scan on 8B (untied
  unembedding, so a real null space may exist), not a per-concept stage.

### H9. Answer-format neurons - FAILED (shared neurons are not more format-bound)
`src/mechanisms/format_bound.py` (ledger only, 8B, E2 smoke-test data, 50 concepts). Format matters a lot: the median
activation shift between 'just the name' and 'one short sentence' is 0.59 (normalised), vs 0.06 between two short-answer
wordings. 35% of the 137 neurons seen in >= 5 concepts are format-bound (vs 0-2% under the control comparison). But
widely shared neurons (n_users >= 50) are format-bound no more often (28%) than rarely shared ones (35%): the prediction
failed. L20:7473 (used by 874 concepts) IS format-bound - it fires more when a sentence is asked for.

### H6. Contrastive 'say' neurons - FAILED as tested
`src/mechanisms/rival_push.py` (8B weights; 400 of 3,826 say neurons, 400 random late neurons). Say neurons push their
top word's nearest tokens UP (median z +2.6; random late neurons +1.7); 3.0% contrastive vs 3.5% for random. **Caveat
that limits the verdict:** the nearest tokens by unembedding cosine are mostly other spellings of the same word ('log',
'.log', '_log'), not real competitors. The test that would settle it uses each concept's actual rival answer (run 1's
second-best token at the answer position) instead.

### H5 hint from the 8B ledger (massive activations) - inconclusive
In the ledger, the largest answer-position activations are L35:9510 (257), L35:11333 (234), L35:9490 (198) vs a median
unit max of 2.5; but these are answer-position values, not first-token values, so they say nothing about attention
sinks. H5 needs its own test.

### Follow-ups (both done)
- H1 re-test (seed 1): done (above) - replicated, including on the 8 concepts not in run 1.
- H3 with 200 neurons zeroed: done (above) - general damage, not copying-specific.

## 3. What to scale to Qwen3-8B on GPUs, in order

1. **E7 inhibitors (H1).** Run 1 already saved each concept's full attribution array, so the bottom 30 cost no new
   attribution: knock each out alone on the recall prompts (same procedure as the 200 candidates), plus run 1's nulls.
   Predicted from 0.6B: every concept has some, a handful are shared dampeners, and rivals split into format and content.
   **Built 18 Oct** (enrich.py stage E7, opt-in; importer rows; CPU checks) - see LOOP_V3.md 'Order' item 7. Measured cost
   basis: ~6 s per concept on the 5090 -> ~50 min for a 500-concept pilot, ~8-9 GPU-hours for all 5,028. Not run yet.
   Ledger rows: `knockout_drop` with `context=stage=E7;selection=bottom_attr;...` (negative = raise).
2. **Language neurons (H4)**, a global pass: content-matched sentences in 5+ languages, AUC + group knockouts, ~0.5
   GPU-hour. Then per concept, one recall prompt in a second language in E2 (~10 s per concept) would split concept
   neurons from English-answer neurons.
3. **Repeat neurons (H3):** not as a causal stage (K=200 showed no copying-specific need); at most a cheap global tag.
4. **Confidence neurons (H2)**: a weights-only 8B scan on the CPU first (minutes), then a causal check of ~20.
5. **H6** re-designed with real rival answers before any scaling.
Not run (still open): H5 position / sink neurons, H7 negation, H8 question-type vs answer neurons, H10 ordinal,
H11 antonym scales, H12 attention gating, H13 word stitching.

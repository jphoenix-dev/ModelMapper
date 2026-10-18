# Results log

Every run gets logged here, failures included. Newest at the bottom.

---

## 15 Oct 2026: setup and pipeline debugging (Qwen3-0.6B; not evidence)

**Setup**
- Environment: `.venv`, PyTorch 2.11 (cu128), transformers 5.20, RTX 5090.
- Hook check: SiLU(gate) × value reproduces the input to down_proj to within 0.17% of max activation, which is
  bfloat16 rounding. So we're reading the right number.
- `huggingface_hub` downloads stalled on this machine (a known local issue), so models are now
  fetched with resumable curl (`scripts/download_models.sh`) into a local models folder (`$MODEL_MAPPER_MODELS`).
- Jev client (`src/jev.py`): smoke tests passed and cost about $0.0002.

**Contrastive search, first attempt.** The score saturated at about 2.3 for every top neuron. Measuring the gap in
units of the spread across *all* texts has a ceiling when group sizes are fixed. Fixed by measuring the gap in units
of the *control* texts' spread.

**Contrastive search, 0.6B, second attempt** (`results/gg_candidates_Qwen3-0.6B.json`)
- Early layers (2–5): strongly selective (score 7–12) and fire on ~96–98% of Golden Gate texts above every control.
  But they do **not** fire on texts that describe the bridge without naming it ("unnamed" ≈ 0). So these detect the
  word pair "Golden Gate", not the concept, which is what the essay's spelling-fragment → name-recognition story would
  predict for early layers.
- Middle and late layers (17–18): a few neurons fire as much or more on *unnamed* descriptions (unnamed ≈ 1.3–1.6),
  i.e. concept-like. But they're much less selective (they beat every control on only 10–12% of texts).
- New check added that wasn't in the plan: "gg_unnamed" texts, to tell word detectors from concept detectors.

**Steering, 0.6B:** clamping L17:845 and L18:1470 to 10× and 30× their typical Golden Gate level changed the replies
slightly but produced no Golden Gate mentions (0/4 prompts). This is a debug run on a tiny model with weak candidates,
so not evidence either way.

---

## 15 Oct 2026 (evening): Experiment GG on Qwen3-8B, steps 0–2

**Neuron ranges:** max and min recorded for all 442,368 neurons over 2.0M tokens of FineWeb in 399 s
(`data/stats_Qwen3-8B.pt`).

**Contrastive search, pre-registered ranking** (`results/gg_candidates_Qwen3-8B.json`)
- Many neurons are strongly selective for sentences that *name* the bridge. Example: L7:4091, score 14.6,
  reaches 71% of its own peak, and beats every control sentence on 74% of Golden Gate sentences.
  Selective units exist in early, middle and late layers, on both poles.
- Almost all have "unnamed" ≈ 0: they don't fire on descriptions of the bridge that leave out the name.
  The best is ~0.2 (L12:3282, L13:2962, L31:7303).

**Recognition check:** Qwen3-8B names the Golden Gate Bridge for 14/15 unnamed descriptions (the miss answered
"Bay Bridge"). So the model *does* recognise the descriptions. Recognising the name and recognising a description run
through different neurons.

**Exploratory (not pre-registered): neurons selective for unnamed descriptions**
- L26:1311 (unnamed 9.4 vs named 1.7 vs controls 0.9) and L25:6245 (12.7 vs 2.6 vs 1.0) fire on descriptions but
  only weakly once the name is present.
  Hypothesis (untested): late-layer "about to say Golden Gate" neurons that are quiet once the name has been said.
  Prediction for the scan: their top web firings sit just *before* the words "Golden Gate" appear.
- L3:4061 (negative pole, score 104) is early-layer and probably a word detector for a word over-used in the
  unnamed set (e.g. "orange"). It's a flaw in the hand-written prompts, not a finding.

**Scan setup:** 62 units (32 candidates incl. 8 exploratory, 20 random neurons from the same layers, 10 planted
mixtures), 10M tokens, 40 top examples per pole (`results/units_Qwen3-8B.json` → `results/scan_Qwen3-8B.json`).

**Scan restart:** the first 8B scan slowed from ~4,850 to ~1,500 tokens/s. The GPU showed 100% busy at only 128 W and
memory was full (32.0/32.6 GB): the Windows driver was spilling into system RAM. It was also the moment C: ran out of
space (255 MB free; ~26 GB cleared). Restarted with 6k-token batches and a 30 GB PyTorch cap: 403 W, 23 GB, ~5,000 tok/s.

**Labeller test on 0.6B debug units (Luna + Jev):**
- Luna correctly described the comma neurons ("comma after a fronted introductory phrase").
- A random direction got a vague explanation that covered 17/20 of its examples, and the same explanation fitted
  16/20 random tokens. Without a specificity check it would have counted as "clean". Added the random-token check.
- Jev's blind verdict said ONE for every pole, so it isn't used.
- Luna's verdicts vary a little between runs (NONE vs MIXTURE on the same unit).
- Cost for 5 units: Jev ~$0.012; Luna ~41k input + 12.6k output tokens.

---

## 15 Oct 2026 (late): Experiment GG, scan + blind labelling on Qwen3-8B

**Scan:** 10.0M tokens of FineWeb, 62 units, 40.3 min (`results/scan_Qwen3-8B.json`).

**Eyeballed first, then confirmed blind:**
- The exploratory "unnamed" neurons are about the **word "orange"**, not the bridge. L26:1311 fires on "Orange Is the
  New Black", "Orange Range"; L25:6245 on "Orange", "Tangerine", "lemon"; L3:4061 (negative pole) on "ORANGE Beach",
  "City of Orange". They fired on my unnamed descriptions because those said "International Orange" and "orange
  towers". The "about to say Golden Gate" hypothesis is **refuted**; the prompt set was confounded.
- The top pre-registered candidate L7:4091 fires in the wild on name and number completions ("Rosenzwe|ig",
  "Paul Newton", ISBN digits). L14:7392 fires on the second half of fixed terms ("pull-down", "strike price",
  "World Heritage"). Coherent jobs, but not Golden Gate-specific.
- **"Golden Gate" appears only 5 times in the 10M scanned tokens** (~1 per 2M). A random 10M-token scan can't show
  a rare concept's neurons properly. Follow-up: an enriched scan of all 622 Golden Gate documents in the shard.

**Double-negative regime:** L3:4061's positive top 40 is 39/40 double-negative firings (an echo of its "orange"
negative pole), the essay's warning in action. 1 of 62 units. All other units' top firings were the main regimes
(++ for positive, +- for negative), so negative poles are not contaminated by the mirror leak.

**Blind labels (Luna writes, Jev grades), 62 units** (`results/labels_Qwen3-8B.json`, `results/summary_Qwen3-8B.md`).
Cost: Jev ~$0.14 (labelling) + ~$0.09 (intruder); Luna ~403k input / 128k output tokens.

| Unit type | n | Pos clean | Neg clean | Luna pos verdicts ONE/TWO/MIX/NONE | Luna neg verdicts | Intruder P pos/neg |
|---|---|---|---|---|---|---|
| GG candidates | 24 | 9 | 6 | 16/1/4/3 | 14/2/4/4 | 0.40/0.39 |
| Exploratory | 8 | 5 | 5 | 7/0/0/1 | 6/0/0/2 | 0.45/0.46 |
| Random neurons | 20 | 10 | 3 | 16/2/0/2 | 9/4/3/4 | 0.39/0.33 |
| Planted mixtures | 10 | 0 | 1 | 5/0/4/1 | 2/1/5/2 | 0.29/0.32 |

(Clean = Luna says ONE AND Jev finds ≥ 16/20 examples fit AND ≤ 4/20 random tokens fit. Intruder chance = 0.20.)

**Reading:**
- **Tool check passed:** planted mixtures are almost never clean (0/10 pos, 1/10 neg) and have the lowest intruder
  scores. **Luna alone is lenient:** it called 5/10 mixtures ONE on the positive pole. The cross-check by Jev is what
  catches them.
- **Random neurons:** 50% of positive poles and 15% of negative poles are clean under the strict bar; Luna-only
  verdicts give ONE or TWO_RELATED for 90% of positive and 65% of negative poles. The truth is probably nearer the
  strict end (given Luna's leniency on mixtures), but the strict bar is also known to under-credit syntactic units.
  Either way it's below the essay's ~90% for Qwen3-32B. Not a like-for-like comparison: smaller model, 20-example
  windows from only 10M tokens, a different rubric and graders.
- **Why poles fail:** mostly partial coverage (Luna's ONE explanation fits < 16/20 by Jev), then TWO_RELATED (often
  sensibly related, e.g. L31:2281 "taste", literal and figurative, which is the essay's recursive generalisation),
  then genuine NONE. Some top lists are padded with repeated boilerplate documents (FineWeb duplicates).
- **Candidates are no cleaner than random neurons**: they're mostly word-piece and compound-completion units that
  include "Golden Gate" among many names.

---

## 15 Oct 2026 (late): enriched Golden Gate scan + steering

**Enriched scan:** all 622 FineWeb documents mentioning "Golden Gate" (440,947 tokens, windows centred on the first
mention), merged losslessly into each unit's top 40 (`results/scan_Qwen3-8B_merged.json`).
- **L7:4091** (top pre-registered candidate): 31/40 of the merged top firings sit next to "Golden Gate". But its very
  strongest are still name or number completions ("Rosenzwe|ig" 5.7, ISBN digits 5.0) vs "Golden Gate| Bridge" 4.4
  (78% of peak). The 31/40 is inflated by enrichment. Reading: a **"finish a multi-part proper name"** neuron with GG
  as one instance. Same kind: L6:4064 ("kott|ke", "Kett|ner"), L12:9831 negative pole ("Kanaw|ha", "Berg|en",
  "Golden| Gate Park").
- **L29:6105** (late): fires on "Golden" *before* "Gate" ("under the |Golden Gate Bridge", "in |Golden Gate Park"),
  at 30–34 (its peak), and on "Delta" in "Delta Stewardship Council". Reading: an **"about to say a Bay Area/California
  place name"** prediction neuron.
- **Tool caveat:** the planted mixture of L7:4091 + a random neuron also shows 28/40 GG-adjacent top firings. The strong
  half hides the other half, so **top-k views can't rule out a weaker packed job**; only causal tests can.

**Steering** (`results/steer_*.json`): 20 neutral prompts, clamp at 0/2/5/10/20× own peak, Jev-judged.

| Clamped | 2× | 5× | 10× | 20× |
|---|---|---|---|---|
| L29:6105 alone | GG 0%, fluent 100% | 0%, 95% | 0%, 95% | 0%, 0% |
| L7:4091 alone | 0%, 100% | 0%, 95% | 0%, 60% (switches to an "Okay, I need to figure out…" monologue) | 0%, 0% (empty) |
| Top 10 candidates | 0%, 80% | 0%, 0% ("Engranted…") | 0%, 0% | 0%, 0% |
| 10 random neurons (same layers) | 0%, 0% ("Okay, 4.1.1…") | 0%, 0% | 0%, 0% | 0%, 0% |

**Against PREREG:**
1. Selective unit with a clean positive pole: **met** (9/24 candidates have clean positive poles), but the clean
   candidates are word-piece or name-completion patterns, not the bridge concept.
2. Explanation survives: assessed via the Jev coverage and random-token check (a near-miss test was not run).
3. Intruder: candidates 0.40 vs planted mixtures 0.29, so the GG unit doesn't reach the 0.8 bar. The tool-check part
   (mixtures lower) holds, but the gap is modest.
4. Steering ≤ 10 neurons → GG in ≥ 50% of replies: **failed** (0% everywhere).

**Overall:** the pre-registered **ambiguous** case: incoherence before any topic, for candidates and random neurons
alike. Most likely reason: the noun-first contrastive search surfaced word-piece, name-completion and say-next
neurons. Clamping those injects "finish a name" / "say a place now", not a concept to drift towards. Not evidence of
superposition. It is evidence that (a) an 8B model may have no single noun-shaped Golden Gate neuron, consistent with
the essay's "composition of legible neurons" picture, and (b) our search method was biased toward nouns, which the
essay warns about.

**Next:** trace the computation instead of hunting a noun. Use prompts that require recalling the bridge (e.g. "The
famous orange suspension bridge at the mouth of San Francisco Bay is called the …"), rank neurons by contribution to
" Golden" (attribution), ablate the top-k to test necessity, and label those neurons with the existing pipeline. The
essay's prediction: a handful of individually legible neurons carry it.

---

## 15 Oct 2026 (late): Experiment GG-2, tracing the Golden Gate computation (EXPLORATORY: designed after GG-1's results)

Motivation (James): GG-1 hunted for a noun-shaped "Golden Gate neuron"; the essay warns neurons are patterns or
distinctions, and that a Golden Gate computation should be "ordinary composition" of a few legible neurons.

**Setup:** 14 recall prompts where Qwen3-8B answers " Golden" (P = 0.26–1.00, mean 0.70), none containing "Golden Gate".
6 controls with other answers (Brooklyn, Bay [Bridge], Al[catraz], [Sydney] Harbour, Everest, [Big] Ben).
Attribution = activation × gradient of log P(" Golden"), summed over positions, averaged over prompts.

**Ablation (zero the top-k attributed neurons; `results/attribution_Qwen3-8B.json`):**

| k | P(Golden) | controls (mean) | k random neurons, same layers: P(Golden) / controls |
|---|---|---|---|
| 0 | 0.70 | 0.94 | – |
| 1 | 0.42 | 0.93 | 0.71 / 0.94 |
| 2 | 0.16 | 0.85 | 0.70 / 0.93 |
| 5 | 0.02 | 0.69 | 0.70 / 0.94 |
| 20 | 0.002 | 0.51 | 0.71 / 0.94 |
| 500 | 0.000 | 0.36 | 0.70 / 0.93 |

- **Five neurons are enough to abolish Golden Gate recall**; 500 random ones do nothing. The linear "share of attribution"
  (top 10 = 1%) badly understates this, because thousands of tiny contributions roughly cancel. Ablation is the
  measure to trust.
- **Per-control damage matches the labels:** with the top 10 off, Brooklyn 0.97→0.16, Sydney Harbour 0.97→0.44,
  Bay 1.00→0.73, Alcatraz 0.68→0.25, but **Everest 0.99→0.99 and Big Ben 1.00→1.00**. Bridge answers lose their "bridge"
  neurons; non-bridges are untouched.
- With the top 5 off, Qwen still knows it is naming something (`"`, "San…", "city", "the only…") but loses the name.

**What the top neurons write (direct vocabulary projection of their output weights):**
- L34:5510 (#1): gold, Gold, golden, GOLD, 金, i.e. "say gold"
- L28:6711, L31:9611: bridge(s), 桥, 大桥, 橋, tunnels, i.e. "say bridge"
- L26:11492: tower, landmark(s), monuments, 地标, i.e. "say landmark"
- L29:6056: tall, tallest, height
- L35:8393, L35:3129: push *down* "____"/"______", i.e. "don't answer with a blank" (the competing answer format)
- Early-layer entries (L0:3546, L1:9791, L10:488) act indirectly; their direct projection is noise. They're used by
  only 1/14 prompts (L0, L1), i.e. prompt-specific.
- So "Golden Gate" is **composed**: gold + bridge + landmark + "actually name it". That's the essay's "ordinary
  composition of legible neurons". The output readouts are multilingual (金/桥/橋/地标), i.e. concept-level output
  neurons, not English word pieces.

**Steering with the 4 composed neurons** (L34:5510 gold, L28:6711 + L31:9611 bridge, L26:11492 landmark), clamped to
k × own peak, 20 neutral prompts:

| Strength | Jev: about Golden Gate | Jev: fluent | 4 random neurons (same layers): GG / fluent |
|---|---|---|---|
| 0.5× | 0% | 100% | – |
| 0.75× | 10% | 90% | – |
| **1×** | **45%** | 40% | 0% / 95% |
| 2× | 20% | 5% | 0% / 100% |
| 3–8× | 0% | 0% (bridge bridge…) | 0% / 100→0% |

Examples at 1×: "The most beautiful thing I can think of is the Golden Gate Bridge in San Francisco… its iconic
orange color…"; "Fun fact: The Golden Gate Bridge… International Orange"; coffee shop "The Golden Span… a cozy bridge
over the Golden Gate". Composition is visible in the failures too: "Golden Bridge, Vietnam" (Cầu Vàng),
"Bridges of gold beneath the sky".

**Verdict for GG-2 (exploratory):** Golden Gate Qwen *was* made from 4 raw MLP neurons, no SAE. The neurons are
legible by their output projections and ablation side-effects. This supports the essay's composition picture and its
"neurons are already usable" thesis for this concept and model. Caveats:
1. Designed after GG-1, on prompts I wrote, all bridge-themed.
2. The output projection is the direct effect only.
3. Ablation side-effects on bridge controls show the neurons are "bridge"/"gold" general, not Golden Gate-specific
   (as the composition story predicts).
4. Input-side labels pending (scan running).

**GG-2 input-side labels** (5M FineWeb tokens; blind Luna + Jev; dossiers from natural text only, not the GG-enriched
docs; `results/labels_attr_Qwen3-8B.json`):

| Units | Pos clean | Neg clean | Luna pos ONE/TWO/MIX/NONE |
|---|---|---|---|
| Top 20 attributed | 10/20 | 9/20 | 18/0/1/1 |
| 10 random, same layers | 2/10 | 4/10 | 5/1/2/2 |

Input labels match the output readouts:
- L34:5510 (writes "gold/金"): negative pole fires in gold text (prices, mining, Gold Rush), clean.
- L28:6711 and L31:9611 (write "bridge/桥/橋"): named bridges / bridge language, clean.
- L26:11492 (writes "landmark/tower"): tourist sights and cultural landmarks.
- L29:6056 (writes "tall"): skyscrapers and tall structures, clean.
- **L30:2600**: Bay Area / Silicon Valley places (+) vs Sacramento and other Northern California (−), both poles clean.
  A *distinction*, as the essay describes.
- **L4:12199**: "pont" (French "bridge") → *puente*, *Pont*oon, *Pont*iac. Recursive generalisation in miniature.
- L10:488 and L8:5740: bridges on their negative poles, clean. L29:8780: San Francisco / Bay Area on its negative pole
  (25/40 merged top firings next to "Golden Gate"); Luna's label, but Jev coverage only 3/20.
- The early-layer hits are prompt-specific word detectors: L0:3546 "field" and L1:9791 "Christopher" (both from the
  "Crissy Field" prompt), which is why each appeared in only 1/14 prompts.
- Exception: L35:8393 ("don't answer with a blank") has no specific input pattern (explanation fits 16/20 random
  tokens); its role shows only on the output side.

**Overall (GG-2, exploratory):** the Golden Gate answer is produced by a small set of individually legible neurons:
Bay Area place detectors, several bridge detectors (incl. French "pont"), landmark and tall-structure neurons, a gold
output neuron, and blank-suppressors. Ablating 5 abolishes recall, with damage confined to bridge-related controls.
Clamping 4 produces Golden Gate talk (45% at 1× peak, with a fluency trade-off). This is the essay's "ordinary
composition of legible neurons" picture.

**Next (agreed with James, 15 Oct):**
1. Build a neuron atlas (top-40 per pole for all 442k neurons) so any neuron can be labelled instantly.
2. Automate the concept pipeline (prompts → attribution → ablation → labels → steering → nudgeability score) and
   improve steering quality.
3. Pre-register ~20 concepts plus controls (unknown/made-up concepts, shuffled attribution) and run them.
4. Census of random-neuron cleanliness, and a scale ladder (1.7B → 32B) for the essay's "bigger = cleaner" claim.
5. Then an outer loop that improves the method against a frozen held-out evaluation.

---

## 15 Oct 2026 (night): infrastructure for scale

**Neuron atlas (`src/build_atlas.py`, `src/atlas.py`):** top 40 positive and top 40 negative firings for every MLP
neuron, with the gate sign (so the sign regime is recoverable), from one pass over FineWeb.
- Qwen3-8B: 10,005,329 tokens, 20,359 docs, all 442,368 neurons, 43.6 min (~3,850 tok/s), stored in `data/`.
- Validation on Qwen3-0.6B (2M tokens): each neuron's #1 entry matches the independently computed max/min for 99.9%
  of 86,016 neurons. Double-negative leaks make up 2.0% of all top-40 positive entries in 0.6B.

**Automated concept pipeline (`src/concept_pipeline.py`):** tested end to end on 0.6B. Records recall, neuron and
head attribution (top 200 neurons, bottom 50, per-prompt and per-token), ablation curves vs random (neurons and heads,
with controls), single-neuron ablation, atlas-based blind labels and output readouts, a steering grid with every reply
and per-reply Jev scores, a random-neuron steering control, and provenance hashes.

**Batched labeller (`src/label_batch.py`):** same clean bar as `label.py`, but the random-token check is batched
(~100 explanations per Jev call) and everything is heavily parallel. Test on 40 random 0.6B neurons: 49 s,
Jev ~$0.0005 + Luna ~$0.0009 per neuron (vs ~$0.004 before). Projected full map of 0.6B (86k) ≈ $120; a 10% sample
of 8B ≈ $60. (Test numbers, not results: 18/40 positive and 13/40 negative poles clean, with no mixture controls yet.)

---

## 16 Oct 2026 (~00:10): Experiment CEN, census of 1,000 random Qwen3-8B neurons (pre-registered)

`results/census_Qwen3-8B.json` (+ `_summary.json`). Labels: `label_batch.py` (Luna explains from 15 atlas examples;
Jev grades 10 per pole; batched random-token check). Cost: Jev ~$0.50, Luna ~3.3M input / 1.1M output tokens.

**Near-silent neurons (new finding, post-hoc category):** 46/1000 sampled neurons never fire above 0.05 in 10M
tokens; all are in layers 1–4. Model-wide (from the atlas): ~20,000 neurons (4.5%), i.e. 56% of layer 1, 57% of
layer 2, 42% of layer 3, 9% of layer 4. Excluded from the percentages below; they have no job to label. (17 of them
also triggered a division-by-zero bug in dossier rendering, now fixed.)

**Live neurons:**

| | Positive poles (n=939) | Negative poles (n=953) |
|---|---|---|
| Essay-style (Luna verdict): ONE / TWO_RELATED / MIXTURE / NONE | 81% / 2% / 8% / 9% (ONE+TWO 83%) | 61% / 4% / 14% / 21% (ONE+TWO 65%) |
| Strict clean (ONE + ≥80% coverage + ≤20% random fit) | **41%** | **32%** |
| Strict by depth (early / middle / late thirds) | 53% / 34% / 39% | 37% / 28% / 32% |

- Of 762 positive poles Luna called ONE, 332 fail on coverage (Jev agrees with < 8/10 examples) and 72 are too vague
  (fit > 20% of random tokens).

**Against PREREG (CEN):**
- Essay-style ≥ 85% with MIXTURE ≤ 5%: **missed** (positive 83% / 8%, negative 65% / 14%).
- Strict 40–70% (my guess): positive 41%, inside; negative 32%, below.
- Depth (late ≥ early): **not supported**; early is cleanest (53%). Likely because early layers hold easy-to-label
  spelling and word-piece detectors, i.e. labelability rather than cleanliness, but that's an interpretation.
- Overall: **ambiguous, leaning against.** Pooled essay-style ~74% (between the 70% and 85% bars); pooled strict ~36%
  (< 40%, which counts as "against" only if the planted mixtures show the tools catch mess — pending).

**Caveats:** 8B, not 32B; fast automatic labels from 15 examples out of 10M tokens, not in-depth investigations;
Jev is known to under-credit syntactic jobs.

---

## 16 Oct 2026 (~01:00): Experiment CR, concept replication (pre-registered, frozen 23:10; steering = clamp, reply only)

23 concepts run (20 real + 3 made-up) with `concept_pipeline.py` on Qwen3-8B. 11 recalled. 12 failed recall: the
3 made-up controls (correctly) plus 9 real concepts (all 5 people, Sahara, the Moon, chess, Harry Potter) failing
because of a prompt-ending flaw (ISSUES #9/#10: the model wants "…is **called** chess", "…was **the**…"; the answer
is in its top 5 but not next).

| Concept | k50 | 50 random / base | Unrelated-control drop | Top-10 clean (either pole) | Nudgeability, joint (pre-reg product) | Random |
|---|---|---|---|---|---|---|
| coffee | 3 | 100% | 7pp | 7/10 | 0.85 (0.85) | 0.05 |
| democracy | 5 | 98% | 11pp | 7/10 | 0.55 (0.59) | 0.00 |
| Eiffel Tower | 5 | 101% | 0pp | 7/10 | 0.40 (0.40) | 0.00 |
| Golden Gate Bridge | 1 | 98% | 0pp | 7/10 | 0.35 (0.35) | 0.00 |
| Mount Everest | 2 | 98% | – | 9/10 | 0.30 (0.27) | 0.00 |
| photosynthesis | 2 | 100% | – | 5/10 | 0.20 (0.32) | 0.00 |
| the French Revolution | 1 | 102% | −1pp | 7/10 | 0.15 (0.23) | 0.00 |
| the Titanic | 3 | 96% | – | 4/10 | 0.00 | 0.00 |
| the violin | 2 | 101% | – | 6/10 | 0.05 | 0.00 |
| Tokyo | 5 | 100% | – | 6/10 | 0.10 | 0.00 |
| Venice | 2 | 100% | 9pp | 8/10 | 0.00 | 0.00 |

**Against PREREG CR:**
- **P1 handful:** 11/11 (100%) have k50 ≤ 10 (actually 1–5), with 50 random ablations keeping 96–102% of baseline.
  ✅ Strongly supported.
- **P2 specific:** mean unrelated-control drop 4.4pp (≤ 10pp). ✅ Thin: 6 concepts, 1–2 usable controls each.
- **P3 legible:** top-10 clean on either pole 66% vs random census neurons 58%. ✅ Modest gap.
- **P4 nudgeable:** pre-registered product metric 6/11 = 55% ✅; corrected joint metric (ISSUES #13) 5/11 = 45% ⚠️.
- **Controls:** made-up concepts 3/3 fail recall ✅; shuffled judge (steer A, ask about B) 0% in all 5 pairs ✅.
- **Heads (exploratory):** halving recall needs 5–50 heads vs 1–5 neurons. Concept knowledge looks more concentrated
  in MLP neurons than in specific heads.

**Reading:** strong support for the essay's composition claim. For every concept the model recalled, recall runs
through a handful of neurons, specifically (random ablations and unrelated facts untouched), built from shared,
often multilingual, legible parts (landmark, tall, mountain/山, bridge/桥, coffee/咖啡, democracy/民主, Paris). Steering
works for about half the concepts with the frozen late-layer rule. Failures are concentrated where the late-layer
selection picks generic name-completion neurons (Venice, Titanic, violin, Tokyo; ISSUES #11).

**Limits:**
- 11 concepts, no people (recall-template flaw).
- 60-token steered replies and other caps (CAPS.md).
- Labels from 10M-token atlas examples.
- Jev and Luna graders (validation pending).

---

## 16 Oct 2026 (~01:30): cap check on the census (does our capping drive the 41%?)

Same 200 live census neurons re-labelled with aligned, uncapped settings: Luna sees 20 examples × ~1,500 chars, no
word limit; Jev grades all 20 at 1,500 chars; random snippets also 1,500 chars. Original census: 15 × 400 chars to
Luna, 35-word cap, 10 graded at 500, random snippets 320. (`results/census_capcheck_aligned.json`; Luna ~$0.36, Jev ~$0.25.)

| | Capped | Aligned, uncapped |
|---|---|---|
| Positive poles strictly clean | 40% | 37% |
| Negative poles strictly clean | 36% | 22% |
| Positive ONE (Luna only) | 79% | 81% |
| Mean coverage (pos / neg) | 0.65 / 0.60 | 0.63 / 0.53 |
| Mean random fit (pos / neg) | 0.07 / 0.08 | 0.07 / 0.04 |

- **The census headline is not a cap artefact:** more context and no word cap did not raise the clean rate.
  Grading 20 examples (incl. weaker firings, ranks 11–20) makes the 80% bar harder, which plausibly explains the
  negative-pole drop. Several settings changed at once, so the drop can't be attributed to one of them.
- **Per-neuron verdicts are noisy:** 44/200 positive poles flipped clean status between runs (25 lost, 19 gained)
  while the aggregate moved 3 points. Population rates are reasonably stable (≈ ±3pp); a single neuron's label has
  ~20% flip risk, so any per-neuron claim needs a repeat.

**Noise floor, test-retest (same 200 neurons, identical original settings; `results/census_retest_capped.json`;
Luna ~$0.18, Jev ~$0.10):**

| | Run 1 | Run 2 (identical) | Aligned |
|---|---|---|---|
| Positive clean | 40% | 40% | 37% |
| Negative clean | 36% | 40% | 22% |
| Clean-label disagreement vs run 1 | – | 22% / 20% (pure noise) | 22% / 25% |
| Verdict disagreement vs run 1 | – | 23% / 32% | 19% / 34% |

- Population rates are stable; individual labels are noisy (~1 in 5 flips on an identical re-run).
- Positive poles: the cap change is indistinguishable from noise. Negative poles: the aligned settings are genuinely
  harsher (22% vs 36–40%), likely from grading 20 examples incl. weaker firings.
- Positive poles clean in BOTH identical runs: 29% ("robustly clean"); clean in at least one: ~51%.
- **Census headline restated:** strictly clean ≈ 30–50% (robust to single-run) of positive poles; ~80% "one job" by
  the labeller alone; per-neuron claims need a repeat.

---

## 16 Oct 2026 (~00:10): H-DEPTH, stages by depth (pre-registered 00:35 on 15 Oct, before the test)

11 recalled concepts × top 50 attributed neurons = 550 pairs (415 unique neurons, labelled blind with `label_batch.py`
default settings; 49% / 40% of their pos / neg poles clean). Jev classified each pair CONCEPT_SPECIFIC / GENERIC /
UNRELATED from the neuron's explanations and output words, with option order rotated ×3 (`results/hdepth_classes.json`).

Representation of each class by depth band, relative to the depth distribution of all 550 pairs:

| Class (n) | 0–5 | 6–12 | 13–21 | 22–32 | 33–35 |
|---|---|---|---|---|---|
| Concept-specific (101) | ×0.6 | **×2.3** | ×0.2 | **×1.4** | ×0.3 |
| Generic (342) | ×0.8 | ×0.3 | ×1.4 | ×0.9 | **×1.4** |
| Unrelated (107) | ×2.1 | ×2.0 | ×0.5 | ×1.1 | ×0.3 |

Most recurrent generic neurons (number of the 11 concepts' top-50 they appear in): L35:10037 (10), L35:9896 (9),
L33:10933 (9), L35:11693 (8), L34:1452 (7), L35:8393, L35:3129, L35:122, L31:5473, L21:10152 (6 each).

**Against PREREG H-DEPTH:**
1. Concept-specific over-represented in ~6–12 and ~22–32 ✅ (×2.3, ×1.4; nearly absent in 13–21).
2. Generic over-represented in 32–35 ✅ (×1.4).
3. The same generic neurons recur across ≥ 50% of concepts ✅ (10 neurons in ≥ 6/11).

**Reading:** recognise (6–12) → concept parts (22–32) → shared answer machinery (33–35). 62% of a concept's top 50 is
generic machinery; 18% concept-specific. "Unrelated" (prompt-specific word detectors) cluster in 0–12.
**Caveats:** one labelling run + one classification; per-label noise ~20% (test-retest), so the pattern rests on
aggregation over 550 pairs. A repeat classification is the planned noise check.

---

## 16 Oct 2026 (~00:20): grader validation (Jev coverage), and the census figure is threshold-driven

100 census neurons; each explanation graded against its own 8 top examples and 8 of another random neuron's
(`results/grader_validation.json`, Jev ~$0.09):

| Wording | AUC own vs other | Decisive (<0.1 or >0.9) | Near 0.5 | Mean P own / other |
|---|---|---|---|---|
| W0 current yes/no | 0.90 | 15% | 43% | 0.57 / 0.22 |
| **W1 3-level rubric** | **0.92** | **37%** | 29% | 0.67 / 0.18 |
| W2 "does it show" | 0.89 | 17% | 42% | 0.51 / 0.19 |

Share of neurons whose explanation "covers" its own examples, vs false matches on another neuron's:

| Bar | Own | Other |
|---|---|---|
| W0, ≥ 0.5, ≥ 80% (census bar) | 35% | 1% |
| W0, ≥ 0.5, ≥ 60% | 72% | 3% |
| W1, ≥ 0.5, ≥ 80% | 48% | 3% |
| W1, ≥ 0.5, ≥ 60% | 79% | 5% |

**Reading:** Jev discriminates well (AUC ~0.9), but its own-example probabilities hover near 0.5. The census
"strictly clean ≈ 40%" is largely set by the uncalibrated "≥ 80% of examples at ≥ 0.5" bar; defensible bars give
35–79%. **The earlier "leaning against the essay" reading of the census is withdrawn** pending calibration on planted
mixtures (known two-job units) and a census re-grade with the rubric wording and soft scores.

**H-DEPTH noise check (independent repeat: fresh Luna labels + fresh Jev classification):** essentially identical:
concept-specific ×2.3 (6–12), ×0.2 (13–21), ×1.3 (22–32); generic ×1.4 (33–35); same top recurrent generic neurons;
**91% of 550 pair classifications agree** between runs. Jev was confident on these (median top-class P 0.91), and
probability-weighted counts give the same pattern. **H-DEPTH: supported and replicated.**

**Census re-grade with the rubric grader, soft scores (all 1,000; explanations unchanged; Jev ~$0.28;
`results/census_regrade_rubric.json`):** positive-pole coverage_soft median 0.69 (IQR 0.58–0.80), random_soft median
0.13. "ONE + coverage_soft ≥ t + random_soft < 0.25" for positive poles = 60% / 50% / 37% at t = 0.5 / 0.6 / 0.7
(negative: 47% / 40% / 29%). The bar t is calibrated next on planted mixtures.
**Jev consistency:** 30 random-snippet grades re-asked singly differ by 0.012 on average (max 0.06), and batched by
0.017 (max 0.085), on a 0–1 scale. Jev is near-deterministic, and batching doesn't change answers.

---

## 16 Oct 2026 (~00:45): recall-format repair, mixture calibration, middle-of-range decay

**Recall variants on all 23 concepts (`results/recall_variants.json`):** real concepts recalled (≥ 5/20 prompts):
V0 frozen 11/20 (mean pass 28%), V1 bridge 11/20, V2 rewritten 5/20, **V3 question-style 13/20 (56%)**; made-up 0%
for all. V3 fixes the people (Einstein 5→100%, Shakespeare 0→90%, Napoleon 0→80%, Mozart 0→65%, Kahlo 15→55%) and
the Moon, Titanic, Harry Potter. With answer forms (ISSUES #26), the remaining V3 failures recover (coffee 75%,
democracy 65%, photosynthesis 100%, chess 75%, violin 75%, French Revolution 100%); only Sahara stays low (20%).

**Planted activation-mixture calibration (60 mixtures; `results/mix_regrade.json`):** mixture vs census pass rate,
positive poles: t=0.5 35% vs 60%; t=0.6 27% vs 50%; t=0.7 17% vs 37%; t=0.8 8% vs 20%. Luna called 41/60 mixtures
ONE. **Control judged weak by construction** (ISSUES #25: the top-k of a blend is dominated by one component). Not
used for the headline; replaced by spliced dossiers (in progress).

**Middle-of-range (94 neurons with all bands; soft rubric; `results/strata_graded.json`):** how well each neuron's
top explanation fits its firings at lower strengths:

| | Top | ~50% of peak | ~25% | ~10% | Random text |
|---|---|---|---|---|---|
| All | 66% | 51% | 41% | 33% | 20% |
| Clean at the top | 77% | 59% | 42% | 28% | 8% |

Smooth decay, staying above the random baseline down to 10% of peak (clean neurons: 28% vs 8%). Consistent with the
essay's "meaning radiates outward" rather than an unrelated job hidden in the weak firings; near 10% of peak the fit
approaches the random level, as the essay itself describes for near-zero activations.

---

## 16 Oct 2026 (~00:55): CENSUS HEADLINE, calibrated on spliced two-job dossiers

**Spliced control** (`results/spliced_regrade.json`; Luna ~$0.12, Jev ~$0.20): 100 units, each = 10 top examples of
census neuron A + 10 of an unrelated neuron B (≥ 6 layers apart), shuffled; labelled and rubric-graded exactly like the
census. Luna verdicts: ONE 48, MIXTURE 37, NONE 13, TWO_RELATED 2. **Luna alone calls half of known two-job dossiers
"one job"**, so labeller-only "one job" rates (census 81%) are inflated.

Clean = Luna ONE + rubric coverage_soft ≥ t + random_soft < 0.25:

| t | Spliced pass (pos) | Census pos clean (95% CI) | Spliced pass (neg) | Census neg clean (95% CI) |
|---|---|---|---|---|
| 0.5 | 15% | 60% (57–63) | **4%** | **47% (44–50)** ← calibrated (neg) |
| **0.6** | **8%** | **50% (47–54)** ← calibrated (pos) | 0% | 40% (37–43) |
| 0.7 | 2% | 37% (34–40) | 0% | 29% (26–31) |

**Headline (Qwen3-8B, 954 live neurons):** at a bar that passes ≤ 10% of known two-job dossiers, **~50% of positive
poles and ~47% of negative poles have a verified single job.** Plus 4.5% of all neurons are near-silent (mostly layers 1–3).
- This is closer to a floor than a point estimate: sensitivity for true single-job poles is < 100% (noisy coverage
  grading, long-context jobs under-credited), and the false-pass rate is measured only on blatant mixtures.
- **Versus the essay** (~90% one main job, Qwen3-32B, in-depth audits): ours is "at least ~half, verified" with
  fast automatic labels on a 4× smaller model. Not a refutation; short of the essay's figure, with a known downward bias.
- **PREREG CEN leniency check:** spliced (the meaningful mixed control) pass 8% (pos) / 4% (neg) at the calibrated bars,
  within the ≤ 10% target. Activation-blend mixtures: control weak by construction (ISSUES #25).

---

## 16 Oct 2026 (~01:15): Experiment CR-2, replication with repaired recall (exploratory, predictions pre-written)

Fast pipeline v2; question-style recall (V3) with natural answer forms; leak filter; realistic made-up controls;
clamp, reply only; joint nudgeability (`results/concepts_v3/`, `results/cr2.log`). ~45 s per concept.

| Concept | k50 CR | k50 CR-2 | Nudge CR (joint) | Nudge CR-2 | Random | Top-10 overlap CR vs CR-2 |
|---|---|---|---|---|---|---|
| Albert Einstein | – | 10 | – | 0.05 | 0.05 | – |
| Eiffel Tower | 5 | 20 | 0.40 | 0.25 | 0.00 | 4/10 |
| Frida Kahlo | – | 5 | – | 0.00 | 0.00 | – |
| Golden Gate Bridge | 1 | 2 | 0.35 | 0.20 | 0.00 | 5/10 |
| Harry Potter | – | 10 | – | 0.00 | 0.00 | – |
| Mount Everest | 2 | 2 | 0.30 | 0.15 | 0.00 | 4/10 |
| Napoleon Bonaparte | – | 2 | – | 0.15 | 0.00 | – |
| Tokyo | 5 | > 50 | 0.10 | 0.25 | 0.00 | 1/10 |
| Venice | 2 | 3 | 0.00 | 0.35 | 0.00 | 3/10 |
| William Shakespeare | – | 5 | – | 0.00 | 0.00 | – |
| Wolfgang Amadeus Mozart | – | 1 | – | 0.00 | 0.00 | – |
| chess | – | 50 | – | 0.35 | 0.00 | – |
| coffee | 3 | 20 | 0.85 | 0.60 | 0.05 | 2/10 |
| democracy | 5 | 1 | 0.55 | 0.50 | 0.00 | 2/10 |
| photosynthesis | 2 | 50 | 0.20 | 0.20 | 0.00 | 3/10 |
| the French Revolution | 1 | 5 | 0.15 | 0.30 | 0.00 | 3/10 |
| the Moon | – | 20 | – | 0.60 | 0.05 | – |
| the Titanic | 3 | 3 | 0.00 | 0.00 | 0.00 | 0/10 |
| the violin | 2 | 2 | 0.05 | 0.10 | 0.00 | 2/10 |

**Against pre-written predictions:**
- Recall 19/20 real incl. 5/5 people ✅ (Sahara only failure).
- Realistic made-up 0/3 recalled ✅.
- **P1 13/19 = 68%** (≥ 70%) ⚠️ narrowly missed.
- P3 top-10 clean either pole 68% vs census 58% ✅.
- **P4 8/19 = 42%** (≥ 50%) ❌.

**Findings:**
1. **Concentration depends on the prompt format.** Completion-style recall (CR) gives 11/11 with k50 ≤ 5; question-style
   (chat) gives 13/19, with six concepts needing 20–50+. Caveat: k50 ablates in attribution order; if attribution ranks
   worse in chat format, k50 overstates the minimal set. So this shows "looks less concentrated by our method", not yet
   "is less concentrated".
2. **A concept's top-10 list is not a stable fingerprint across prompt sets** (overlap 0–5/10). Recurring parts are stable
   (gold, landmark); exact rankings are prompt-dependent. This is the prompt-set noise term.
3. **People and characters are recalled through 1–10 neurons but not steerable** (all ≤ 0.15). Places, objects and
   ideas often are (the Moon 0.60, coffee 0.60, democracy 0.50). Hypothesis: names come out through name-piece neurons,
   not meaning-level ones.

**k50 re-check (CR-2 caveat 1; `results/concepts_v3/k50_check.json`):** re-ranking the top 200 by measured
single-neuron effect: chess 50→5, coffee 20→10, Eiffel 20→10, photosynthesis 50→20, the Moon 20→20, Tokyo > 50 → > 50,
Golden Gate 2→2, democracy 1→1. The concepts that stay "spread out" are the near-certain ones (baseline P 0.97–1.00):
"halve P" is a saturating measure, so CR-2's format-dependence finding is probably mostly a measurement artefact
(ISSUES #27). To be re-measured on log-odds.

## 16 Oct 2026 (after CR-2): lexical-confound check on recall prompts (ISSUES #2)
`src/lexical_check.py`, all CR + CR-2 runs (29 concept runs). For each concept: content words recurring in ≥ 30% of
its recall prompts (template words and the answer's own words excluded). Then: which top-10 neurons have a recall-pole
label naming such a word but not the concept. **17/29 runs have one, 42 neurons in total** (e.g. "landmark" for
Eiffel/Golden Gate, "ship" for Titanic, "artist" for Frida Kahlo, "instrument" for violin).

A label match alone does not make a confound, because "landmark" is also a genuine part of the concept. The decider uses
the per-prompt attribution: does the neuron only matter on prompts that contain the word?
- **Clue-readers (attribution without the word < 10% of with it): 4/42.** democracy L0:10457 ("political"), violin
  L10:7290 and L8:11566 ("instrument"), Mozart L11:1120 ("opera"). **All in layers 0–11**, the "recognise the
  question" stage of H-DEPTH, where reading the prompt's words is the expected job.
- **The opposite: 6/42.** The neuron matters *only when the word is absent*: chess L27:92 and L28:1512 ("game"),
  Eiffel L26:11492, L31:10160 and L28:4830 ("landmark"), Tokyo L5:3145. That looks like the model supplying the
  category itself when the prompt doesn't. The Eiffel "landmark" neurons were neutral (no split) in the
  completion-format CR run, so this is format-dependent.
- The rest (32/42) contribute on prompts with and without the word, at roughly the expected share.

**Verdict:** no GG-1-style "orange" confound in the late (≥ 22) concept-part neurons. Prompt words are read by
early neurons as expected. Prompt generation still has no lexical-diversity constraint: a cheap addition for CR-3 is
"no content word in more than 30% of prompts". (`results/lexical_check.json`, `results/lexical_split.log`)

## 16 Oct 2026 (after CR-2): Luna reasoning-effort A/B (ISSUES #24)
`src/effort_compare.py`: 200 random census neurons were relabelled at the API default, `low` and `none`, each re-graded
with the rubric grader and scored with the calibrated clean rule. The default re-run is the noise floor.

| arm | Luna output tokens | clean pos | clean neg | "ONE" pos | random-fit (soft) | agrees with original |
|---|---|---|---|---|---|---|
| original census | – | 48% (42–55) | 51% (44–58) | 78% | 0.19 | – |
| default re-run | 216k | 49% (42–56) | 50% (43–57) | 84% | 0.20 | 82% / 86% |
| low | 44k (5× fewer) | 36% (30–44) | 34% (27–40) | 55% | 0.26 | 80% / 78% |
| none | 18k (12× fewer) | 34% (27–40) | 28% (22–34) | 50% | 0.29 | 78% / 73% |

- **The census pipeline reproduces:** a default re-run gives the same population rate (49/50 vs 48/51). Per-neuron
  agreement is 82–86%, consistent with the earlier ~20% test-retest flip rate.
- **Less reasoning gives vaguer labels.** Explanations fit random web text more often (0.26–0.29 vs 0.20), and Luna
  calls fewer neurons "one job". Clean rate falls 12–22 points, well outside the noise floor. Coverage rises slightly
  (0.70 vs 0.67), as vaguer explanations would.
- **Decision: keep the default.** Reasoning is ~40–50% of Luna's output cost, but it buys specificity. Caveat: lower
  effort says "ONE" less often, which on its own could mean more honesty (default Luna calls ~half of spliced two-job
  dossiers ONE). The random-fit rise is the evidence that it is vagueness, not honesty.

## 16 Oct 2026 (after CR-2): does short grading context push "clean" down? (ISSUES #15)
`src/ctx_compare.py`: the same 200 census neurons and explanations, re-graded with up to 3,500 chars of left context
(900 atlas tokens) vs the usual 500. Random snippets are at the same length in each arm. Checked the long arm really is
long: median 1,258 chars, 58% of examples over 1,000. Shorter ones fire near the start of their document, where no more
context exists.

| pole | clean @ 500 chars | clean @ 3,500 chars | coverage (soft) | flips |
|---|---|---|---|---|
| pos | 48% (42–56) | 46% (40–53) | 0.670 → 0.651 | 1 up, 6 down |
| neg | 51% (44–58) | 50% (43–57) | 0.653 → 0.634 | 2 up, 4 down |

**Not supported.** Longer context does not rescue "misfits". Coverage drops very slightly (the grader has more text to
find a mismatch in), and the largest per-neuron gain is +0.04. The census headline is not held down by the 500-char
context. (The 500-char arm reproduces the original re-grade closely but not exactly: every per-neuron score within 0.03, so Jev is near- but not fully deterministic on identical input.)

## 16 Oct 2026 (after CR-2): strict steering judge (ISSUES #14)
`src/strict_judge.py` re-judged every saved steering reply (whole grid plus random controls, CR and CR-2) with "Is X a
central theme or main topic of this text (not just a passing mention)?". Fluency is reused from the original run.

| run | mean best joint, lenient | strict at the same setting | strict, best setting | concepts ≥ 0.5 lenient → strict | random control, strict |
|---|---|---|---|---|---|
| CR (11) | 0.27 | 0.14 | 0.16 | 2 → 1 | ≤ 0.05 |
| CR-2 (19) | 0.21 | 0.12 | 0.14 | 3 → 1 | ≤ 0.05 |

Per concept, strict best: coffee 0.85 → 0.50 (CR), 0.60 → 0.20 (CR-2); the Moon 0.60 → 0.50; democracy 0.55 → 0.40 /
0.50 → 0.35; Golden Gate 0.35 → 0.25 / 0.20 → 0.20. Every concept steerable under the lenient wording stays above
random under the strict one.

What the gap looks like: lenient-pass, strict-fail replies are the model answering the user's question with the concept
pushed to the front. Examples: "Here are some ideas… ### ☕ Morning Coffee & Tea"; "popular hobbies… 1. **Tea or Coffee
Brewing**". That is a real nudge, not a takeover. The replies are capped at 60 tokens, which makes "main topic" harsh.
**Both numbers are now reported.** Lenient ≈ "the concept intrudes"; strict ≈ "the concept takes over". Golden-Gate-Claude-style
takeover happens for coffee (CR) and the Moon at ~50%, and for most concepts at ≤ 25%.

## 16 Oct 2026 (after CR-2): second grader (ISSUES #16, part)
`src/grader2.py`: 400 (explanation, example) pairs sampled from the census re-grade, half the neuron's own top examples
and half random web snippets. Luna graded each with the identical rubric (0/1/2).

| | own examples, mean | random snippets, mean | AUC own vs random |
|---|---|---|---|
| Jev (soft) | 0.66 | 0.20 | 0.90 |
| Luna (0/0.5/1) | 0.85 | 0.21 | 0.88 |

Rank correlation 0.81; 80% agreement on fits / doesn't fit. **The grader is not idiosyncratic.** Two different models
rank the evidence the same way and discriminate equally well. Luna is more generous on own examples, plausibly
because it wrote the explanations (self-grading); Jev is the stricter of the two. The clean bar is calibrated on known
two-job dossiers, so a uniformly stricter grader is absorbed by the calibration rather than biasing the headline.
Still open under #16: human spot-checks.

## 16 Oct 2026 (after CR-2): steering-neuron selection rules, first run (prompt + reply steering, by mistake; ISSUES #11, #28)
`src/steer_select.py`, 19 recalled CR-2 concepts. Each rule's top 2/4/6 neurons over the usual 15-setting grid; best
joint nudgeability per rule. **This run steered the prompt as well as the reply** (ISSUES #28), unlike CR/CR-2. Rules
are compared fairly with each other, but the numbers are not comparable to CR-2. A reply-only re-run is queued.

| rule | mean best joint | concepts ≥ 0.5 | ≥ 0.25 | people (6) mean |
|---|---|---|---|---|
| R1 late-layer attribution (the frozen CR rule) | 0.26 | 5 | 9 | 0.04 |
| R2 attribution, any layer | **0.34** | **7** | 11 | 0.15 |
| R3 measured single-ablation effect | 0.32 | 5 | **12** | **0.16** |
| R4 "says the answer" (direct effect on answer token) | 0.30 | 4 | 11 | 0.12 |
| random neurons | 0.01 | 0 | 0 | – |

- **The layer cut-off was costing us.** Dropping it (R2) or ranking by measured effect (R3) beats the frozen rule.
  Everest 0.15 → 0.85 (R2), Harry Potter 0.05 → 0.50 (R2/R4), Titanic 0.00 → 0.30 (R2), Mozart 0.00 → 0.20.
- **"People can't be steered" weakens:** 0.04 → 0.15–0.16 on average, mostly from Harry Potter and Mozart. Einstein,
  Shakespeare and Kahlo stay ≈ 0 under every rule.
- No rule wins everywhere: Tokyo prefers R4 (0.60), Venice R3 (0.60), democracy R1 (0.75). Best-of-4 averages 0.43, but
  picking the max of four noisy scores inflates that. Hence the held-out re-test on fresh prompts.
- Same-rule check: R1 here vs CR-2 differs by up to 0.25 at identical neurons and settings. The cause is the
  steering mode (prompt steering is stronger), not noise.

## 16 Oct 2026 (after CR-2): recall concentration on a non-saturating scale (ISSUES #27)
`src/logodds_check.py` + `src/logodds_compare.py`. The measure is ablation's effect on the answer's **log-odds**,
log(p / (1 − p)), averaged over recall prompts. A drop of 3 means the odds fall ~20×, a same-sized effect whether the
model starts at P = 0.5 or 0.998. k_LO3 = fewest neurons (top 200 attributed, re-ranked by measured single effect) whose
joint ablation gives a mean drop ≥ 3. Random neurons from the same layers are the baseline.

| | completion format (CR, V0) | question format (CR-2, V3) |
|---|---|---|
| baseline P(answer) | 0.32–0.78 | 0.71–0.998 |
| k_LO3 ≤ 5 | 8/11 | **19/19** |
| k_LO3 = 1 or 2 | 0/11 | 15/19 (corrected from 11/19, a miscount) |
| random 5 neurons, mean drop | −0.04 to 0.04 | −0.05 to 0.04 |

Same concept in both formats: question format is as concentrated or more in 10/11 (democracy 20 → 1, Titanic 10 → 2,
French Revolution 5 → 1; coffee 3 → 5 is the exception).

**This reverses CR-2 finding 1** ("question format looks less concentrated"). The old picture came from k50's
saturation: halving P from 0.998 needs a ~9-point log-odds drop. On the scale that treats all confidence levels
alike, **1–5 neurons move recall in 27 of 30 concept runs, and every run within 20** (correction: this line first said "every one of the 30"; CR's democracy needs 20, Tokyo and Titanic 10). People are no exception
(Napoleon, Harry Potter, Shakespeare, Mozart: 1 neuron each).

Two views, both true:
- *Shifting the model's confidence* takes a handful of neurons. That is what log-odds measures.
- *Flipping a near-certain answer* (P 0.998 → below 0.5) can take more. That is what k50 measured.

Caveat being checked: a mean of per-prompt drops can be carried by one or two collapsing prompts. A re-run with the
per-prompt median, and with "fewest neurons that flip the answer on half the prompts", is queued (`*_v2`).

## 16 Oct 2026 (after CR-2): steering-neuron selection rules, reply-only (the proper comparison; ISSUES #11, #28)
Same as the first run, but steering the reply only, as in CR/CR-2. 19 recalled CR-2 concepts.

| rule | mean best joint | (prompt+reply run) | ≥ 0.5 | ≥ 0.25 | people (6) | per-concept wins (ties shared) |
|---|---|---|---|---|---|---|
| R1 late-layer attribution (frozen rule) | 0.21 | 0.26 | 3 | 7 | 0.03 | 3 |
| R2 attribution, any layer | 0.24 | 0.34 | 4 | 8 | 0.05 | 6 |
| R3 measured single-ablation effect | 0.25 | 0.32 | 2 | 11 | 0.12 | 9 |
| R4 "says the answer" | 0.24 | 0.30 | 4 | 9 | 0.06 | 10 |
| random | 0.01 | 0.01 | 0 | 0 | 0.01 | – |

- **Reproducibility:** R1 here vs CR-2 (identical rule and mode): mean |difference| 0.013, max 0.05. Steering
  measurements repeat.
- **Rules barely differ on average** (0.21–0.25). With 20 prompts per cell, one concept's score has a standard error
  of ~0.1, so these gaps are within noise. Per concept they do differ (Venice R4 0.55 vs R2 0.05; Titanic R2 0.35 vs
  0); best-of-4 averages 0.34, inflated by picking a max. The held-out re-test decides whether those per-concept
  preferences are real.
- **People stay hard:** best is R3 at 0.12, almost all of it Shakespeare 0.25. The prompt+reply run's Harry Potter
  0.50 drops to 0.15 reply-only.
- **Steering the prompt too is consistently stronger** (+0.05–0.10 for every rule). Golden-Gate-Claude style
  demos would likely steer every token; we have been measuring the harder reply-only version.
- The first run's "the layer cut-off was costing us" largely came from prompt steering. Reply-only, R2 gains only
  +0.03 over R1.

## 16 Oct 2026 (after CR-2): held-out steering re-test (ISSUES #11)
`src/steer_retest.py` + `steer_retest_summary.py`. Each rule's chosen setting (neurons, count, strength) re-run on 20
fresh neutral prompts (`prompts/neutral_heldout.json`, no concept-adjacent asks), judged lenient and strict.
Reply-only clamp.

| rule | selection prompts | held-out lenient | held-out strict | held-out fluent |
|---|---|---|---|---|
| R1 late-layer (frozen) | 0.21 | 0.10 | 0.07 | 0.84 |
| R2 any layer | 0.24 | **0.16** | 0.08 | 0.86 |
| R3 measured effect | 0.25 | 0.11 | 0.05 | 0.82 |
| R4 says the answer | 0.24 | 0.13 | 0.07 | 0.73 |
| random | – | 0.00 | 0.00 | 0.80 |

- **Scores roughly halve on fresh prompts.** Two causes we can't separate here: picking the best of 15 settings
  flatters the original number (winner's curse), and the fresh prompts are deliberately less concept-friendly (the
  original set included "Suggest a name for a new coffee shop" and "Recommend a vacation destination"). Steering
  numbers should be quoted with the prompt set they came from.
- **Random neurons stay at exactly 0.00** on fresh prompts: whatever survives is real steering.
- **Per-concept rule preference is partly real:** the selection winner is still (jointly) best held-out in 12/16
  concepts. Venice, Tokyo and violin prefer "says the answer" (R4); Titanic and coffee prefer any-layer (R2).
- **Robust steerers:** coffee (R2 0.65 held-out), the Moon (0.45), Tokyo and Venice (R4 0.35), chess (R2 0.40).
  People: ≤ 0.20 (Harry Potter), Einstein, Shakespeare and Kahlo 0.
- Practical rule for the dial toy: try R1/R2/R4 per concept on a selection set, keep the winner, and report its
  held-out score.

## 16 Oct 2026 (after CR-2): prompt-set noise, CR-2 vs CR-2b (same pipeline, independently written prompts)
`src/replicate_compare.py`. CR-2b reran the CR-2 concepts with recall prompts written from scratch by Luna (same V3
format, same reply-only steering rule and neutral prompts).

| | CR-2 | CR-2b |
|---|---|---|
| real concepts recalled | 19/20 | **20/20** (Sahara now recalled, 10/20 prompts) |
| k50 ≤ 5 (old measure) | 11/19 (95% CI 37–79%) | 10/19 (32–74%) |
| mean nudgeability (joint) | 0.21 | 0.19 |

Per concept:
- **Steering moves by 0.10 on average between prompt sets** (max 0.30: chess 0.35 → 0.05; coffee 0.60 → 0.80, Tokyo
  0.25 → 0.00). The neutral steering prompts are identical, so this comes from different neurons being picked.
  One concept's steering score is ± ~0.1 from the prompt set alone, on top of ± ~0.1 from sampling 20 prompts.
- **k50 is unstable per concept** (Einstein 10 → 50, Venice 3 → 20, chess 50 → 5, coffee 20 → 2). Expected,
  since k50 saturates (ISSUES #27); log-odds is the measure to trust.
- **Top-10 neurons: 4.8/10 exact overlap** (range 1–9) within the same format, versus 0–5/10 across formats. 6.9 of
  each run's top 10 appear in the other's top 50. Stable cores for Shakespeare (9/10), Napoleon (9), democracy (8),
  Golden Gate (7); unstable for Tokyo (1) and the Moon (2).

**Reading:** population statements (how many concepts recall through a handful of neurons, how steerable concepts are
on average) replicate across prompt sets. Single-concept numbers need several prompt sets before they mean much.
"Coffee is steerable, Shakespeare isn't" holds in both runs; "chess 0.35" does not.

## 16 Oct 2026 (after CR-2): coverage wave 2, 40 new concepts in 10 categories (exploratory)
`prompts/concepts_wave2.json` (fixed before running), V3 recall, reply-only clamp, frozen R1 steering rule, same
neutral prompts as CR-2. Summary: `src/wave_summary.py` → `results/w2_summary.txt`.

- **Recalled 37/40.** The 3 misses (evolution by natural selection, rainbows, Egypt's pyramids) score exactly 0 on
  every prompt, the signature of an answer-wording mismatch (ISSUES #29), not missing knowledge.
- **k50 ≤ 5 for 30/37** on the old, saturating measure (which undercounts concentration).
- **Steering (joint, selection prompts): mean 0.28; ≥ 0.5 for 9/37, ≥ 0.25 for 21/37. Random control: mean 0.01,
  max 0.10** (the heart).
- By category, mean steering: body/medicine 0.43 (the brain 0.65, vaccines 0.55), emotion/abstract 0.41 (justice 0.65,
  nostalgia 0.50), nature 0.37, sport 0.31, science 0.30, animals 0.28, art/music 0.25, food 0.24, technology 0.14,
  **people 0.07** (Marie Curie 0.00, Charles Darwin 0.00).
- **People are unsteerable in a fresh sample too.** Across CR, CR-2, CR-2b and wave 2: 8 people, all ≤ 0.25. Places,
  things and ideas reach 0.5–0.8.
- Per-concept numbers carry ± ~0.1 from the prompt set and ± ~0.1 from 20-prompt sampling (CR-2b), and are
  inflated by best-of-15 selection (held-out re-test). Category means are the readable signal.

**Across CR, CR-2, CR-2b and wave 2: 87 recalled concept runs (of 106 attempted, made-up controls included), 57 distinct concepts.** The "handful of neurons + steerable
except people" picture holds across 10 categories.

**Wave 2 update (ISSUES #29 fixed): 40/40 recalled.** `src/answer_discovery.py` let the model answer each recall
prompt itself: "Natural selection" 15/20, "Rainbow" 14/20, "Pyramids" / "Pyramids of Giza" 11/20 (Luna checked each
reply names the concept). With those wordings scored, all three recall (12–14/20 prompts): evolution k50 1, steer 0.05;
rainbows k50 20, **steer 0.70**; pyramids k50 3, steer 0.35. Updated wave totals: k50 ≤ 5 for 32/40; steering mean
0.29, ≥ 0.5 for 10/40; random mean 0.01. Places/people category 0.14 (pyramids 0.35, Great Wall 0.20, Curie and Darwin
0). **Every real concept we have called "not recalled" so far turned out to be our prompt format or answer wording (CR's 9 via V3, Sahara via new prompts in CR-2b, these 3 via discovery), not missing knowledge.**

## 16 Oct 2026 (after CR-2): log-odds v2, robustness (ISSUES #27)
`src/logodds_check.py` (v2) + `src/logodds_summary.py`. Three views per concept, neurons in measured order:
k_mean (mean log-odds drop ≥ 3), k_median (same on the median prompt) and k_flip (P(answer) below 0.5 on at least half
the prompts the model got right at baseline). Random: 50 neurons from the same layers.

| | k_mean ≤ 5 | k_median ≤ 5 | k_flip ≤ 5 | k_flip ≤ 10 | random 50, mean drop |
|---|---|---|---|---|---|
| CR (completion, 11) | 8/11 | 8/11 | 9/11 | 10/11 | −0.01 |
| CR-2 (question, 19) | 19/19 | 18/19 | 14/19 | 16/19 | −0.01 |

- **The mean isn't carried by a few prompts:** the median gives the same picture (CR-2 18/19 ≤ 5; 14/19 ≤ 2).
- **Actually changing the answer usually takes ≤ 5 neurons too** (23/30 runs). The exceptions are the near-certain ones:
  the Moon (P 0.998, flip at 20), Tokyo (0.998, 50), photosynthesis (0.966, 50). Titanic (CR) has too few prompts
  right at baseline (P 0.32) to count flips.
- Corrected count from the first log-odds entry: 15/19 CR-2 concepts (not 11/19) reach a 3-point drop with 1–2 neurons.
- Same pattern as the census and H-DEPTH: a handful of specific neurons do the work; the most entrenched facts have
  backups.

## 16 Oct 2026 (after CR-2): log-odds v2 out of sample: wave 2 (40 new concepts) and CR-2b (new prompts)
Same three views as above, on concept runs the measure wasn't developed on. `results/logodds_summary.txt`.

| run | k_mean ≤ 2 | k_mean ≤ 5 | k_median ≤ 5 | k_flip ≤ 5 | k_flip ≤ 10 | random 50, mean drop |
|---|---|---|---|---|---|---|
| CR (completion, 11) | 0/11 | 8/11 | 8/11 | 9/11 | 10/11 | −0.01 |
| CR-2 (question, 19) | 15/19 | 19/19 | 18/19 | 14/19 | 16/19 | −0.01 |
| CR-2b (question, new prompts, 20) | 18/20 | 20/20 | 20/20 | 16/20 | 17/20 | +0.01 |
| **wave 2 (question, 40 new concepts)** | **38/40** | **40/40** | **40/40** | **36/40** | 37/40 | +0.04 |

- **Out of sample, the claim holds without exception in question format: 79/79 runs are moved by ≤ 5 neurons**
  (log-odds drop ≥ 3), 71/79 by 1–2. The median prompt agrees.
- **Flipping the answer takes ≤ 5 neurons in 66/79 question-format runs.** The holdouts are the near-certain facts
  (P ≥ 0.96): the Moon, Tokyo, photosynthesis, Venice (CR-2b), DNA, Bitcoin, rainbows. Each needs 20–50 neurons or
  doesn't flip within 50, consistent with redundant backup routes for the most entrenched facts.
- Random control: one random neuron happened to matter for poker (0.95 drop; poker's top 5 give 9.5). Every other
  random set stays ≤ 0.24.
- Completion format (CR) remains the less concentrated one (8/11 ≤ 5).

**Top-10 label cleanliness replicates** (pipeline's original strict rule, either pole clean; census baseline under
the same rule 58%): CR-2 130/190 = 68%, CR-2b 133/200 = 67%, wave 2 254/400 = 64%. The neurons that carry a concept
are slightly cleaner than random neurons, consistently across prompt sets and 40 new concepts.

## 16 Oct 2026 (after CR-2): answer discovery as a standard pipeline step (ISSUES #29, #31)
`FastPipeline.discover_answers`: in V3 the model answers every recall question itself (greedy, unsteered). Luna checks
which replies name the concept, and recurring accepted wordings become extra answer forms to score. Recall still needs
P(answer) ≥ MIN_P, so discovery only fixes wording; it can't create recall on its own.

The made-up controls are the safety test, and the first version **failed** it: the fictional Draxwell Bridge
"recalled" 10/20, because the checker accepted names the model built from the question's clue words ("Zanesville
Bridge"). With the fixes (canonical name only, clue-word leak rule, a wording must recur ≥ 2 times; ≥ 3 was
tried and wrongly lost Sahara at 4/20). Final code, fresh run:

| concept | before discovery (CR-2) | discovery v1 | final |
|---|---|---|---|
| Sahara Desert (real) | 0/20 | 6/20 | **6/20 recalled** |
| Elsabeth Quorrin (made up) | not recalled | not recalled | **not recalled** |
| Mount Velkaran (made up) | not recalled | not recalled | **not recalled** |
| Draxwell Bridge (made up) | not recalled | **10/20 ✗** | **not recalled** |

What the model says to the made-up ones is informative in itself: confident real neighbours (Huascarán, Salkantay for
the Peruvian peak; Chomsky, Greenberg for the linguist). It retrieves the nearest real thing rather than inventing a
name.

## 16 Oct 2026 (after CR-2): why are near-certain facts hard to flip? Iterative knockout (no grader calls)
`src/backup_routes.py`. Round 1 ablates the top-10 attributed neurons. Attribution is then re-run *with them ablated*
to find what carries the answer now, 10 more are added, and so on to 50. After each round: mean log-odds drop and
share of baseline-correct prompts flipped. Six hard-to-flip facts vs six easy ones (CR-2 / wave 2 prompts, discovery
off so the answers match the log-odds runs). Random: same-layer neurons, 50.

| concept | baseline log-odds | k=10 drop / flipped | k=20 | k=30 | k=50 | random 50 |
|---|---|---|---|---|---|---|
| the Moon | 9.4 | 8.2 / 42% | 16.5 / 89% | 20.3 / 100% | 22.6 / 100% | 0.01 |
| Tokyo | 11.4 | 3.3 / 5% | 7.9 / 21% | 8.9 / 26% | 14.6 / 68% | −0.12 |
| photosynthesis | 7.8 | 6.4 / 26% | 11.7 / 63% | 15.1 / 89% | 15.5 / 100% | 0.06 |
| DNA | 9.5 | 8.0 / 29% | 12.2 / 65% | 14.1 / 65% | 17.5 / 94% | 0.08 |
| Bitcoin | 12.7 | 6.3 / 16% | 9.9 / 37% | 10.8 / 37% | 15.3 / 47% | 0.02 |
| rainbows | 6.0 | 8.1 / 33% | 16.4 / 100% | 17.5 / 100% | 22.7 / 100% | 0.07 |
| democracy | 2.7 | 15.7 / 100% | | | 21.7 / 100% | −0.02 |
| Napoleon | 4.8 | 19.6 / 100% | | | 25.6 / 100% | 0.01 |
| the violin | 1.5 | 7.2 / 100% | | | 21.9 / 100% | 0.00 |
| jazz | 2.5 | 9.3 / 100% | | | 20.7 / 100% | 0.09 |
| vaccines | 3.3 | 18.2 / 100% | | | 23.0 / 100% | 0.00 |
| the elephant | 3.3 | 8.5 / 100% | | | 20.3 / 100% | −0.03 |

- **Re-attribution flips hard facts much faster than the static ranking.** The Moon fully flips at 30 (static order:
  half at 20), photosynthesis at 50 (static: half at 50), DNA 94% at 50. Tokyo (68%) and Bitcoin (47%) are the
  stubborn ones.
- **"New carriers outside the original top 200" is NOT specific to hard facts.** After round 2, most newly found
  neurons were outside the original top 200 for easy concepts too (violin round 4: 0/10; vaccines round 3: 1/10). Once
  you remove neurons, others take over (self-repair). It happens everywhere, so it doesn't explain the difference.
  (Correction: I first read the Moon's result alone as "the signature of backup routes"; the easy-concept controls
  show the pattern is generic.)
- **What separates hard from easy:** (1) a much bigger confidence margin (log-odds 9.4 on average vs 3.0), and (2) each
  removal dents them less (top-10 drop 6.7 vs 13.1 on average; Tokyo only 3.3). Their support is more spread out,
  not a separate backup system.
- Random 50 neurons: no effect anywhere (≤ 0.1).

## 16 Oct 2026 (after CR-2): are people unsteerable because their neurons just "say the name"? No. (no grader calls)
`src/says_name.py`. For each recalled run (CR-2, CR-2b, wave 2; 79 runs), read each top-10 neuron's direct effect on the
vocabulary (output weights → final norm → unembedding, with its recall sign). It "says the answer" if any token of
the answer ranks in the top 50 of ~152k.

| | runs | "says the answer" neurons in top 10 | steering (joint) |
|---|---|---|---|
| people (8 distinct) | 14 | 2.0 | 0.05 |
| everything else | 65 | 2.6 | 0.29 |

- **Hypothesis refuted.** People's top neurons are not more name-output-like; slightly less. (Counting only the
  first token, the result is 1.3 vs 2.0, the same direction.)
- Within non-people, **more "says the answer" neurons goes with *more* steering** (≥ 3: 0.34, n = 31; ≤ 1: 0.14,
  n = 18). Steering with output-side neurons works, consistent with R4 winning for some concepts.
- Why people resist steering remains open. Candidates: name recall is a lookup ending in tokens that can't
  "colour" a reply the way a topic can; or our neutral prompts give a person no natural way in (a coffee shop
  name invites coffee; nothing invites Einstein). The second is testable by steering with person-friendly neutral
  prompts ("Tell me about someone you admire").

## 16 Oct 2026 (after CR-2): people on person-friendly prompts (no grader calls)
`src/people_prompts.py`. The 8 recalled people steered with the frozen rule's neurons over the usual grid (reply-only),
on 20 person-friendly neutral prompts ("Tell me about someone you admire", `prompts/neutral_person_friendly.json`)
and on the original neutral prompts. On-topic = the name word appears in the reply (no judge; **fluency not
checked**). Baselines: unsteered, and 6 random same-layer neurons at 2×.

| person | friendly: unsteered → best steered (random) | original prompts: unsteered → best (random) |
|---|---|---|
| Napoleon | 0.00 → **0.80** (0.00) | 0.00 → 0.70 (0.00) |
| Marie Curie | 0.20 → **0.60** (0.00) | 0.00 → 0.00 |
| Charles Darwin | 0.00 → **0.55** (0.00) | 0.00 → 0.15 |
| Einstein | 0.10 → 0.35 (0.10) | 0.05 → 0.10 |
| Frida Kahlo | 0.00 → 0.20 (0.00) | 0.00 → 0.00 |
| Harry Potter | 0.00 → 0.05 | 0.00 → 0.00 |
| Shakespeare | 0.00 → 0.00 | 0.00 → 0.00 |
| Mozart | 0.00 → 0.00 | 0.00 → 0.00 |
| **mean** | 0.04 → **0.32** (0.01) | 0.01 → 0.12 |

- **Prompts that give a person a way in roughly triple people-steering** (0.12 → 0.32). Part of "people can't be
  steered" was our neutral prompts: they invite places, food and hobbies but never a person.
- **Quality differs.** Curie and Darwin come out fluent at the weakest setting ("I admire Marie Curie, a pioneering
  scientist…"). Napoleon's 0.70–0.80 is at the strongest setting and the text is damaged ("Napoleon Bonapoleon",
  "拿破仑拿破仑"): pushed in, not written well. On the earlier joint (on-topic **and** fluent) judge, Napoleon was
  0.15, which fits.
- **Shakespeare, Mozart and Harry Potter stay at 0** even when invited, so for some people the frozen rule's neurons
  really don't steer.
- Caveat: the keyword measure counts disfluent mentions. A fluency check on the best cells would cost ~300 Jev calls
  (~$0.01) and is the next step if Jev budget allows.

**People × neuron-picking rule (person-friendly prompts; `src/people_rules.py`, fluency via `src/people_fluency.py`).**
Each cell: name appears / name appears **and** Jev says fluent (best setting of 15).

| person | R1 late | R2 any layer | R3 measured | R4 says the answer |
|---|---|---|---|---|
| Shakespeare | 0.00 / 0.00 | 0.00 / 0.00 | 0.10 / 0.05 | 0.00 / 0.00 |
| Mozart | 0.00 / 0.00 | 0.75 / 0.20 | 0.50 / **0.35** | 0.00 / 0.00 |
| Harry Potter | 0.05 / 0.00 | 0.15 / 0.15 | 0.15 / 0.15 | **0.85 / 0.00** |
| Einstein | 0.35 / 0.30 | 0.35 / 0.30 | 0.35 / 0.30 | 0.65 / 0.30 |
| Frida Kahlo | 0.20 / 0.20 | 0.20 / 0.20 | 0.00 / 0.00 | 0.30 / 0.10 |

- Other rules make the model **say** the name far more (Harry Potter 0.85, Mozart 0.75), but mostly by breaking the
  text ("potter-potter-potter", "Wolfgang Wolfgang Wolfgang"). **Fluent people-steering tops out around 0.3**
  (Mozart 0.35, Einstein 0.30), against 0.6–0.8 for the best places, things and ideas.
- So people are harder to steer *well*, not impossible. Pushing a person in readily damages the text, consistent
  with person concepts ending in name tokens rather than in a topic that can colour a reply. This is a hypothesis.
- The earlier keyword-only person-friendly results (Curie 0.60, Darwin 0.55, Napoleon 0.80) did not save all
  replies, so they have no fluency check; the samples suggest Curie and Darwin are fluent and Napoleon is not.

**People: fluency judged on every setting, clamp vs add** (`people_fluency.py` now judges all 15 settings and picks the best by
name-present-AND-fluent; best over 4 rules shown, so flattered by selection over 60 cells):

| person | clamp: best fluent joint | add (nudge on top of natural activity): best fluent joint |
|---|---|---|
| Mozart | 0.50 (R2, 6 neurons, 1.0×) | 0.55 (R2) |
| Einstein | 0.35 (R4) | **0.55** (R4, 4 neurons, 1.0×) |
| Harry Potter | 0.25 (R4) | 0.25 (R4) |
| Frida Kahlo | 0.20 | 0.25 (R4) |
| Shakespeare | 0.10 (R3) | 0.10 (R3) |
| mean | 0.28 | 0.34 |

- Judging every setting (not just the most-mentions one) finds gentler settings that stay fluent: Mozart 0.35 → 0.50
  under clamp.
- Add mode leans better (+0.06 mean), within the ±0.1 noise of 20-prompt cells. A lean, not a result.
- Bottom line for people: **fluent steering of ~0.25–0.55 is reachable for 4 of 5** with the right rule, prompts and
  mode. Shakespeare resists everything tried (≤ 0.10). The earlier "people can't be steered" was mostly our default
  rule (late layers only) plus prompts that never invite a person.
- Jev cost of both fluency passes: ~$0.09.

**Shakespeare's one dedicated neuron.** Shakespeare's top-10 attributed neurons are mostly generic machinery: non-Latin
scripts, function words, lists of names, software docs, "being solo". Only L30:1702 is labelled "strongly associated
with Shakespeare and his works" (rank 5–8 depending on prompt set). Steering with that **one** neuron
(`src/single_neuron_steer.py`, person-friendly prompts):

| strength (× its peak) | 0.5 | 1 | 2 | 3 | 4 | 6 | 8 |
|---|---|---|---|---|---|---|---|
| clamp: name appears | 0.05 | 0.25 | 0.65 | 0.85 | 0.45 | 0.00 | 0.00 |
| add: name appears | 0.05 | 0.30 | 0.75 | 0.85 | 0.40 | 0.00 | 0.00 |

Best fluent joint: **0.50** (add, 2×) and 0.45 (clamp, 2×), vs ≤ 0.10 for every attribution-based rule. At 3× the text
degrades ("Hamlet's solilo shakespeare akespeare akespeare"), and at ≥ 6× the name disappears entirely (collapse).
**Attribution rank is not the same as steering usefulness.** The neuron whose *label* is the concept steers best,
which suggests a label-matched rule (R5), being tested across concepts.

**R5 label-matched rule vs R1 (frozen), same metric** (`src/label_rule.py`; add mode, reply-only, strengths 0.5–4×;
original neutral prompts, person-friendly prompts for people; score = concept word in reply AND Jev-fluent, best
setting). R5 = the concept's top-10 neurons whose label names it (simple word match, 1–5 neurons).

| concept | R1 | R5 | | concept | R1 | R5 |
|---|---|---|---|---|---|---|
| democracy | 0.75 | 0.75 | | Golden Gate | 0.10 | 0.15 |
| coffee | **0.60** | 0.25 | | photosynthesis | 0.20 | 0.20 |
| the Moon | 0.55 | 0.45 | | violin | 0.10 | 0.15 |
| Napoleon | **0.40** | 0.25 | | **Shakespeare** | 0.00 | **0.30** |
| Venice | **0.35** | 0.15 | | **Titanic** | 0.00 | **0.25** |
| chess | **0.30** | 0.10 | | **Everest** | 0.15 | **0.35** |
| Eiffel | **0.25** | 0.00 | | Harry Potter | 0.05 | 0.05 |
| French Revolution | 0.25 | 0.10 | | **mean (15)** | **0.27** | **0.23** |

- **Not a general improvement** (0.23 vs 0.27, within noise; R1 also gets 3× more settings to pick from).
- **R5 rescues concepts whose attributed top neurons are generic**: Shakespeare, Titanic, Everest (both previously
  ≈ 0 or low). It loses where attribution already finds good neurons (coffee, chess, Eiffel).
- Shakespeare's 3 label-matched neurons give 0.30 vs 0.50 for his single dedicated neuron: the word match also
  pulled in a generic neuron whose label merely mentions him. Better matching (Luna: "is this neuron *about* X?")
  would be cleaner.
- **Method takeaway for the dial toy:** offer several candidate neuron sets per concept (R1, R2, R4, R5), score each
  on a selection prompt set for named-and-fluent, keep the winner, report held-out. No single rule is best.

**R5b: Luna-judged label matching ("is this neuron *about* X?") instead of word search** (`src/label_match_luna.py`,
190 Luna calls; steering re-run on the 7 concepts whose set changed, same metric):

| concept | R5 word match | R5b Luna "about" |
|---|---|---|
| democracy | 0.75 | 0.65 |
| the Moon | 0.45 | 0.25 |
| Mount Everest | 0.35 | 0.10 |
| Shakespeare | 0.30 | **0.45** |
| Titanic | 0.25 | 0.00 |
| French Revolution | 0.10 | 0.05 |
| Eiffel Tower | 0.00 | 0.05 |
| **mean** | **0.31** | **0.22** |

**Negative: the stricter, cleaner-looking selection steers worse.** A neuron being precisely "about" the concept
doesn't predict steering usefulness. The broader word match kept parts that help (Everest's early "mountain" neurons,
Titanic's L10:1252). Luna also rejected all of Golden Gate's gold/bridge/landmark parts. Only Shakespeare benefits
(dropping a generic neuron). **Lesson for the dial: choose neuron sets empirically per concept (try several, keep
the best on selection prompts, report held-out); no labelling heuristic beats measuring.**

## 16 Oct 2026 (after CR-2): a free local fluency check, validated against Jev (budget relief)
`src/fluency_proxy.py`, 6,000 steered replies Jev had already judged ("fluent, grammatical English?",
P(yes) ≥ 0.5 = fluent). Features computed locally:

| feature | AUC vs Jev |
|---|---|
| distinct-token fraction (repetition) | **0.907** |
| mean log-prob under the unsteered model | 0.582 |
| share of non-ASCII characters | 0.461 |

- Rule fitted on half, tested on the other half: **distinct-token fraction ≥ 0.64 AND mean log-prob ≥ −2.3 AND
  non-ASCII < 3% → fluent.** Held-out agreement with Jev **88.7%** (always-"fluent" baseline 61.2%); precision 0.89,
  recall 0.93.
- Log-prob alone is weak because language models find their own repetition loops likely ("Albert Albert Albert"),
  which is exactly the failure mode steering produces.
- Use: free fluency for the dial's auto-selection and quick sweeps. Pre-registered tests keep the Jev judge.
  Validated only on our steering replies (people and label-rule runs).

## 16 Oct 2026 (after CR-2): auto-selected steering at scale, judge-free (59 concepts)
`src/auto_steer_scale.py` → `results/auto_steer_summary.txt`. For every recalled concept (CR-2 19 + wave 2 40):
candidate sets R1/R2/R4 × 2/4 neurons × 0.5–2×, add mode, reply-only. **Select** on the 20 original neutral prompts;
**score** that one setting on 20 held-out neutral prompts. Score = a distinctive answer word appears AND the reply
passes the free fluency check (88.7% agreement with Jev). No judge API calls.

| held-out (20 fresh neutral prompts) | mean over 59 concepts |
|---|---|
| **auto-selected per concept** | **0.203** |
| fixed default (R1, 4 neurons, 1×) | 0.106 |
| random neurons (4, 2×) | 0.001 |
| unsteered | 0.001 |
| (selection-prompt score of the chosen setting) | (0.295) |

- **Choosing per concept doubles held-out steering** over the fixed rule: better in 31/59, worse in 4, tied in 24.
  Biggest gains: Tokyo 0.00 → 0.55 (R4), steam engine 0 → 0.50, periodic table 0 → 0.45, justice 0.15 → 0.75.
- **Rules chosen:** R1 31, R4 19, R2 9. No single rule wins.
- Held-out ≥ 0.5 for 7/59 concepts, ≥ 0.25 for 22/59, 0 for 16/59. People average 0.01 on these generic prompts
  (they need person-friendly prompts, see above); non-people 0.23.
- Winner's-curse shrinkage from selection to held-out: 0.295 → 0.203, about a third.
- The baselines are essentially zero, so every point here is steering. Caveats: keyword on-topic is stricter than the
  lenient judge (it needs the word itself), and the fluency check is a proxy.

**Check of the judge-free score against Jev** (`src/auto_steer_jev_check.py`, the 1,180 auto-selected held-out replies,
Jev lenient on-topic + fluent, $0.02): per-concept Pearson r = **0.65**; means 0.203 (judge-free) vs **0.161 (Jev)**;
concepts ≥ 0.25: 22 vs 19. **The free score overstates.** It counts word fragments ("steampunk steampunk…",
"Steampara" for steam engine) and bare word intrusions ("a gentle, soothing revolution of nature" for the French
Revolution), and partial loops at the end of long replies can pass the repetition check. Biggest gaps: steam engine
0.50 vs 0.00, periodic table 0.45 vs 0.05, French Revolution 0.40 vs 0.05. **Fine for quick picking in the dial; not
for reported numbers.** The auto-vs-default comparison is being re-judged with Jev on both sides.

**Auto vs default, Jev on both sides** (default's held-out replies regenerated, greedy so identical to the scale
run; Jev lenient on-topic AND fluent; $0.03 total): **auto 0.161 vs default 0.087** over 59 concepts. Auto better in
31, worse in 7, tied in 21. **The doubling holds under the real judge** (~1.85×). The free score inflated both sides'
levels but not the comparison.

**Is a concept's preferred neuron set stable across prompt sets?** (`auto_steer_scale.py` on CR-2b, prompts written
from scratch, vs CR-2; judge-free score; `results/auto_steer_stability.txt`): **the same rule is chosen for 13/19
concepts** (chance ≈ 6), the same rule and neuron count for 9. Per-concept held-out score correlates **r = 0.76**
across prompt sets (mean |difference| 0.10). Stable: Tokyo R4 0.55/0.60, coffee R4 0.50/0.50, the Moon R4 0.45/0.45,
Golden Gate R1 0.35/0.35. Auto vs fixed default on CR-2b: 0.25 vs 0.05. **The preferred neuron set is mostly a
property of the concept, not prompt noise**, so the dial can pick once per concept and reuse it.

**Audit of MORNING.md headline numbers** (recomputed from the result files, `results/morning_audit.txt`): question
format 79/79 ≤ 5 neurons, 71/79 ≤ 2, flips ≤ 5 in 66/79; completion 8/11, all within 20; wave 2 40/40; auto vs default
0.161 vs 0.087 (31/7/21); rule stability 13/19; top-10 overlap 4.8/10. All match. One stale count corrected: **60**
distinct concepts recalled (the earlier "57" predates the three answer-discovery re-runs; likewise "87 recalled
concept runs" is now 90).

## 16 Oct 2026 (early morning): coverage wave 3, 40 new concepts in 10 new categories, judge-free
`src/wave_free.py` (`prompts/concepts_wave3.json`, fixed before running; one overlap swapped before running).
Recall with answer discovery; log-odds concentration; auto steering scored by the free check (known to inflate
levels; it ranks like Jev). No Jev at all.

- **Recalled 39/40.** The oak tree was first lost to our leak filter (ISSUES #32, fixed, 17/20 after). The robot
  stays unrecalled: its questions describe specific robots and the model names them (Roomba, robotic arm).
- **Concentration:** k_mean ≤ 5 for **39/39** (≤ 2 for 34); median prompt 38/39; answer flips with ≤ 5 neurons
  in 35/39. Random 50 neurons: mean drop 0.00.
- **Steering (held-out, judge-free):** auto 0.188 vs fixed default 0.088 (better 20, worse 5); random 0.013. Best:
  speed of light 0.85, helicopter / friendship / gravity 0.55, courage / Steve Jobs 0.50. By category: science/math
  0.41, vehicles 0.28, abstract 0.28, fiction 0.25, … mythology 0.06, landmarks 0.05. Modern people 0.17 on
  person-friendly prompts (Steve Jobs 0.50).

**Running total, question format: 118/118 concept runs are moved by ≤ 5 neurons** (CR-2 19, CR-2b 20, wave 2 40,
wave 3 39), with random neurons at ≈ 0 every time. **99 distinct concepts recalled**, and **every "not recalled"
except the robot traced to our prompts, wording or filters.**

## 16 Oct 2026 (morning): coverage wave 4, 40 concepts incl. deliberately generic ones, judge-free
`src/wave_free.py` (`prompts/concepts_wave4.json`). Crashed once at "cheese" (Luna output-format drift, ISSUES #33,
fixed) and resumed.

- **Recalled 40/40**, including professions (a doctor, a firefighter, a chef, an astronaut), colours, weather and
  everyday objects. "The robot" (wave 3) was a one-off, not a limit for generic concepts.
- **Concentration:** k_mean ≤ 5 for 39/40 (≤ 2 for 37); the clock needs 10. Median prompt 39/40. Flips ≤ 5 for
  36/40; the holdouts are near-certain facts again (the colour red and Paris at log-odds > 10; the clock).
  Random 50 neurons: mean −0.01.
- **Steering (held-out, judge-free):** auto 0.299 vs fixed default 0.169 (better 23, worse 5); random 0.006.
  Everyday objects steer best (0.56), then food 0.46 and historical events 0.39; sport is lowest (0.16).

**Running totals (`results/running_totals.txt`, computed from the files): 158 question-format concept runs; moved by
≤ 5 neurons in 157 (≤ 2 in 142), all within 10; answer flipped with ≤ 5 in 137. 139 distinct concepts recalled.**

## 16 Oct 2026 (morning): do related concepts share neurons? (composition check; no grader calls)
`src/shared_parts.py` → `results/shared_parts.txt`. Top-50 attributed neurons for 10 related pairs (each concept on
its own saved recall prompts), vs the same 20 concepts re-paired at random (200 unrelated pairs). Neurons in the
top 50 of ≥ 3 of the 20 concepts (18 of them) count as generic machinery and are reported separately.

| pair | shared (all) | shared (non-generic) |
|---|---|---|
| steam engine ~ printing press | 12 | 8 |
| the Moon ~ the Moon landing | 4 | 4 (incl. the Moon neuron L31:9683) |
| colour red ~ colour blue | 6 | 4 |
| dolphin ~ shark | 3 | 3 |
| coffee ~ tea | 7 | 2 (incl. the coffee neuron L29:2750) |
| chess ~ chess grandmasters | 2 | 2 |
| snow ~ thunderstorms / Golden Gate ~ gold / Zeus ~ Thor | 2 / 1 / 3 | 1 / 1 / 1 |
| Eiffel Tower ~ Paris | 0 | 0 |
| **related mean** | **4.0** | **2.6** |
| **unrelated mean (95th pct)** | 1.0 (4) | **0.13** (1) |

- **Related concepts share specific parts ~20× more than unrelated ones** once generic machinery is removed. 6/10
  related pairs exceed the unrelated 95th percentile. This is direct support for the essay's composition picture:
  concepts are built from reusable parts, and neighbours reuse each other's.
- Eiffel Tower ~ Paris share nothing in the top 50 (each recalled on its own prompts). Possibly the Eiffel prompts
  lean on landmark/structure parts and the Paris prompts on country/capital parts. Not chased.

**Lexical control for shared parts** (LITERATURE.md follow-up 1, prompted by Ait Hou & Hwa 2026; `src/shared_parts_lexical.py`
→ `results/shared_parts_lexical.txt`). For each of the 26 non-generic neurons shared by a related pair, we read its
top-20 atlas firings on both poles. **None is a word-form detector**: none fires on the pairs' shared word in ≥ 50% of
its top firings (0/26). Two caveats:
- **"Not lexical" isn't proven "meaning".** The Moon-pair neurons fire *around* moon contexts (the word is nearby in
  55–75% of top examples) but on other tokens, which fits a concept neuron. The 8 steam engine / printing press neurons
  fire mostly on punctuation and symbols (":", "=", "|"), the summary-motif pattern. They could be shared topic
  neurons or shared formatting machinery; this test can't tell.
- **Bugs caught on the way:**
  - The first run reported 0/26 lexical because token extraction silently failed (the atlas renderer omits the
    [[ ]] marker when `after=0`). An assertion now fails the run if any marker is missing.
  - That assertion then caught a newline token that the pattern didn't handle.

**Reading:** the part-sharing result survives the lexical control. Whether every shared neuron carries meaning needs
a case-by-case look.

**The same knockouts in plain probability** (for comparison with Dai et al. 2022's ~29% drop with ~4 knowledge
neurons; `results/prob_view.txt`; per-prompt data from the log-odds v2 runs; 90 concept runs):

| run | top-1 neuron off: mean P(answer) | top-5 off | random 5 off | correct prompts flipped (top-5) |
|---|---|---|---|---|
| CR completion (11) | 0.567 → 0.348 (−39%) | → 0.104 (−82%) | 0.566 | 78% |
| CR-2 question (19) | 0.890 → 0.638 (−28%) | → 0.340 (−62%) | 0.891 | 67% |
| CR-2b question (20) | 0.904 → 0.655 (−28%) | → 0.352 (−61%) | 0.903 | 70% |
| wave 2 question (40) | 0.845 → 0.508 (−40%) | → 0.183 (−78%) | 0.844 | 84% |

A single measured top neuron gives about the drop Dai et al. report for ~4 neurons. Five neurons remove 61–82% of the
answer's probability and flip most correct answers, while five random same-layer neurons change nothing (to 3
decimals).

## 16 Oct 2026 (morning): human spot-check of the graders (ISSUES #16, last part)
James blind-rated 40 (description, text) cards (a private rating page; the cards, order, answer key and ratings are in `results/spotcheck_*`; 20 = the neuron's own top firing, 20 = random
text). The middle option turned out to mean "unsure" in practice, so it is treated as an abstention. The first 6
cards may be order-influenced (ISSUES #34) and are reported with and without.

| | all 40 | excluding first 6 |
|---|---|---|
| human unsure | 8 (20%), all random cards | 8 (24%) |
| **human (confident) right about own vs random** | **88%** (28/32) | 85% (22/26) |
| **Jev right about own vs random** (≥ 0.5 = fits) | **88%** (35/40) | 88% |
| human–Jev agreement on confident cards | 78% | 77% |

- **Jev matches a careful human on telling a neuron's real firings from random text** (both 85–88%). The census grader
  is not worse than a person at this task.
- **All 4 confident human errors are "clearly fits" on random text**: the description happened to fit an unrelated
  token. This is the over-fitting-to-random-text effect the census's random-snippet check is built to penalise.
- **Jev is not less certain where the human was unsure** (distance from 0.5: 0.37 unsure vs 0.34 confident). It doesn't
  share human hesitation.
- **Limits:** one rater, 40 cards; "own example" is ground truth for *source*, not for whether the description is
  right; Luna had not graded the rebuilt cards, so there's no Luna comparison.

**Independent reproduction (explainer session, separate process and code path, 16 Oct ~09:15):** re-deriving the
neurons from the saved prompts gave the same top-10 neurons (10/10) for the Eiffel Tower, Einstein and chess, the
same k (log-odds drop ≥ 3) each time, and an Eiffel knockout curve matching `logodds_cr2_V3_v2` to two decimals.
**Full run (138 concept runs):** the top-10 neurons matched 10/10 wherever a saved list exists, and k matched in 137/138.
The Acropolis gave 2 vs a saved 3; the explainer batches its single-neuron knockouts, so small bf16 numerical
differences can reorder near-ties. The pipeline is reproducible on the same hardware, with near-ties as the only
source of variation.

## 16 Oct 2026 (11:15): CR-3, confirmatory replication, scored against frozen predictions (PREREG CR-3)
Frozen at commit `a0c3702` before any CR-3 data: protocol, 40 mechanically picked never-used concepts (10 categories)
plus 4 new made-up controls, and the scoring script. The Jev check script was added at `edf7668`, also before data.
Run 10:06–11:15 (`results/concepts_cr3/`, `results/cr3.log`). Output of `src/cr3_score.py` (`results/cr3_score.txt`):

| prediction | result | numbers |
|---|---|---|
| P1 recall ≥ 90% of real; 0/4 made-up | **PASS** | 40/40 real recalled; made-up 0/4 |
| P2 k_mean ≤ 5 ≥ 90%; k_median ≤ 5 ≥ 80%; random < 0.5 ≥ 95% | **PASS** | 98%; 98%; 100% |
| P3a k_flip ≤ 5 ≥ 70% | **PASS** | 85% |
| P3b hard-to-flip have higher baseline log-odds | **PASS** | 7.66 vs 4.04; Mann–Whitney U = 170, p = 0.005 |
| P4a auto ≥ 1.5× default; random < 0.05 | **PASS** | 0.237 vs 0.097 (2.44×); random 0.006 |
| P4b Jev sample: auto ≥ default | **PASS** | 0.095 vs 0.045 (10 concepts, 380 calls, $0.006) |
| P5 people > 0 and below non-people | **FAIL** | people 0.425 (n = 4) vs non-people 0.217 |

**P5 failed in the unexpected direction.** Exploratory follow-up (after scoring; `src/cr3_people_baseline.py`,
`results/cr3_people_baseline.txt`, `results/cr3_people_jev.txt`):
- **Not the prompts supplying the name:** unsteered, none of the four names appears on the held-out person-friendly
  prompts (0.00 each).
- **Not a loose keyword:** a strict surname-only check gives identical steered rates.
- **Not the free fluency check:** Jev scores the steered people replies 0.20–0.60 on-topic and fluent (mean 0.40;
  fluency 0.80–1.00).

**So people do steer, given a question that invites a person.** The prediction itself was mis-specified: it compared
people on person-friendly prompts with other concepts on generic prompts. The robust reading revises an earlier
headline:
- "people are hard to steer" was mostly prompts that never invite a person;
- some individuals (Shakespeare, Mozart, Harry Potter) resist even then;
- others (Cleopatra, Lincoln, Gandhi, Martin Luther King Jr.) do not.

**Confirmed (pre-registered, on 40 never-used concepts):**
- every real concept recalled and no made-up one;
- recall depends on ≤ 5 neurons for 98%, random neurons never matter;
- answers flip with ≤ 5 neurons for 85%, and the resistant ones are the most confident;
- per-concept steering selection beats a fixed rule (2.4×; confirmed in direction by Jev).

## 16 Oct 2026 (afternoon): EXPLORATORY pilot, swapping out the "name-start" neuron (explainer session)
Prompted by Serena Williams' steering collapsing into "Ser Ser Ser" (L34:9038 writes "Ser"). Scripts:
`explainer/gen/no_name_start.py` and `explainer/gen/compare_nostart.py`; outputs in `explainer/data/raw_nostart/`.
Qwen3-8B, no API; same 8 held-out prompts, strengths 0.25–3×, auto-selected sets, add mode, reply-only.
- **"Name-start" neuron:** one of a steered neuron's top-8 direct-output tokens matches the answer's first token. 114 of
  138 steered sets contain one. The match is broad: whole-word cases like coffee or gold count.
- **Arms:** (a) the original set; (b) the name-start neuron swapped for the next-best neuron from the same rule; (c) the
  name-start neuron swapped for a random same-layer neuron.

Pilot, 10 concepts (coffee, Eiffel Tower, Golden Gate Bridge, Harry Potter, Mount Everest, the Moon, Tokyo,
Shakespeare, Serena Williams, Taylor Swift); crude counts, arms a / b / c:

| | (a) original | (b) next-best swap | (c) random swap |
|---|---|---|---|
| name-repetition loops at ≥ 1.5× | 177 | 79 | 71 |
| best fluent concept-naming, mean of 8 | 4.4 | 2.5 | 2.7 |

Reading:
- **The name-start neuron drives the repetition loops, and also does much of the naming.** Without it, coffee and
  Tokyo fall to ~0 named. For one-word names, the "name-start" neuron *is* a core concept neuron (coffee's is the coffee
  neuron L29:2750; Golden Gate's is the gold part).
- **The replacement adds little:** next-best ≈ random (b vs c: better 2, worse 2).
- **For people, removing it lets the parts through:**
  - Serena at ×1: "Marie Curie… women in STEM" (women / female-name neurons).
  - Serena at ×2: "Serena Venus… served… Grand" (tennis, her sister).
  - Serena at ×3: loops over "Venus / Shar(apova) / Nad(al)".
  - Taylor Swift: the loop moves from "Taylor" to "artists".
  - This is composition visible in a single concept.

Caveats:
- n = 10, crude scoring.
- The loop metric fails for single-letter first pieces (the Eiffel Tower's "E"), so that row ignores those.
- **Arm (c) controls for "the replacement matters", not for "any change to the set helps".** That needs swapping a
  non-name neuron, which wasn't run.
- Exploratory only, and not on the page unless James agrees.

## 16 Oct 2026 (afternoon): ATTENTION study, steps 1–2 (EXPLORATORY; research branch, not yet approved for publication)
`src/head_logodds.py` + `src/head_summary.py` (`results/head_summary.txt`). 59 concepts (CR-2 + wave 2), the same
prompts and answers as the neuron log-odds runs. Heads are ranked by attribution (top 50), re-ranked by measured
single-head knockout (zeroing the head's output at every position), then knocked out cumulatively. The same three
views as for neurons; control: random heads from the same layers.

| | heads | neurons (same concepts) |
|---|---|---|
| k_mean ≤ 2 | 20/59 | 53/59 |
| k_mean ≤ 5 | 49/59 | 59/59 |
| k_mean ≤ 10 | 59/59 | 59/59 |
| k_flip ≤ 5 | 33/59 | 50/59 |
| flips at all within 50 | 53/59 | 59/59 |
| mean log-odds drop, top 5 | 5.35 | 8.88 |
| random 5 | +0.12 | ≈ 0 |

- **Recall also depends on a handful of attention heads (≤ 10 for every concept), but less sharply than on neurons,
  and heads alone flip answers less often.**
- **Random-head control:** 5 random heads do nothing. 50 random heads are not null for heads (mean +1.21; up to +9.1
  for the heart, 8.9 infinity, 5.8 the Moon, 5.6 the internet). Fifty heads is 4% of all 1,152, so a random set can
  hit important shared heads or degrade the model generally. Use the 5–10 level as the control for heads.
- **Shared vs specific heads (step 2):**
  - A small set of heads recurs across concepts: L20H9 in the top 10 of 36% of concepts, L19H29 34%, L35H25 27%,
    L35H24 25%, L23H10 25%. Chance would be ~1%.
  - 11 heads appear in ≥ 11 concepts' top 10, and on average 2.7 of each concept's top-10 heads are these shared ones.
  - The top heads sit mostly in layers 17–23 and 34–35, with some at 0 and 9.
  - So the same split as neurons: generic "fact moving" machinery plus concept-specific heads.
- Steps 3 (where they look) and 4 (heads → neurons) are next.

**Attention study, step 3: where the important heads look** (`src/head_looks2.py`, `src/head_looks_summary.py`; every
recall prompt; attention from the answer position, split into sinks, chat template and question content).
- Most top heads put most of their attention on "sink" tokens: shared heads 51%, specific heads 64%. Attention
  weights alone say little about them.
- **One shared head is a clear clue reader: L23H10.** It's in the top 10 of 25% of concepts, and at the answer
  position it puts 82–93% of its attention on the question's most telling words:
  - Marie Curie: "Pierre / Nobel / -onium"
  - pizza: "toppings / cheese / dough"
  - Egypt's pyramids: "Nile / burial"
  - Golden Gate: "Francisco / Strauss / crossing"
  - Bitcoin: "BTC"
- Some concept-specific heads do the same: Everest L26H17 ("mountain / peak / summit"), insomnia L25H26 ("sleep /
  asleep"), the Amazon L24H26 ("America / Brazil").
- **Bug caught on the way:** the first pass counted the fixed "Answer with just the name." suffix as question
  content. Token-overlap detection now covers it, with an assertion. Fixed and rerun before any summary.
- Caveat: attention weight is a first look; the value vectors also matter.

**Attention study, step 4: do the key heads feed the concept neurons?** (`src/heads_to_neurons.py`,
`results/heads_to_neurons_summary.txt`). For each concept, its top-5 measured heads are switched off, and the signed
activation of its top-10 concept neurons is read at the answer position, against 5 random same-layer heads.
- **Sanity check passed for every concept:** neurons in layers below the switched-off heads were exactly unchanged,
  as they must be.

| 58 concepts (infinity: no downstream neurons) | top-5 heads off | 5 random heads off |
|---|---|---|
| downstream concept-neuron activation (median, total-ratio) | 0.93× | 0.99× |
| concepts cut below 0.8× | 12/58 | 0/58 |
| lower than with random heads | 48/58 | – |

- **The key heads feed the concept neurons consistently but only partly.** Most concepts reach their neurons by several
  routes. The most head-dependent: Mozart 0.35×, Einstein 0.45×, the internet 0.53×, the periodic table 0.59×, the
  Moon 0.64×.
- **People are not more head-dependent** (median 0.97×, n = 8). This **refutes** a hypothesis we floated from the first
  19 concepts, where Mozart, Einstein and Shakespeare looked strongly head-dependent; with all 8 people the pattern
  disappears.
- **Bugs caught on the way:**
  - The first summary averaged per-neuron ratios, which blow up near zero baselines; it now uses total and median.
  - A background run used relative paths from the wrong folder and silently did nothing; it now uses absolute paths.
  - A divide-by-zero for a concept with no downstream neurons; it now saves after every concept.
- **Picture so far (exploratory):** a shared mid-layer clue-reader head (L23H10) gathers the question's clue words, and
  concept neurons a few layers later assemble the answer. Many routes carry the information, so no single set of five
  heads is essential for most concepts.
- **Not done:** path patching (cutting only the head → neuron route), which would show *what* message each head carries.

**Attention study, deeper: who feeds each concept neuron?** (`src/feeders.py`, `src/feeders_summary.py`,
`results/feeders_summary.txt`). An exact decomposition of each concept neuron's input at the answer position into its
sources: the token embedding, every earlier head (36 × 32) and every earlier MLP neuron. It treats the per-token
normalisation scale as fixed. Six concepts (Golden Gate, Eiffel Tower, coffee, Mozart, the Moon, Marie Curie), their
top-5 neurons, every recall prompt.
- **Checks:** the sources reproduce the neuron's input to within 0.9% in every concept, and the model's own gate and
  value inputs to within bf16 rounding.
- First-order contributions sum to ~2× the activation (the neuron multiplies two inputs), so read them as shares.

Findings (exploratory, 6 concepts):
1. **Entry neurons are fed mainly by attention.** The first concept neuron in each chain gets most of its input from
   heads: Eiffel L26:11492 71%, the Moon L20:2258 74%, coffee L29:2750 59%, Golden Gate L28:6711 52%, Curie L28:5740
   52%.
2. **Later concept neurons are fed mainly by the concept's own earlier neurons (a chain).**
   - The Moon: L31:9683 supplies ~100% of L32:6316's net input and 70% of L33:2482's.
   - Golden Gate: L28:6711 supplies 48% of L31:9611 (via L26:11492 → L28:6711 → L31:9611 → L34:5510).
   - Eiffel: L26:11492 supplies 40% of L29:12094.
3. **A recurring "feeding team" of mid-late heads serves many concepts.**
   - L26H17 feeds Golden Gate, Eiffel, coffee, the Moon and Mozart.
   - L24H19 feeds Golden Gate, Eiffel, coffee and Curie.
   - L28H4 and L28H6 push the entry neurons up.
   - **L28H7 consistently pushes them down** (−9 to −12), like a regulator next to the accelerators.
   - L24H26 and L23H10 also feed (Mozart, Curie, Eiffel).
4. **Feeding heads are mostly not the knockout-important heads:** 0–3 of each neuron's top-8 feeders appear in the
   knockout top 10. The knockout-important shared heads (L20H9, L19H29, L35H25) barely feed these neurons, so they
   presumably act through other routes (e.g. more directly towards the answer). That is why knockout alone couldn't
   show the head → neuron relationship.

Not yet tested:
- **Causation:** this decomposition is exact arithmetic about one forward pass, not a causal test. Next: cut only
  specific head → neuron routes.
- **What the feeding heads carry, and from which words.**

**Causal test of head → neuron routes** (`src/route_cut.py`, `results/concepts_v3/route_cut_v3.json`). For each
concept's entry neuron, at the answer position, we removed only the contribution of its top-5 positive feeding heads
to that one neuron's inputs; those heads still write everywhere else. The normalisation scale is held at its
unmodified value. Compared with random-head routes, with switching the neuron off, and with switching the 5 heads off
entirely. Recall = log-odds drop of the answer's first token.

| concept (entry neuron) | neuron: base → route cut | recall drop: route cut | random routes | neuron off | heads fully off (neuron) |
|---|---|---|---|---|---|
| coffee (L29:2750) | 65.8 → 21.2 | **0.71** | 0.02 | 0.92 | 0.36 (40.3) |
| Golden Gate (L28:6711) | 56.4 → 27.5 | **0.42** | 0.02 | 1.20 | 1.81 (31.9) |
| Eiffel (L26:11492) | 27.0 → 8.7 | **0.38** | −0.02 | 0.64 | 0.21 (10.2) |
| the Moon (L31:9683) | 119.2 → 75.1 | **0.91** | 0.06 | 3.05 | 2.27 (83.0) |
| Mozart (L30:8885) | 48.3 → 38.8 | **0.33** | −0.04 | 2.39 | 1.01 (42.6) |

- **The specific head → neuron links are causal for recall.** Cutting only these routes lowers the neuron by 20–70%
  and recall by 0.33–0.91 log-odds; random routes do nothing (|drop| ≤ 0.06).
- **How much of the neuron's effect flows through these routes varies:** coffee 77%, Eiffel 59%, Golden Gate 35%, the
  Moon 30%, Mozart 14% (route cut ÷ neuron off).
- **Whole-head knockout can understate a head's role.** For coffee and Eiffel, switching the 5 heads off entirely
  lowers recall less than cutting only their route into the neuron, and the neuron recovers more (coffee 40.3 vs 21.2).
  The likely cause is the normalisation step: removing whole heads shrinks the residual, normalisation rescales the
  remaining inputs upward, and the neuron partly recovers. That is a known self-repair mechanism (Rushing & Nanda 2024
  attribute ~30% of self-repair to normalisation scaling). It plausibly explains why knockout-ranked heads and
  feeding heads differ (finding 4 above).
- **Caveats:**
  - Five concepts.
  - The cut is at the answer position only.
  - First answer token only.
  - The normalisation-scale explanation is inferred, not separately tested.

**What the feeding heads deliver, and from which words** (`src/route_sources.py`,
`results/concepts_v3/route_sources_v3.json`). Each feeding head's push into the entry neuron, split exactly by source
position (attention weight × that position's value vector). Check: the per-position pieces rebuild each head's actual
output to within 0.2%. Five concepts × their top-5 positive feeding heads, all recall prompts.

Two kinds of feeding head:
1. **Meaning carriers** (deliver from the question's words):
   - **L26H17 carries category words:** Eiffel "structure / monument / landmark / attraction", the Moon "body /
     satellite / object", Golden Gate "crossing / span / structure", Mozart "composer".
   - **L24H19 carries category words too:** coffee "grounds / drink / beverage / press", Golden Gate "suspension /
     span".
   - **L23H10 carries specific clue words:** Golden Gate "Strauss / Francisco", Eiffel "Paris", the Moon "satellite /
     Earth / sky".
   - **L24H26 carries very specific clues for Mozart:** "-zburg" (Salzburg), "-ute" (The Magic Flute), "-aro" (Figaro).
   - **L26H20 for the Moon:** "months / eclipse".
2. **Content-free "gain" heads** (deliver almost entirely from formatting tokens: line breaks, the start-of-turn
   marker, "?"):
   - L28H4, L28H6, L29H25, L30H17 and L25H15 push the entry neuron up regardless of the question's content.
   - L28H7 (previous entry) consistently pushes it down.

**Working sketch of recall (exploratory, 5–6 concepts):**
1. Category heads (L26H17, L24H19) and clue heads (L23H10, L24H26) carry the question's meaning to the answer position.
2. An entry concept neuron combines them ("structure + monument + Paris → Eiffel Tower").
3. Format-driven heads set the gain (L28H4 / L28H6 up, L28H7 down).
4. The concept amplifies through a chain of its own neurons.
5. Late layers turn it into the answer word.

Every arrow in steps 1–2 rests on exact decomposition, and the head → entry-neuron links were shown to be causal for
recall (route cut). Still untested: steps 4–5 causally, the generality beyond 5–6 concepts, and whether L26H17 /
L24H19 behave as "category readers" in general.

**Generality of the feeder picture: 59 concepts, top-3 neurons each** (`src/feeders.py` with precomputed weight
products, regression-tested against the earlier run to 4 decimals; `src/feeders_general.py`,
`results/feeders_general.txt`). 128 active concept neurons; every decomposition rebuilds the neuron input to ≤ 1.1%.

| concept-neuron layer | n | median share of input from heads | median share from the concept's own top-10 neurons |
|---|---|---|---|
| 0–15 | 2 | 0.95 | 0.00 |
| 16–23 | 11 | 0.82 | 0.00 |
| 24–29 | 31 | 0.54 | 0.00 |
| 30–35 | 84 | 0.29 | 0.19 |

- **The deeper the concept neuron, the less it depends on heads and the more on earlier neurons.** Concept neurons feed
  each other mainly in layers 30–35.
- **The feeding team is general** (chance ≈ 0.5% per head): L23H10 feeds 40% of all concept neurons, L26H17 32%,
  L24H19 32%, L30H17 27%, L29H27 22%, L28H6 20%, L28H4 20%, L26H20 18%.
- **The brakes are general:** L28H7 pushes down 42% of all concept neurons, L25H14 16%.
- **Bug caught on the way:** the first 59-concept run did nothing. It loaded the model and exited, because the
  concept-list builder passed a bash-style path to Windows Python. It is now run with Windows paths and an empty-list
  guard.

**What each team head carries, across 50 concepts** (`src/route_sources.py` with TEAM_HEADS,
`src/team_sources_summary.py`, `results/team_sources_summary.txt`). For each concept's strongest active concept
neuron in layers ≥ 30, each team head's push into it is split exactly by source position into question words,
formatting tokens and the fixed suffix. Value-vector reconstruction is checked per head and prompt (< 2%).

| head | median net push | question share | format share | words most often carried |
|---|---|---|---|---|
| L23H10 | +3.18 | **0.92** | 0.08 | specific clue fragments |
| L26H17 | +1.63 | **0.85** | 0.14 | **category words:** name, scientist, landmark, animal, game, beverage, system, monument |
| L24H16 | +0.98 | 0.82 | 0.18 | clues (beverage, Atlantic, gravity, photo) |
| L26H20 | +1.82 | 0.80 | 0.20 | clues (electric, morning, voting, beverage) |
| L24H19 | +2.22 | 0.75 | 0.25 | category words (scientist, sport, beverage, light, gravity) |
| L27H29 | +1.62 | 0.54 | 0.45 | mixed (scientist, physicist, landmark, city) |
| L29H24 / L29H27 / L30H17 | +0.5 to +2.3 | 0.26–0.37 | 0.63–0.74 | mostly format |
| L28H4 / L28H6 | +1.8 each | 0.11–0.16 | **0.84–0.89** | format-driven **accelerators** |
| **L28H7** | **−3.27** | 0.08 | **0.92** | format-driven **brake** |
| L25H14 | −1.01 | 0.03 | **0.97** | format-driven brake |

- **Two separate roles:**
  - **Meaning carriers (layers 23–26)** bring the question's category and clue words to the answer position.
    L26H17 is a "category reader".
  - **Content-blind gain control (layers 28–30)** is driven by formatting tokens, not by what was asked. Layer 28 has
    a push–pull trio, H4 and H6 up and H7 down, whose net roughly cancels (+1.8 + 1.8 − 3.3).
- The fixed suffix "Answer with just the name." contributes ~0 to every head.
- **Working interpretation (untested):** the layer-28 trio is a dial for *how strongly* to express whatever concept is
  forming, not *which* concept. A causal check would switch the brake (L28H7) off and look for raised, or
  over-confident, answers.

**Causal tests of the layer-28 trio** (`src/gain_trio.py`, `src/route_cut.py` with CUT_HEADS;
`results/gain_trio_summary.txt`, `results/route_cut_brake.log`).
- **Surgical (route only), brake L28H7 → entry neuron, 4 concepts.** Removing only the brake's route raises the entry
  neuron (coffee 65.8 → 78.1, Golden Gate 56.4 → 66.2, the Moon 119.2 → 130.7, Mozart 48.3 → 49.2) and raises recall
  slightly (+0.12 / +0.06 / +0.13 / +0.04 log-odds). Random routes do nothing (|Δ| ≤ 0.06). Eiffel was skipped: the
  brake is above its layer-26 entry neuron. **The brake on concept neurons is confirmed.**
- **Whole-head knockout, 59 concepts.** Answer log-odds change:

| switched off | up | down | ~0 | mean \|Δ\| | entropy change |
|---|---|---|---|---|---|
| brake L28H7 | 21 | 30 | 8 | 0.199 | ≈ 0 |
| accelerators L28H4 + H6 | 22 | 27 | 10 | 0.210 | ≈ 0 |
| random layer-28 head | 4 | 14 | 41 | 0.043 | ≈ 0 |

- The trio matters (~5× a random head), but with mixed, concept-dependent direction and no change in output sharpness.
  **The "global gain dial" reading is rejected.** The trio acts on concept neurons as predicted (surgical tests), but
  whole-head removal touches many neurons, triggers normalisation compensation, and has mixed net effects on answers.
- This is consistent with the earlier finding that whole-head knockouts can mislead about a head's role, while route
  cuts don't.

**Causal test of concept-neuron chains** (`src/chain_cut.py`, `results/concepts_v3/chain_cut_v3.json`). For a link
S → T at the answer position, we removed only S's contribution to T's inputs (S still writes everywhere else).
- **Control:** the largest feeder of T that is not one of the concept's top-10 neurons (a size-matched generic input).
- **Also measured:** S switched off entirely.
- **Measures:** T's activation and the answer's first-token log-odds.

| link | T: base → chain cut | recall drop: chain cut | matched control (feeder) | S fully off: recall drop |
|---|---|---|---|---|
| the Moon L31:9683 → L32:6316 | 38.9 → **−6.2** | **0.59** | 0.07 (L29:3948) | 3.05 |
| the Moon L31:9683 → L33:2482 | 38.2 → **5.5** | **0.43** | −0.06 (L32:3953) | 3.05 |
| the Moon L31:9683 → L32:5142 | 30.2 → 18.7 | **0.61** | 0.18 (L29:3458) | 3.05 |
| Golden Gate L28:6711 → L31:9611 | 73.7 → **35.9** | **0.68** | 0.16 (L29:4289) | 1.20 |
| Eiffel L26:11492 → L29:12094 | 30.8 → 20.0 | 0.18 | −0.06 (L27:12062) | 0.64 |
| Golden Gate L26:11492 → L28:6711 | 56.4 → 48.6 | 0.08 | 0.09 (L26:752) | 0.42 |

- **[Corrected in the evening entry: at scale (35 informative links) a chain link is NOT reliably more important than
  a size-matched non-concept input - 21/35, sign test p = 0.16. These 6 were hand-picked.]**
- **Chain links are causal for recall** in 5 of 6 valid links: cutting a single concept-neuron → concept-neuron route
  lowers the downstream neuron (up to silencing it) and recall, more than a size-matched non-concept input.
- **The Moon neuron L31:9683 is a hub** feeding three downstream Moon neurons; one cut silences L32:6316.
- The Golden Gate landmark part → entry neuron link is weak (≈ control), consistent with the entry neuron being mainly
  head-fed.
- **Excluded row (my error):** the Mozart link L28:10728 → L30:8885 was invalid, because L28:10728 is not among Mozart's
  top-10 neurons. The matched-control selection then picked the same neuron, so the row compared it with itself.
- **Each link carries part of the source's effect, not all of it:** the Moon hub fully off drops recall 3.05, versus
  ~0.5 per link, because it feeds several neurons and other routes.

**Head ↔ neuron picture after all causal tests (exploratory; Qwen3-8B; 59 concepts for the decompositions, 4–6 for
the causal cuts):**
1. Meaning-carrying heads (layers 23–26; category reader L26H17, clue readers L23H10 / L24H26 and others) deliver the
   question's content to the answer position.
2. Entry concept neurons (mostly layers 24–29) combine it. This link is causal (head route cuts).
3. Content-blind layer-28 heads push entry neurons up (H4, H6) or down (H7, brake confirmed by route cut). Not a global
   dial.
4. Concept neurons amplify through chains in layers 30–35. These links are causal (chain cuts); hubs such as the Moon's
   L31:9683 drive several downstream concept neurons. **[Corrected in the evening entry: not above a matched
   control at scale.]**
5. Late layers produce the answer word. Not yet traced. **[Traced in the evening entry.]**

Open: generality of the causal cuts beyond 4–6 concepts; the final step to the answer token; whether the "category
reader" and the "brake" are general mechanisms or artefacts of chat-format recall prompts.

## 16 Oct 2026 (evening): ATTENTION study steps 3-4 at scale - who writes the answer, causal sweeps, circuit maps (EXPLORATORY; research branch)
Same 59 concepts (CR-2 + wave 2), same prompts and answers. No API calls. Every number below is in `results/` with the
script named. Several of my own setup errors were caught on the way; they are listed at the end.

**1. Who writes the answer word** (`answer_readout.py`, `answer_readout_summary.py` -> `results/answer_readout_summary.txt`).
At the answer position the final residual stream is exactly the sum of the embedding, every head's output and every
neuron's output, so the answer's score splits exactly into their contributions (final normalisation scale held fixed;
reconstruction error <= 1.6%, score error <= 0.4%; rerun bit-identical). Two scores:

| | heads | neurons | concept's own top-10 neurons, directly |
|---|---|---|---|
| answer vs the average word (median 41) | 22% | 78% | 17% |
| answer vs its likely rivals (log-odds direction; median 5.5) | 35% | 65% | **88%** (quartiles 0.31 / 0.88 / 1.64) |

- Almost everything comes from layers 24-35 (neurons in layers 30-35: 58% of the vs-average score).
- Writing is spread out: half the vs-average score needs ~14 neurons (range 1-33), 90% needs ~68.
- **Reading:** the crowd of writers mostly writes "an answer-shaped word" (a capitalised name-like start); the part that
  makes it THIS answer rather than a rival is written mostly by the concept's own few neurons, directly. In 38/59 they
  write more than half the margin, in 26/59 more than all of it (other parts push back); in 13/59 less than a quarter.
- **Content vs format writers** (`writer_specificity.py`, static weights): of 590 top-writer slots, 48% content writers
  (push their own answer >= 3 sd above same-shape answers), 14% format writers (push >= 75% of the other answers up),
  37% mixed. 121 of the 161 concept neurons that are also top writers are content writers (the explainer pilot's
  "name-start" neuron is this kind). Format writers are shared: **L33:7759** (19 concepts) pushes every bare capital
  letter; L33:10933 and L34:9002 push spaces / digits / punctuation down. Labels v2 were set after v1 showed a sign bug
  and a token-shape confound (see errors).
- **Late heads** (`writer_sources.py`, value-weighted, reconstruction <= 0.2%): L35H25, L35H27, L34H29, L35H24 take
  75-83% of what they write from the answer position itself (they re-read and adjust); **L34H28 and L35H22 copy mostly
  from the `<|im_start|>` that opens the assistant turn**. Almost nothing comes from the question words or the "?".

**2. The answer is staged at the turn boundary** (`staging.py` -> `results/staging_summary.txt`; logit lens, ranks only).
At the newline after the user's turn and at `<|im_start|>`, the answer is top-10 from layers 28-29 in 70-77% of prompts
(rank 3-6 at layer 33), 6-8 tokens before it is spoken. Not at `assistant` / `<think>` / `</think>`; weakly at "?" (rank
~1,000). A random other concept's answer: rank 50k-100k everywhere. This is where the turn-start heads copy from.

**3. Causal sweeps** (`causal_sweep_args.py`, `route_cut.py`, `chain_cut.py`, `causal_sweep_summary.py` ->
`results/causal_sweep_summary.txt`). One entry neuron and one chain link per concept, picked by a fixed rule.
- **Head -> neuron routes: causal at scale.** 49 of 53 informative tests (4 uninformative: the neuron itself doesn't
  matter) pass (drop >= 0.1 and > 2x a random route). Median recall drop 0.73 vs random route 0.00; ~44% of switching
  the whole neuron off.
- **Neuron -> neuron chain links: not special at scale.** Median drop 0.35 vs 0.20 for the matched strongest
  non-concept input; only 12/35 pass; the link beats its control in 21/35 (sign test p = 0.16). Late concept neurons
  integrate many comparable inputs.

**4. No bow-tie** (`bowtie.py`, `bowtie_summary.py` -> `results/bowtie_summary.txt`). Switching off a concept's top-5
neurons at the answer position removes about what they write directly; the rest of the network partly makes up for it
(42/59) rather than amplifying (8/59). ~63% of their recall effect is at the answer position; the rest happens at
earlier positions (median log-odds drop 4.3 answer position only vs 6.5 every position; random neurons 0.0).

**5. Late head teams** (`head_arms.py`, `norm_heads.py` -> `results/norm_heads_summary.txt`). Switching off L35H25+L34H28+
L35H22 or L35H27+L34H29+L35H24 matters far more than random late heads (|d log-odds| > 0.5 in 50 and 44 of 59 vs 1)
but not in one direction. Not via the final normalisation (removing any one changes the final rms by <= 1.3%). By
direction against the rivals, **L35H27 is the one genuine suppressor** (removing it raises recall in 44/59); L34H29 and
L35H24, grouped with it from the vs-average readout, actually help.

**6. Circuit maps** (`circuit_maps.py` -> `results/circuit_maps_summary.txt`). Joining the above per concept:
- **Heads are a shared skeleton:** 3 writer heads fill 70% of writer-head slots (L35H25 in 59/59); 13 reader heads
  (L23H10 44, L24H19 38, L26H17 35, ...) fill 39% of reader slots; brakes L28H7 (36) and L25H14 (16).
- **Concept neurons are almost all private:** 167 distinct among 177 top-3 slots; content writers 267 distinct of 284;
  format writers shared (14 distinct over 82 slots).

**Revised head <-> neuron picture (exploratory):** meaning-carrying heads (layers 19-26) bring the question's content
to the answer position and the turn boundary; private concept neurons (mostly layers 24-33) are fed by those heads
(causal); the answer is staged at the turn boundary by layers 28-29 and copied forward by turn-start heads; at the
answer position a crowd of neurons writes "an answer-shaped word" (shared format writers among them) while the
concept's own neurons write most of what separates the answer from its rivals; one late head (L35H27) damps it.

**Caveats:** one prompt format (chat template, "answer with just the name"); first answer token only; 59 famous
concepts; attention weight vs information moved handled by value-weighting, not full path patching; the logit lens is
crude below the last layers (ranks only).

**My errors caught on the way (all fixed before the numbers above):**
- writer specificity v1 oriented neurons by contribution sign instead of activation sign, and confused "pushes single
  capital letters" with content (token-shape confound);
- writer_sources v1 classed the answer position as a template token ("self" read 0 everywhere);
- the first rival readout used the full next-word distribution, whose gradient vanishes when p(answer) ~ 1;
- the weight reader returned tensors backed by a closed file mapping (crash);
- a GPU-queue waiter could have started a second model in the gap between jobs (replaced before it ran);
- **a hidden cap:** log-odds computed as log(p/(1-p)) from a float32 softmax with p clamped to 1-1e-9 cap at 20.72.
  `logodds_cap_audit.py`: 3 of 869 recall prompts hit it (true 16.5-16.8), 11 are off by > 0.1. Sign-based results
  (flips) are unaffected; it is in the public pipeline too (wave_free.py, logodds_check.py and others) - fix on main
  pending James's OK. New code uses the exact logit - logsumexp(others).

## 16 Oct 2026 (late evening): how are concepts coded? Dedicated, compositional or barcode (EXPLORATORY; research branch)
Prompted by James: if a concept is a COMBINATION of neurons, 442k neurons allow vastly more concepts than one-per-
neuron. In the literature that is superposition seen from the concept side (Elhage et al. 2022): more concepts than
neurons forces overlap, so each neuron serves several concepts. The question that matters is what KIND of overlap:
dedicated neurons, compositional (meaningful shared parts, related concepts share), or barcode-like (arbitrary
overlap, neurons look polysemantic). No API calls.

**Step 1 - one neuron ranking for every recalled concept** (`rank_save.py`). The judge-free waves had kept only counts.
Re-ranked all **178** recalled concepts (CR-2 19, wave 2 40, wave 3 39, wave 4 40, CR-3 40) exactly as wave_free.py
does. Every concept with an original run reproduced its recall count and k_mean / k_flip exactly (**0 mismatches,
119 concepts**). For CR-2 / wave 2 the older attention-study lists agree on the strongest neuron (old #1 in the new
top-3 for 57/59; top-10 overlap median 6-7).

**Step 2 - cross-activation** (`cross_activation.py`, `cross_activation_summary.py` -> `results/cross_activation_summary.txt`).
Every concept's top-10 neurons measured on every other concept's recall prompts. 'Fires' = at least half its own-concept
activation, answer position. Sanity: own-concept ratio median 1.000. Families fixed beforehand (`families.py`, 17).
- Raw: same family 29.3% vs different family 22.4%.
- **Post hoc (after the first look):** 35 neurons, almost all in layers 16-24, fire for >= 90% of concepts - "always
  on" question-answering machinery, not codes. Set aside, the rest (1,074 neuron-owner pairs):
  - **same family 12.7% vs different family 4.8% (2.6x)**; 504 fire for no other of the 177 concepts; 228 mainly
    within their family; 95 broadly.
  - Strongest for people (24.4% vs 1.4%), myth (21.8 vs 4.1), animals (16.0 vs 2.4), fiction (7.0 vs 1.3).
- **Answer-type confound check:** people-owned neurons fire for other real people 26.8%, named characters (Harry
  Potter, Sherlock Holmes, Mickey Mouse, Zeus, Thor, the Buddha, Medusa) 12.5%, places 6.1%, everything else 4.1%.
  Part of the family effect is "a person-like name is coming"; part is specific to real people.
- Layers 24-35: median activation on unrelated concepts ~0.

**Step 3 - corpus profile** (`corpus_profile.py`, `corpus_profile_summary.py` -> `results/corpus_profile_summary.txt`).
The atlas's 10M web tokens; 1,100 concept neurons (166 with own peak < 1 not profiled, listed in the log) plus a random
same-layer neuron each at the same threshold. EVERY firing >= half the neuron's peak on its own concept was kept (101.9M).
- Concept neurons' strong firings sit near their concept's name (+-64 tokens) **20x more often than chance** (median;
  706/1,100 >= 5x; random neurons ~0). Most specific: Moon hub L31:9683 0.91, Bitcoin L9:2665 0.91, chocolate
  L31:8429 0.87, cheese L32:1778 0.87, Eiffel/Paris L31:10160 0.85.
- But the median share of firings near the name is **5%** (a lower bound: related words such as "lunar" don't count).
- 'Near another of our concepts' (median 0.54) equals its chance rate (0.54): uninformative, dropped.
- **Eyeballed random firings (4 typical neurons, 8 each):** democracy L32:8254 -> representatives, councils,
  parliament (a broader ingredient: representative governance); Serena Williams L31:11800 -> Djokovic, Nadal,
  Federer, Mayweather, an F1 crash (elite athletes); Titanic L31:5874 and hamburger L32:54 -> mixed, partly unrelated
  (casino, stock index, sports drafts): polysemantic-looking.

**Reading (exploratory):** the coding is neither one-neuron-one-concept nor a barcode. About half of the late concept
neurons are dedicated within our 178 concepts; the rest are mostly shared within families, and in ordinary text many
look like broader ingredients the concept is built from (a compositional code), with a minority that look
polysemantic. A concept is a combination of a few dedicated neurons and several broader, reusable ingredient neurons.

**Limits:** 178 hand-picked, mostly famous concepts; one prompt format; keyword matching is a lower bound and cannot
tell 'related' from 'unrelated' contexts; the eyeballing is 4 neurons. Next: a judge reading firing samples
(dedicated / ingredient / unrelated), and a random, fame-stratified concept sample.

## 17 Oct 2026 (night): Luna reads the concept neurons' web-text firings - most are polysemantic in the wild (EXPLORATORY; research branch)
`luna_firing_judge.py`, `luna_judge_summary.py`, `luna_judge_extras.py` -> `results/luna_judge_summary.txt`,
`results/luna_judge_extras.txt`. Luna (~$0.53 at its real price). Turns the "ingredient" eyeballing of the previous
entry into a measured number.
- **Design:** every profiled concept neuron with >= 12 strong firings in the 10M web tokens (1,095), 12 RANDOM firings
  each (>= half its peak activation on its own concept; +-20 tokens of context), labelled A = about the concept,
  B = related (same topic, or a broader category / ingredient), C = unrelated. Leniency control: each neuron's paired
  random same-layer neuron, judged against the SAME concept name (778 with >= 12 firings; 314 had fewer, skipped and
  logged). Luna is not told which is which. Classes fixed beforehand: dedicated A >= 6/12; ingredient A < 6 and
  A + B >= 9; unrelated C >= 6; else mixed.

| | A (about it) | B (related) | C (unrelated) | dedicated | ingredient | mixed | unrelated |
|---|---|---|---|---|---|---|---|
| concept neurons (1,095) | 14.9% | 29.9% | 55.1% | 130 (12%) | 242 (22%) | 94 (9%) | 629 (57%) |
| random controls (778) | 0.2% | 4.9% | 94.9% | 0 | 5 (1%) | 2 | 771 (99%) |

- **Luna is not lenient** (controls ~95% unrelated), and it agrees with the keyword measure where both apply (median A
  share 0.42 where >= 20% of firings are near the concept's name, 0.00 where < 2%).
- By layer: early (0-15) neurons are the most concept-related (A 23%, B 42%); mid (16-23) almost all unrelated (the
  'always on' machinery); late (30-35) 63% unrelated.
- **The 'unrelated' ones are not token detectors:** like every class, their single most common firing token covers
  only ~8% of firings (9/629 dominated by ' the' or '.'); Luna's themes for their unrelated firings are nearly all
  "miscellaneous topics" - spread across many topics, not one second meaning, and rarely formatting (22/629).
- **Not 'clean at the top':** the A share rises only from 13% (firings at 0.5-0.75x the neuron's own-concept peak) to
  ~20% (>= 1x); about half are unrelated even at full strength.

**Reading (exploratory):** neurons that are causally necessary for recalling a concept (and, within the 178 recall
question sets, mostly dedicated or family-shared) spend most of their strong firings in ordinary text on unrelated,
scattered material. That is the signature of superposition: the neuron is specific in the context that matters and
polysemantic in general, with low interference in the recall setting. It qualifies "already clean" - clean in the
task context, not in general - and argues that a neuron-by-neuron map of the whole model would need a different unit
(sparse-autoencoder / transcoder features) on top of these causal tests.

**Limits:** "related" is Luna's judgement from +-20 tokens; a non-topical feature (syntax, discourse position) would be
labelled unrelated; random firings above a threshold, not a full activation distribution; one corpus (10M FineWeb tokens).

## 17 Oct 2026 (night): hidden log-odds cap fixed on main (ISSUES #35; approved by James)
See ISSUES #35 on main (commit 251140b; cherry-picked here as d692c66). log(p/(1-p)) on a float32 probability clamped
to 1-1e-9 capped at 20.72 (3/869 recall prompts) and lost precision above ~10. `pipeline_fast.batch_logodds` is exact
(float64, expm1). Tested by `logodds_exact_check.py` (1.5e-9 vs an independent formula; 0.001 vs the old values below
10). End to end (`rank_save.py` with LOGODDS=exact, `cap_fix_compare.py` -> `results/cap_fix_compare.txt`): all 178
concepts keep identical k_mean / k_flip; every published wave result reproduces; top-10 order changes in 10 (near-ties);
no published number changes. Near saturation any log-odds carries ~0.1 (max 0.45) of bf16 batch-shape noise.

## 17 Oct 2026 (night): does "a handful of neurons" hold for obscure concepts? Fame-stratified random sample (EXPLORATORY; research branch)
`wiki_sitelinks.py` (sample), `wave_budget.py` (runs wave_free.py unchanged, round-robin over tiers, Luna-budgeted),
`fame_summary.py` -> `results/fame_summary.txt`. Luna ~$0.42 at its real price.
- **Sample:** 210 concepts drawn at random from ~12,600 English Wikipedia articles (2023 and 2025 monthly top-1000
  lists plus 5,000 random articles), stratified by fame = number of Wikipedia language editions (Wikidata sitelinks),
  7 tiers of 30 on a doubling scale (>= 128, 64-127, 32-63, 16-31, 8-15, 4-7, 1-3; fixed before looking). Filters:
  article existing by 2023-06-30 (first revision), no 2024+ year in the title, no disambiguation pages, no
  adult-industry categories; every rejection logged. Duplicates of earlier sets kept here (random-sample integrity)
  and excluded from the coding analyses.
- **How the sample was chosen (corrections on the way):** 2025 page views (v2) favoured topics that became notable after
  the model's training data (2025 films, elections) - recency, not fame; 2023 page views (v3) needed one request per
  title and Wikimedia throttled us (HTTP 429); language editions come 50 per request and are not tied to one year.

| fame tier (editions) | recalled | knows it* | k_mean <= 5 | median k_mean | k_flip <= 5 | median baseline log-odds |
|---|---|---|---|---|---|---|
| >= 128 | 27/30 | 29/30 | 27/27 | 1 | 23/27 | 5.4 |
| 64-127 | 24/30 | 27/30 | 23/24 | 2 | 19/24 | 3.0 |
| 32-63 | 12/30 | 16/30 | 12/12 | 1.5 | 12/12 | 1.5 |
| 16-31 | 3/30 | 8/30 | 3/3 | 2 | 3/3 | 3.5 |
| 8-15 | 4/30 | 5/30 | 4/4 | 1 | 3/4 | 1.3 |
| 4-7 | 1/30 | 1/30 | 1/1 | 1 | 1/1 | 1.6 |
| 1-3 | 1/29 | 1/29 | 1/1 | 3 | 1/1 | 0.2 |

*knows it = recalled, or the model's own free answers name the concept in >= 2 questions (answer discovery) - separates
"does not know" from "names it differently from the article title" (e.g. "Russian invasion of Ukraine (2022-present)").

- **Knowledge thins out fast with obscurity** (recall 90% -> 3%), **but wherever the model knows a concept, a handful
  of neurons still carries it:** k_mean <= 5 in 71 of 72 recalled concepts, median 1-3 in every tier; no trend with
  fame (Spearman -0.08 with log editions). k_flip rises slightly with fame (+0.27), tracking baseline confidence
  (+0.52): near-certain facts are harder to flip (saturation, as documented before).
- **Reading (exploratory):** obscure concepts do not show up as "more spread out" - they are simply not known (or known
  weakly). The concentration finding generalises from the hand-picked famous sets to a random fame-stratified sample,
  within the concepts the model can recall.
- Steering on these Wikipedia titles is mostly near zero (median held-out 0.00-0.15; random 0.00); the keyword check
  needs the exact name in the reply, which is strict for long titles. Not pursued.

**Limits:** few recalled concepts in the obscure tiers (1-4 each); recall needs the answer's exact first tokens
(awkward titles lose recall); Luna declined one ambiguous title ("A. L. Holt", recorded as prompts failed).

## 17 Oct 2026 (early morning): coverage expansion - 157 more concepts from the tiers the model knows (EXPLORATORY; research branch)
`wiki_extra.py` (a second draw, seed 8, from fame tiers >= 128 / 64-127 / 32-63 editions: 60 / 60 / 40, duplicates of
earlier sets dropped: Cleopatra, Steve Jobs, Moon -> 157), run through the improved loop (wave_free.py via
wave_budget.py -> neuron IDs saved by rank_save.py -> cross-activation over every set). Luna ~$0.29.
- Recalled 111/157 (tiers: 56/57, 43/60, 12/40 - the same fall-off as the first fame draw).
- **k_mean <= 5 in 108/111 (<= 2 in 82; median 2); k_flip <= 5 in 98/111; flips within 50 in 109/111.**
- Steering on these titles: median held-out 0.00, mean 0.08 (random 0.003) - as in the first fame draw.
- **Running total: 359 distinct recalled concepts mapped** (178 hand-picked + 70 fame sample + 111 coverage), every
  one with saved top-30 neurons.

## 17 Oct 2026 (morning): 'dedicated' is relative to the population - and keeps falling (EXPLORATORY; research branch)
`dedication_curve.py` -> `results/dedication_curve.txt`. James's point: "fires for no other concept" is only as strong
as the set of concepts it was tested against. From the 359-concept cross-activation (no new runs): for each concept's
top-10 neurons (|own act| >= 1; the 34 'always on' neurons excluded once, on the full set), draw N other concepts at
random (20 repeats) and count the neurons that fire (>= 0.5x own activation) for none of them.

| N other concepts | 10 | 25 | 50 | 100 | 150 | 200 | 250 | 300 | 358 |
|---|---|---|---|---|---|---|---|---|---|
| dedicated, hand-picked owners | 78% | 71% | 63% | 54% | 48% | 44% | 41% | 39% | 36.5% |
| dedicated, all owners | 78% | 68% | 59% | 49% | 43% | 38% | 35% | 32% | 30% |

- **No flattening:** the dedicated share falls by ~9-10 points per doubling of N, steadily from 50 to 358. No sign of a
  core of private neurons within this range; neurons serve 0 (724), 1 (332), 2-4 (461), 5-19 (434) or 20+ (470) of the
  other 358 concepts - a heavy-tailed spread, as sparse superposed coding would give.
- **Prediction, written before the data (coverage-3 run of 289 more concepts, started 08:15):** if the log-linear
  trend holds, the hand-picked owners' dedicated share against ~648 other concepts will be **about 28-29%**
  (36.5% - 9.3 x log2(648/358) = 28.5%); flattening would leave it above ~33%, steepening below ~25%.
- Reading: "dedicated" earlier (47% vs 177 others) was mostly a small-population effect; with growing coverage almost
  every concept neuron turns out to take part in some other concept's code.
- **Clarification, added before the coverage-3 data exists:** not all 289 concepts will be recalled, so N will be below
  648. The prediction is the formula, evaluated at the actual N: dedicated(N) = 36.5% - 9.3 x log2(N/358) (hand-picked
  owners, same rules); 'holds' = observed within +-3 points of it.

**Correction (17 Oct, after James's question):** the two entries above call these results "the signature of
superposition". That overclaims. Neither test separates three different things: (a) the same meaning reused by several
concepts (a "gold" neuron in Golden Gate Bridge and gold jewellery - composition, not superposition); (b) related facets
of one broader meaning (Moon / month); (c) genuinely unrelated meanings packed into one neuron (superposition proper).
Cross-activation and the dedication curve show that sharing grows with the population - which (a) alone would produce.
The Luna judge asked whether firings relate to the OWNER concept; a gold neuron firing on jewellery counts as
"unrelated to the Golden Gate Bridge" while keeping one meaning. Whether concept neurons carry several unrelated
meanings has not been tested yet; a neuron-centred test (how many meanings, how far apart) is proposed.

## 17 Oct 2026 (day): does the map tell us something real? Pre-registered predictions (research branch)
James: first prove the loop yields meaningful findings, then scale. `PREREG_MAP.md` (frozen at 533f3f1 before any of
the data below) states five predictions with pass/fail rules; results are appended under each there.

| | prediction | result |
|---|---|---|
| P1 | a concept's top-5 neurons (found on Luna's questions) also break recall on independent prompts - Wikipedia opening sentences with the name blanked (cloze) | **holds**: median drop +5.59 log-odds vs random-5 -0.003; flips >= half the correct prompts in 66% (32 concepts tested) |
| P2 | dedication curve out of sample: dedicated(N) = 36.5% - 9.3 x log2(N/358), +-3 points | **holds, narrowly**: 32.0% observed vs 29.4% predicted at N = 609 (+2.6) |
| P3 | a neuron shared by two concepts' maps matters to both | **holds**: drop on the other concept +0.835 vs its non-shared neuron +0.002 and a random one +0.000 (150 pairs) |
| P4 | a new concept's neurons can be predicted from its Wikidata family's neurons (predictions committed before coverage-4, 82fdb29) | **holds**: own-family neurons at 9.2x the other-family rate (needs 3x), 118/185, p = 0.0001 (191 new concepts); shuffled labels never reach it; holds without people (8.3x). Caveat: family ~ answer type |
| P5 | a neuron's description in web text reproduces on new text | **holds**: firing-rate rho 0.986, on-concept share rho 0.883 (2M new tokens) |

**Reading:** the map is about concepts, not prompt wording (P1); its sharing structure is causal (P3); its corpus
descriptions are stable (P5); its population-level law extrapolates (P2); and its structure predicts the neurons of
concepts it has not seen (P4). All five pre-registered predictions hold. These are tests the map could have failed.

## 17 Oct 2026 (day): one meaning or several? Luna's view vs Qwen's own (EXPLORATORY; research branch)
Prompted by James: a shared neuron that keeps one meaning (a 'gold' neuron in several concepts) is composition, not
superposition; and are 'unrelated' meanings really different, or facets of one concept we can't see (an 'Ur concept')?
- **Stage A** (`meaning_judge.py`, Luna blind, 30 random strong firings per neuron; calibrated on spliced two-neuron
  units, which Luna separates 65% of the time - it undercounts mixtures): 196 concept neurons: one meaning 19%, two 31%,
  three 28%, four+ 22%; an unrelated pair in 83 of 159 multi-meaning neurons (controls 75%, spliced 94%).
- **Stage C** (`meaning_space.py`; group centroids in Qwen's own whitened MLP-input space, the neuron's read directions
  removed): meanings Luna calls closely related sit close (cosine +0.312), unrelated ones near the known-unrelated
  reference (+0.034 vs +0.010; same-meaning +0.124). Spearman(Luna, Qwen) -0.50. Only 6/278 'unrelated' pairs are close
  in Qwen's space. **Mostly genuine packing, not hidden facets.** Ur-hubs (one meaning central to the others, beating a
  permutation null) in 8/98 concept neurons (5% expected by chance; 0 in controls and spliced): Moon (lunar phases <-
  space exploration, cultural moon), volcanoes <- mountains / craters, AI <- robots / autonomous vehicles, human evolution
  <- archaeology / primate diet. Weak evidence (8 vs ~5).
- **Stage D** (`meaning_jobs.py`; each firing context run with the neuron switched off): the pre-set Luna-free clustering
  FAILED its own calibration (the silhouette rule called 40/52 known two-neuron mixtures 'one meaning'), so its
  'one meaning' share is not reported as a finding. Second reading (`meaning_jobs_luna_groups.py`, Luna's groups, Qwen's
  measured effects): across a concept neuron's meanings the downstream job barely differs - gap 0.18 of the way from
  'same job' (random halves) to 'different neurons' (spliced) - even for meanings Luna calls unrelated (effect cosine
  0.130 vs 0.173 for closely related); the next-tokens it pushes are the same (Jaccard gap ~0).
- **Reading: packed triggers, one job.** Concept neurons are triggered by several, often unrelated contexts (input side)
  but do essentially the same thing downstream whatever triggered them (output side). Open: harmless reuse or
  interference; and whether the meaning count saturates with more firings (only 30 sampled - the same population
  effect as the dedication curve; a meaning-count vs sample-size curve is the next test).

## 17 Oct 2026 (day): the loop itself
- `map_loop.py` replaces wave_free.py + rank_save.py for coverage: Luna and GPU as separate queues meeting through the
  spec files, ranking once, no steering, everything saved (all 200 candidates, per-prompt drops, activation maps that
  make cross-activation incremental, attention heads: attribution, answer-position outputs, attention patterns, top-50
  single-head knockouts), per-step timing and cost. Validated: reproduces the old pipeline exactly on 12 concepts;
  maps match cross-activation (64,836 values, max |diff| 0.125).
- Coverage-3 remainder (186 concepts): 0.45 min per concept overall vs ~1.65 (recalled) / 0.64 (not) for the old loop;
  GPU idle 12% (a judge-starvation bug at the start, fixed); Luna $0.0018 per concept.
- Batched single-neuron/head knockouts v1 FAILED their gate (bf16 batch-shape noise vs a separately computed
  baseline); v2 measures against an unablated copy in the same pass - to be re-gated.
- Batched knockouts v2 ALSO FAILED (16:34): neuron order and curves identical to v1's failure (the baseline shift is common
  to a pass, so it was never the cause; the noise is between rows of a large bf16 batch vs a small one), head drops off by
  0.44-2.7. Decisive: the speed-up is only x1.1 (single knockouts 166 -> 149 s, heads 41 -> 37 s) - each knockout pass
  already batches all of a concept's prompts, so the GPU is already full. Batching DROPPED; the loop stays unbatched.
  Outputs kept in results/maploop_repro/failed_batched_v1 and _v2.
- Running total: 610 recalled concepts mapped before coverage-4; coverage-4 (concepts_fame5, 376 drawn) finished 16:08,
  unbatched loop, 0 failures.

## 17 Oct 2026 (late afternoon): do a concept's key neurons sit on the same layer? (EXPLORATORY; research branch)
James: is the combination spread one-per-layer, or can two concept neurons share a layer? (`same_layer.py`, 914 mapped
concepts, ranked by knockout effect; chance = the same overall layer mix drawn independently.)
- Sharing a layer is common: 46% of concepts have >= 2 of their top-5 on one layer (top-10: 94%; up to 5 on one layer).
  Of the k neurons each concept actually needs (k_mean >= 2, median 2), 17% share a layer.
- But slightly LESS than chance: top-3 15% vs 22%, top-5 46% vs 54% (~5 SD below with 914 concepts). The chance model
  ignores concept-specific layer clustering (which would raise chance), so the spreading tendency is if anything stronger.
- Layers are heavily skewed late (L28-35 hold ~70% of top-5 neurons) with a second cluster at L8-10.
- Reading: no one-concept-neuron-per-layer rule; a mild preference for spreading across layers.

## 17 Oct 2026 (late afternoon): is stage D's 'one job' an artefact of bf16 noise? No. (EXPLORATORY; research branch)
The gpu-fan-out session measured the bf16 noise floor of single-neuron knockouts (8B, L18, 128-token windows): batch-shape
noise alone gives KL median 3.2e-4 / p90 1.9e-3, and a typical single-token knockout is only ~2x that. Stage D's two
passes were both batch-1 with the same shape (no batch-shape noise between them), but small effects can still depend on
the kernel's rounding. Check (`meaning_jobs_above_noise.py`, no GPU): rerun the second reading keeping only contexts
with knockout KL above thresholds (stage D KL median 2.1e-3 concept / 1.4e-3 control):
| KL >= | concept contexts kept | spliced gap ('different neurons') | concept position (0 same job, 1 different) |
|---|---|---|---|
| 0 | 100% | +0.150 | 0.18 |
| 1.9e-3 (noise p90) | 53% | +0.182 | 0.22 |
| 3.2e-3 (10x noise median) | 43% | +0.217 | 0.21 |
| 1e-2 | 26% | +0.250 | 0.18 |
| 3e-2 | 14% | +0.277 | 0.22 |
The calibration sharpens as noise is removed (spliced separation rises, absolute cosines rise) while concept neurons stay at
~0.2: 'packed triggers, one job' holds on clearly above-noise knockouts. Not re-checked: the next-token Jaccard (top-20
up/down lists are probably noise-dominated for small KL), and per-context reproducibility across kernels (a GPU check).

## 17 Oct 2026 (evening): is a concept the intersection of its neurons' jobs? (EXPLORATORY; research branch)
James: a neuron has a job, not a concept; a concept is what a few jobs add up to. Test from weights only
(`job_overlap.py`, `job_overlap_z.py`; logit lens of each top-5 neuron's direct push = activation at the answer position
x its down_proj column, through the final-norm weight and unembedding; late neurons = layers >= 20; 909 concepts with
>= 2 late neurons). Answer = the first token of the recalled answer. Nulls: another concept's answer token; random
same-layer neurons with the same activation (20 draws each). Criterion stated in the script before running.
**Bug caught and fixed:** the first run took the first LETTER of the saved answer token (a string), so every concept was
scored against e.g. 'N'; that output is kept aside (results/job_overlap_letterbug/, not committed) and is invalid.

| | real | answer-shuffle | random neurons |
|---|---|---|---|
| dedicated: one late neuron alone has the answer in its top-10 (of 151,936) | 59.8% | 0.4% | 0.2% |
| intersection: none alone, but the sum of the late pushes does | 6.8% | 0.0% | 0.0% |
| neither | 33.3% | 99.6% | 99.8% |
| >= 1 late neuron raises the answer (z >= 2 over the vocabulary) | 93% | 17% | 9% |
| >= 2 late neurons raise it ('votes') | 71% | 2% | 0.8% |

- Without a dedicated neuron, the sum reaches top-10 in 62/365 (17%) vs ~0.01% for the nulls (criterion met).
- Summing does not usually sharpen the answer: the sum's z (median 8.4) is below the best single neuron's (16.4),
  and the sum beats the best single in 17% ('voting').
- **What the jobs look like** (each neuron's most-raised tokens), e.g. Jimmy Carter = first names (' Bill') + 1970s
  (' disco', ' Sevent') + presidency (' administration', ' oval') + ' Georgia'/' Atlanta'; Golden State Warriors = Bay
  Area (' SF', '湾区') + gold + ' Golden'; William Shakespeare = first names + European first names + ' Ham'/' Juliet';
  Elizabeth I = 'el'-spelling + ' Renaissance'/' Tud'. There are also spelling jobs (push tokens starting with a
  letter) for answers that begin with a fragment.
- **Reading:** both pictures occur. Most concepts have one late neuron that is close to 'say this word' (60%), usually
  alongside 1-3 attribute neurons (place, era, role, name type) that also raise the answer, much less strongly, among
  related words. Where no single neuron says it, the overlap of attribute jobs alone points to the answer in about 1
  in 6. Jobs are often legible attributes ('Georgia', 'Bay Area', '1970s', 'first name').
- **Caveats:** direct path only (a layer-30 neuron also acts through layers 31-35); the first answer token only;
  neurons were selected by their effect on the answer, which favours direct 'say it' neurons, so the 60% partly
  reflects the selection; logit-lens readings of layer-20s neurons are rougher than of layer-30s.

## 17 Oct 2026 (evening): are attribute jobs reusable parts? (EXPLORATORY; research branch)
James chose this after job_overlap.py. Question: can a neuron's JOB, read from the weights, pick out which concepts
USE it, according to the knockout maps? Those are two independent sources. (`attribute_sharing.py`; Luna as a blind
judge, $0.47: Luna's hidden reasoning tokens made it ~10x my $0.05 estimate.)
- Candidates: 1,645 late neurons (layer >= 20) in the top-30 maps of 3 to < 15% of the 914 concepts. Most are shared by
  only 3-5 concepts; 90 by 20 or more. Job = the 30 tokens its direct push raises most.
- 200 sampled at random. Luna saw only the job tokens and 10 concept names (k = up to 5 users + distractors), was told
  k use it, and picked k. Control: the same lists with a different neuron's job tokens.

| | hit rate |
|---|---|
| real job tokens | **74.6%** |
| another neuron's job (control) | 55.4% |
| chance (k/10) | 44.1% |

- Per neuron, the real job beats the control 103 times, loses 21, ties 76 (sign test p = 2e-14). On users whose name
  and answer don't appear among the job tokens: 71.7% vs 44.2% chance. Legible jobs (mostly English-word tokens) 76.1% vs
  control 59.6%; less legible ones (often Chinese or fragments, which Luna can still read) 72.5% vs 49.6%.
- **The pre-stated criterion was NOT MET, and it was badly set:** ">= 2x chance" is near-impossible when chance
  averages 44% (k = 5 of 10 gives 50%, so the ceiling is 2x). Recorded as not met; I am not replacing it after the fact.
  The two other parts, the sign test vs control and the non-lexical users, were met.
- **The control beats chance (55% vs 44%), and that is a finding in itself.** With no relevant job tokens, Luna still
  finds a neuron's users among random distractors more often than chance, because the concepts sharing a neuron are
  themselves similar (e.g. Meloni, Sunak, Zelenskyy, Trudeau and Hitler share a 'minister/cabinet' neuron).
- Examples of legible shared jobs: ' minister', ' cabinet' -> five heads of government; ' sky', ' par' -> Parkour,
  surfing, the helicopter; given names (' Chloe', ' Dylan') -> Verstappen, Ohtani, Chalamet, Prince William;
  ' psych', ' cardi', ' therm' (Greek roots) -> democracy, a doctor; spelling jobs ('H', 'W') -> honey, honeybee, wine.
- **Reading:** attribute neurons are reused across concepts that share the attribute, and their weight-read job
  says which ones, well above a matched control. Concepts are built partly from shared, nameable parts.
- **Caveats:** Luna's one-line job names are partly inferred from the concept list, not just from the tokens (it named
  a job 'recognizes country names' from unreadable tokens). The real-vs-control comparison is the valid test; the job
  names are illustrations. One judge; one sample of 200; direct-push jobs only.

## 17 Oct 2026 (evening): interference - when a concept neuron fires off-topic, does its push help or hurt? (EXPLORATORY; research branch)
James: many triggers, one job - harmless reuse, a leak the network tolerates, or a push it uses everywhere?
(`interference.py`, `interference_summary.py`; GPU 55 min, no API.) Contexts: the 12 random web-text firings per
neuron that Luna labelled in the firing judge (rebuilt exactly from its seeds: 0 of 22,476 snippets differ), 1,093
concept neurons + their paired random control neurons. Per firing, one batch of 3 identical sequences (document up to
the firing token + the next 16 tokens): as is / the neuron zeroed at that token / a random same-layer neuron zeroed
(null). delta = loss(zeroed) - loss(as is) in nats per token over the next 16 tokens: > 0 = the push HELPED predict the
real text.
**Noise (learned on the way, with the gpu-fan-out session):** the null is NOT exact. Any zeroing that changes anything
flips bf16 roundings, and the runs drift apart by ~0.005 nats/token whatever the size of the change. So effects are
read against the null, not against 0. (I first told the other session it was exact, from 6 trial contexts - corrected.)

| firings | n | median abs(delta) vs null | delta minus null, mean +- 2SE | above the null's p90 |
|---|---|---|---|---|
| concept neuron, about its concept (A) | 1,961 | 0.0100 vs 0.0057 (1.8x) | **+0.0040 +- 0.0018** (helps) | 36% |
| concept neuron, related (B) | 3,928 | 0.0086 vs 0.0055 (1.6x) | +0.0015 +- 0.0010 (helps, barely) | 31% |
| concept neuron, unrelated (C) | 7,217 | 0.0056 vs 0.0039 (1.4x) | **+0.0003 +- 0.0005** (neither) | 18% |
| random control neuron, unrelated (C) | 8,834 | 0.0058 vs 0.0044 (1.3x) | +0.0008 +- 0.0004 | 18% |

- Next token alone: on-concept firings help by +0.058 nats on average (1,027 helped vs 874 hurt, p = 0.0005);
  unrelated ones +0.009 (p = 0.07).
- Within the 745 concept neurons that have both kinds of firing: on-concept/related +0.0022 vs unrelated -0.0005
  (on > off in 404 vs 341, p = 0.02).
- By depth: the on-concept benefit is in layers 24-35 (+0.004-0.005); early-layer firings show none.
- **Reading: no interference detected.** When a concept neuron fires on unrelated text, switching it off neither helps
  nor hurts on average. Its unrelated firings matter about as much as a random neuron's firings do (both just above
  rounding noise), with no consistent direction. On its own topic its push helps, a little. So "packed triggers, one
  job" looks like harmless reuse at the resolution of single firings: the off-topic push is tolerated or too small to
  matter, not a harmful leak.
- **Caveats:** single-firing effects are tiny overall (a few thousandths of a nat per token); a weak but systematic leak
  below ~0.0005 nats/token would be missed; one firing at a time (a neuron that fires repeatedly through a passage could
  add up); the 38 firings whose activation didn't reproduce the profile (0.17%) are excluded.
- **Bug, recorded:** an edit swallowed the gate 2/3 counters into a comment, so the run printed 0 for both regardless.
  Checked afterwards from the saved records (`results/interference_gate.txt`, correction appended): 38 activation
  mismatches; zeroing changed the output in 98.8%, the rest being last-layer neurons below bf16 logit resolution. Raw
  per-context data: results/interference.jsonl in the main checkout (16.5 MB, not committed).

## 17 Oct 2026 (evening): the indirect path - attribute neurons feed the 'say it' neuron (EXPLORATORY; research branch)
James asked what the neurons do beyond their direct push. (`indirect_path.py`, `indirect_path_summary.py`; GPU 15 min,
no API.) The 555 concepts whose maps saved their recall prompts (the 359 older maps did not), 7,341 prompts. Each
top-5 neuron is zeroed on its own at every position (as in the maps), in one batch with the as-is prompt and an
ACTIVATION-MATCHED null for each: the same-layer neuron, outside the concept's top-30, whose activation at the answer
position is closest. Measured at the answer position: the answer score (answer logit minus mean logit), split into
DIRECT (the neuron's own write read through the final norm) and INDIRECT (the rest), plus how much every LATER top-5
neuron's activation drops. Neuron types from job_overlap: 'say' (the answer in the top-10 of its direct push),
'attribute' (raises the answer, z >= 2), 'late-other' (layer >= 20, doesn't raise it), 'early' (layer < 20).
Predictions stated in the script before running.

| type | neurons | score drop when zeroed | direct | indirect | direct share* | null |
|---|---|---|---|---|---|---|
| say | 408 | 2.24 | 2.87 | -0.20 | **1.05** | 0.00 |
| attribute | 786 | 1.35 | 1.17 | +0.15 | 0.80 | -0.02 |
| late-other | 1,065 | 0.89 | 0.05 | +0.76 | 0.05 | -0.01 |
| early | 516 | 1.14 | 0.00 | +1.14 | 0.00 | 0.00 |
*median over neurons with a drop > 0.5

- **(a) holds:** 'say' neurons write the answer themselves; the rest of the network slightly makes up for losing
  them (indirect -0.20, as in the bow-tie test). Attribute neurons are mostly direct too (0.80). 'Late-other' and
  early neurons work almost entirely THROUGH later parts.
- **(b) holds - composition:** zeroing an attribute neuron lowers a LATER 'say' neuron's activation by 12.7% (median per
  pair) vs -0.3% for its activation-matched null (282 vs 64 pairs, p = 9e-34). It is general: every type feeds later
  key neurons of the same concept (early -> say 11.7%, late-other -> say 11.6%, say -> later say 17.2%, early ->
  attribute 7.1%; all p < 1e-5), with the nulls at ~0. The only flat pairs: early -> early, and say -> late-other.
- **Reading:** a concept's key neurons form a small feed-forward assembly converging on the 'say it' neuron. Early
  neurons (recognising) and mid/late 'other' neurons feed attribute and say neurons, attribute neurons feed the say
  neuron, and each step also adds its own push to the answer. One upstream key neuron removed costs the say neuron
  ~12% of its activation.
- **Caveats:** answer position only (attention moves information between words, so some 'indirect' effect may run
  through other positions); first answer token; direct effect read at fixed final rms; per-neuron types depend on
  job_overlap's thresholds. Gate: zeroing verified in every prompt; 12 of 7,341 (0.16%) had first-token log-odds 0.5-1.0
  below the saved whole-answer baseline (bf16 batch-size noise). Raw data: results/indirect_path.jsonl in the main
  checkout (9.3 MB, not committed).

## 17 Oct 2026 (evening): the cross-position path - which heads carry the assembly between words (EXPLORATORY; research branch)
James: do upstream key neurons at EARLIER words feed the 'say it' neuron through specific heads? (`cross_position.py`,
`cross_position_summary.py`; GPU 18 min, no API.) 312 concepts with a 'say' neuron S and an earlier-layer key neuron U:
1,210 U -> S pairs, 16,111 pair-prompts. U was zeroed (a) at every word before the answer position, (b) at the answer
word only, (c) an activation-matched null at every earlier word. Each head's share of S's change came from
attribution patching (gradient of S x the head's output change at the answer word), checked by real patching of the
top-3 heads vs 3 random heads, inside one batch. Gate: heads at or before U's layer never changed (0 of 16,111).
Method change after the 3-concept trial, before the full run: S's bf16 activation (resolution 0.25 near 40) is as
coarse as many earlier-word drops, so recovery is measured inside one batch and the attribution (not quantised) is
also reported.
- **(c1) holds: the earlier-word part is real but small.** S drops by a median of 1.05% (mean 4.8%) of its activation
  when U is zeroed at earlier words, vs 4.4% (mean 11%) at the answer word only, and 0.0% (mean 0.1%) for the null.
  Earlier > null in 832 pairs vs 375 (p = 3e-40). Where both are positive, earlier words carry a median 19% of the
  combined drop. **Most of the assembly works within the answer word; about a fifth runs between words.**
- **Attribution tracks the real drops:** Spearman 0.92 over 16,111 pair-prompts.
- **(c2) holds: a few heads carry it.** Patching just the top-3 heads (by attribution) reproduces a median 56% of S's
  earlier-word drop (3,682 pair-prompts with a drop >= 5%); 3 random heads from the same layers, 0%. The top-3 heads'
  attribution is 1.3x the summed total (other heads partly push the other way).
- **(c3) the carriers are a small shared set** (top-3 slots, vs the uniform expectation):

| head | top-3 slots | x expected | concepts | known from the attention study (16 Oct) |
|---|---|---|---|---|
| L34H28 | 3,070 | 33x | 173 | copies the answer from `<|im_start|>`, the turn boundary where it is staged |
| L31H3 | 1,175 | 23x | 136 | - |
| L34H21 | 993 | 11x | 144 | - |
| L34H31 | 808 | 9x | 102 | - |
| L26H20 | 601 | 56x | 80 | meaning carrier: brings clue words to the answer position |
| L29H24 | 594 | 30x | 98 | mostly format |
| L35H22 | 526 | 15x | 71 | copies the answer from `<|im_start|>` |
| L35H25 | 499 | 14x | 70 | late re-reader; part of a late head team |
| L24H26 | 472 | 49x | 54 | - |
| L23H10 | 326 | 37x | 55 | meaning carrier (92% question words) |

  The 10 most frequent heads take 55% of all 16,737 top-3 slots; 254 distinct heads ever appear.
- **Reading:** between words, the concept assembly runs through a small, shared set of heads that independent earlier
  measurements had already flagged: mid-layer meaning carriers (L23H10, L26H20) and late heads that copy the answer
  from the turn boundary (L34H28, L35H22). This adds a causal link: key neurons' writes at earlier words reach the
  'say it' neuron through those same heads.
- **Caveats:** 'earlier words' lumps the question words with the chat-template tokens (including the turn boundary
  where the answer is staged), so how much comes from the question vs the staging position is not separated
  (next: zero U per position group); answer-position measurement only; first-order attribution (validated by
  patching); the top-3 slots double-count a head across a concept's prompts. Raw data: results/cross_position.jsonl
  in the main checkout (50 MB, not committed).

## 17 Oct 2026 (late evening): can we SEARCH for concept neurons instead of grinding? Feasibility (EXPLORATORY; research branch)
Context: James and I stepped back from one-off questions to the loop itself. Our 914 maps touch 12,168 distinct
neurons in any top-30 (2.8% of 442,368) and 2,663 in any top-5 (0.6%). Idea: if verified concept neurons look
different on cheap, model-internal profiles, rank all 442k and verify the top candidates causally (search, then
verify), rather than grinding every neuron. No labels imposed: only numbers measured from Qwen.
(`concept_search_feasibility.py`, `concept_search_size_check.py`; outputs in D:/already-clean-data/concept_search/,
summaries copied to results/concept_search/; GPU ~5 min.)
- Profile per neuron, both signs: 30 features. From the atlas (top-40 / bottom-40 firings over 10M tokens): peak,
  spikiness, decay, distinct tokens and documents, word-start / capitalised-word / gate-open shares, corpus max and
  min. From the weights: max z of its push over the vocabulary; capitalised / word-start / Latin share of its 50
  most-pushed tokens; gate / up / down norms; cos(gate, up).
- Positives: every pre-coverage-4 top-5 neuron (1,986). Negatives: 5 per positive, same layer, in no top-30. Logistic
  regression. Criteria stated before running.
- **Result: criteria NOT MET, narrowly.** 5-fold CV AUC 0.842 (needed 0.85). Out of sample, the 414 NEW coverage-4
  top-5 neurons rank at a median 90.1th percentile of all 12,288 neurons in their layer (needed 90; random 50.2);
  half are in their layer's top 10%. The top 1% of all neurons by score holds 400 known concept neurons, vs 24 at
  random (16.7x).
- **Artefact check, size:** the strongest feature was the gate vector's norm, and attribution (how the maps found these
  neurons) favours big neurons. With negatives matched on corpus peak: AUC 0.777, new neurons at the 87th percentile vs
  55 for size-matched non-concept neurons. With the size features removed as well: AUC 0.760, 82.9th vs 48.8th.
  **About a third of the signal is size; a real ~34-percentile-point signal survives**: concept neurons push towards
  CAPITALISED words (names), and their strongest firings decay slowly (strong again and again, not one-off spikes).
- **Reading:** cheap profiles work as a PRIORITISER, not a classifier. Ranking cuts the search ~10x at 50% recall
  (half the new concept neurons are in their layer's top 10%). The capitalised-push feature also shows the bias of our
  sample: we mapped named things, so the search will find more neurons for named things. The decisive test is the
  next one: of top-ranked neurons in NO map, what fraction can be given a concept and VERIFIED causally (precision of
  search-then-verify)?

## 17 Oct 2026 (late evening): search-then-verify pilot - verification works, proposing from web text doesn't (EXPLORATORY; research branch)
(`search_verify.py`; outputs in D:/already-clean-data/search_verify/, summary copied to results/search_verify/.) 350
neurons, shuffled and unlabelled for Luna: 100 drawn from the top 1% by the feasibility score but in no map; 100
layer-matched random; 100 size-matched; 50 known concept neurons (new in coverage-4) as positive controls. Luna read
each neuron's 15 strongest positive and negative web firings and its 20 most-pushed tokens each way, and named one
specific concept (with 8 questions) or none. Qwen then answered the questions; with >= 3 recalled, the neuron was
zeroed everywhere and the drop in log-odds compared with 2 random same-layer neurons. Verified = drop >= 1.0 and above
the pooled null's 99th percentile (0.17). Criteria stated before running.

| group | n | Luna named a concept | verified |
|---|---|---|---|
| top-ranked, unmapped | 100 | 4 | 0 (best: JavaScript, +0.86) |
| random, unmapped | 100 | 3 | 0 |
| size-matched, unmapped | 100 | 3 | 1 (Talc, L10:11856, +2.02) |
| known concept neurons | 50 | 9 | 4 |

- **Criterion not met** (top 0 vs random 0), so the pilot can't judge the ranking.
- **Verification works:** when Luna named a known neuron's OWN concept (4 times: Netherlands +7.43, The Hunger Games
  +3.14, Guns N' Roses +2.12, Minecraft +1.86), all 4 verified, against nulls of at most 0.18. When it named a different
  concept (5 times, e.g. 'the number three' for an F-35 neuron), none did.
- **The proposal step fails:** Luna found the concept for only 4 of 50 known concept neurons (8%). This fits the
  firing judge (concept neurons are mostly polysemantic in web text): what a concept neuron is FOR doesn't show in where
  it fires in ordinary text, except for the few dedicated ones. One new concept neuron turned up outside all maps (Talc,
  in layer 10).
- **Reading:** search-then-verify needs a proposal step that asks Qwen in the context where concept neurons act, not
  web text. Next design: a 'concept atlas' - run Qwen on cloze prompts for thousands of names (as in P1: the Wikipedia
  opening with the name blanked) and record every neuron's activation at the answer position; each neuron's
  top-activating names become its proposals; verify by zeroing on those cloze prompts.

## 17 Oct 2026 (night): the concept atlas - asking Qwen which names its neurons serve finds new concept neurons (EXPLORATORY; research branch)
(`concept_atlas.py` fetch / build / propose / verify_fast / verify_gate; all output in D:/already-clean-data/concept_atlas/,
summaries copied to results/concept_atlas/.) Fixes the step that failed in search_verify.py (proposing from web text).
- **Atlas:** 6,061 English Wikipedia names (known in >= 16 language editions, plus our Wikipedia-titled mapped
  concepts; disambiguation / list / 2024+ / adult-intro pages excluded). Up to 4 cloze prompts per name from its intro,
  as in P1. Qwen recalled 2,419 prompts for 1,514 names (P >= MIN_P), and every one of the 442,368 neurons was recorded
  at the answer position (7 GB, fp16). Fetch 25 min, build 5 min.
- **Proposals:** per neuron, the names that make it fire unusually strongly (z across names, either sign). Check on
  2,125 known concept neurons (mapped top-5) whose concept is in the atlas: their own concept is their top-1 name in
  **21.0%** (chance 0.13%) and in their top-5 in **43.2%** (chance 0.66%). From web firings, Luna had found 8%.
- **Verification** on INDEPENDENT prompts (Luna-written quiz questions for the proposed name, not the cloze prompts);
  zero the neuron everywhere; verified = recall drop >= 1.0 log-odds and above the null's 99th percentile (0.09; 2
  random same-layer neurons each).

| group | n | tested (recalled >= 3 questions) | verified |
|---|---|---|---|
| top: unmapped neurons with the most name-specific atlas profile (one per name) | 100 | 74 | **16** |
| random unmapped neurons (layer-matched), their own top name | 100 | 68 | **0** |
| known concept neurons, their top atlas name | 100 | 87 | 32 |

- **Criterion NOT MET on its absolute bar** (top >= 20%: 16%), met on the ratio (top >= 3x random: 16 vs 0). Of the
  top neurons that could be tested, 22% verified.
- **16 new concept neurons, in no earlier map**, some very strong: Algebraic notation (chess) L33:3837 +14.2,
  24 Hours of Le Mans L31:7680 +10.2, NGC 7319 L27:6441 +6.3, Hydrogeology L29:8015 +6.0, Selenium tetrachloride
  L32:3220 +5.7, Accelerometer +2.8, Ottoman Empire (a second neuron) +2.4, CD-ROM +2.4, Ketamine +2.1, Sandra Day
  O'Connor +2.0, Greek alphabet +2.0... These are concepts we never chose to ask about: the model pointed to them.
- **Efficiency (James: take the loop's lessons):** the first verify ran Luna, then the GPU, one name at a time with tiny
  batches (GPU flickering); stopped and rebuilt as verify_fast: Luna writes questions in 8 background threads while
  the GPU takes names as they arrive (append-only saving), recall for all forms x questions of 8 names in one pass,
  every knockout measured against an as-is copy in the same pass. 300 neurons in 4.7 min; GPU busy 19%, so this
  stage is now Luna-bound (Luna's hidden reasoning: 197k output tokens for 228 calls, ~$0.10). With questions cached,
  the GPU ran at 100%.
- **Gate (batched vs unbatched path), recorded honestly:** the pre-set rule (Spearman >= 0.95 between the two paths'
  drops, same verdict >= 95%) was NOT met as written. First run: Spearman 0.82, verdicts 29/30, max difference 0.91;
  the difference came from which borderline questions each path counted as recalled (P near the 0.2 cut-off flips
  with batch shape), so the recalled question indices are now saved per neuron. On identical question sets:
  verdicts 30/30, max difference 0.15, Spearman 0.79 - the low rank agreement is among drops near 0, where order is
  bf16 noise (null p99 0.09). Proceeded on the verdict agreement; the Spearman rule was a poor choice for drops that
  are mostly ~0.
- **Caveats:** one sample of 100 per group; Luna's questions can drift from the exact name (some untested names); the
  'known' group verifies on its TOP atlas name, which is its own concept only ~21% of the time; layer-35 neurons
  dominate some groups.

## 18 Oct 2026 (overnight): loop v2 run 1 - 5,000 Vital Articles concepts (research branch)
(LOOP_V2.md; frame / prepare / map / report scripts in src/loop_v2/; raw results 32 GB in
D:/already-clean-data/loop_v2/run1/; summaries in results/loop_v2/run1/.)
- **Run:** 5,000 concepts from Wikipedia Vital Articles Level 4 (section x fame, seed 71). Prepare 26 min (96 Luna
  workers, 0 failures, Luna ~$9.40). Map through the gpu-fan-out runner: 250 batches on up to 8 rented 3090s + the local
  5090, ~$12.30 of RunPod, 0 failed concepts; 3090 52 s and 5090 26 s per recalled concept.
- **Coverage:** 4,194 mapped (84%); by fame 95% (128+ editions) -> 57% (<32); by section 70% (History) - 90%
  (Geography); named 85%, common 83%.
- **Small handful holds across knowledge:** k_mean <= 5 in 1,705/1,722 named and 2,448/2,472 common concepts
  (median 2 named, 1 common).
- **Null clean:** median |null drop| 0.028 (p95 0.077) vs median top-1 drop 3.29; no concept with null >= 20% of top-1.
- **P1 replicates (self-check, 200 concepts, independent Wikipedia cloze prompts):** top-5 zeroed, median drop +7.42
  vs random-5 +0.003; flips >= half the prompts in 164/200 (P1 on the earlier maps: +5.59, 66%); named +6.96,
  common +7.80.
- **P4 replicates (self-check, families = Wikidata P31, first half of batches -> second half):** median own/other
  4.38 (pre-registered bar 3), 415 vs 251 (p = 2e-10) - weaker than the earlier 9.18. Strong for literary works,
  chemical elements, cities, historical countries, languages; NOT for people (Q5: own median 0), unlike the earlier
  people-heavy sample (historical figures here vs contemporary celebrities then - open). With Vital Articles SECTIONS
  as families the check is weak (517 vs 473): sections are too broad.
- **Named vs common:** a 'say' neuron (answer in its top-10 pushed tokens) in the top-5 for 62% named vs 73% common.
  Not a tokenisation artefact (common multi-token answers 74%). Partly section composition (People 46%, all named),
  but within sections common > named in 9 of 10 (+4 to +22 points). On the 230 concepts where Luna's blind label
  agrees with ours it reverses (79% vs 73%, small sample). Reading: modest, probably real; historical people often lack
  a single 'say' neuron.
- **Tags:** frame.py's named/common tag agrees with Luna's blind label on 273/300 (91%); disagreements are borderline
  (languages, peoples, fields, taxa).
- **Sharing:** 39,494 distinct neurons in any top-30 (8.9% of all), 9,860 in any top-5; median 10% of a concept's top-10
  in no other map's top-30 (a different metric from P2's, same direction: 'dedicated' shrinks as coverage grows);
  16,492 neurons shared by >= 2 maps, 29% of them within one section.
- **Heads:** most often among a concept's 5 most damaging single-head knockouts: L35H25 (1,337 concepts), L34H28
  (1,098), L19H29 (951), L20H9, L23H10, L35H24 - the late re-reader / turn-boundary copier / meaning-carrier set again.

## 18 Oct 2026: shared neurons across 4,194 maps - same job? same inputs? (run 1; research branch)
James: neurons used by several concepts - same job each time, the same earlier neurons, or polysemantic?
(`shared_neurons.py`, CPU; `feeders.py`, GPU ~25 min; outputs in D:/already-clean-data/loop_v2/run1/, summaries in
results/loop_v2/run1/.)
- **Census:** 3,010 neurons sit in the top-10 of >= 3 maps (median 4 users, max 874); mostly layers 30-35 (1,764) and
  24-29 (741), with a cluster at 6-11 (357).
- **Same job, mechanically:** a neuron's push direction is fixed by its weights x the sign of its activation. That
  sign agrees across ALL its users for 79% of shared neurons (median agreement 100%). Roles: 1,709 are attribute /
  late-other for every user, 629 'say' for some users, 227 'say' for all, 445 early.
- **Three kinds (readable from the push tokens and users).** NOTE (James, 18 Oct): the names below ('rivers',
  'happily ever after', 'Indonesia', ...) are my INFORMAL readings of two pieces of data - the tokens a neuron's weights
  push and the concepts whose maps use it - not systematic labels; 'happily ever after' is shaky (Romeo and Juliet
  doesn't end happily; 'fairy-tale / romance stories' fits better). The push tokens and users are the evidence (in
  shared_neurons.json); any future labels need a stated method (e.g. blind labeller + agreement check).
  Likewise 'packed unrelated meanings' (Czech + Xi'an) is untested: it may be a shared surface feature (sound or
  spelling, cf. the cats/cars 'V shape' example) rather than two meanings.
  - *topic jobs*, small coherent groups: rivers (7 rivers and a lake), Indonesia (Bali, Aceh, clove, the
    language, the revolution), dance, refrigeration, 'happily ever after' (Cinderella, Aladdin, Romeo and Juliet),
    genus endings (-opsis, -occus), Russian first names (Chekhov, Rachmaninoff, Pasternak);
  - *spelling jobs*: 'v' (Village, Virus, Vegetarianism), 'm' (Maze, Molybdenum, Mackerel), 'New', 'Rock';
  - *generic machinery* used by hundreds of unrelated concepts: a digits/punctuation writer (615 concepts) and
    junk-token pushers (L34:1452, 847; L20:7473, 874) - answer-format machinery, not meaning.
- **Related users:** they share a Vital Articles section 67% of the time vs 37% by chance (first Wikidata type: 33% vs
  33% - too fine-grained a label). Some packed neurons serve unrelated meanings, e.g. L29:4682 pushes 'Czech' / 'Prague'
  AND 'Xi'an' (users: Hussite Wars, Wends, Xi'an).
- **Same earlier neurons? (feeders.py):** for 100 shared neurons x their users (497 pairs), each neuron's 20 strongest
  earlier-layer feeders (activation x gradient of its activation, summed over positions) per concept. Jaccard overlap:

| pair | median |
|---|---|
| same TOPIC neuron (3-10 users), two of its users | 0.111 (above chance for 80/80) |
| same WIDELY shared neuron (>= 100 users), two users | **0.445** |
| same concept, two different shared neurons | 0.111 |
| chance (different neuron, different concept) | 0.000 (mean 0.011) |

  Prediction half wrong: machinery neurons were predicted to vary their feeders by concept; instead they are fed by
  the SAME upstream neurons whatever the concept (12 of 20 in common) - they respond to the question-answering
  format, not to the concept. Topic neurons share a minority of feeders (2 of 20) far above chance; the shared ones are
  themselves readable attribute neurons: the Fellini / Woody Allen / La Dolce Vita neuron <- 'Italy' + 'film /
  director' neurons; the civil-rights neuron <- 'Martin' + 'strike / protest'; the Qing / Habsburg / Plantagenet
  neuron (pushes family names) <- a 'families' neuron; the seas neuron <- a 'rivers' neuron.
- **Reading:** shared neurons are mostly NOT polysemantic in this setting. Machinery neurons do one task-level job with
  fixed inputs. Topic neurons do one output job, reached from partly shared attribute feeders plus concept-specific
  ones - a visible hierarchy (Italy + film -> Italian cinema -> Fellini). A minority appear to pack unrelated meanings (untested: possibly shared surface features).
- **Caveats:** top-20 feeders by first-order attribution (not knockouts); the same-concept baseline is small (27
  pairs); 'topic' vs 'machinery' split by user count, not a test of meaning.

## 18 Oct 2026: forcing one neuron - its most common expression, read with minimal bias (pilot; research branch)
James: like the original "Eiff eiff eiff" when a say-it neuron was clamped hard - garbage is fine; what REPEATS across
many backgrounds is the neuron's own expression. (`neuron_readout.py`; output D:/already-clean-data/neuron_readout/,
summary in results/neuron_readout/pilot1.txt; GPU ~20 min.) The target is clamped at 1x / 4x / 16x its own web-text
peak, at the last context position and every generated position, in three backgrounds: 200 random web snippets; a
bare newline with 50 RANDOM unrelated neurons clamped to their peaks (a new random set per repeat, 200); a bare newline
alone. Read: the mean next-token log-prob change vs the unforced background, and 8 greedy tokens. 10 targets + 3 random
control neurons.
| neuron | forced readout (x1-x4; [x16]) |
|---|---|
| Moon L31:9683 | moon, 月, lunar, full, "full-moon moon-moon-moon" [明明明 - 'bright', as in 明月 'bright moon'] |
| Eiffel Tower L31:10160 | par, _par, -par (the start of 'Paris'?) [par-par-par] |
| Eiffel Tower L26:11492 | tower, landmark, 塔, 地标, monument [leaning leaning leaning] |
| Netherlands L33:5099 | Nether, ETHER, hol (Holland), Pays (Pays-Bas) [hol hol hol] |
| 'rivers' L23:12257 | River, rivers, 河水, downstream, streams [River River River] |
| 'happily/prince' L29:6647 | happy, happily, prince, 王子; generates "... ever after" at x4 [南瓜 pumpkin, 灰 ash (Cinderella), 王子] |
| 'Indonesia' L29:9192 | Indonesian words (dalam, dengan, jumlah, ber-), Jakarta |
| Czech/Xi'an L29:4682 | Xi'an's local geography (西安, 陕西, 鄠, 灞 and 浐 rivers, 潏, 咸阳 Xianyang, 兵马 Terracotta Army) AND Czech, Prague, Slovak |
| 'v' L34:553 | v, )v, =v [_AV] |
| digits L33:10933 | digits / spaces on web text; Russian fragments with no context |
| random controls | 'thanks / 感谢'; '先进 / 优秀' ('advanced / excellent'); one noisy |
- **Low bias (for these hand-picked strong neurons only - see the correction in the fast-readout entry below):** the
  top next-token shifts are nearly the same across real text, random-neuron noise and no context. Strength matters: x1 is often drowned by context in generation, x4 reads best, x16
  collapses to one repeated token - often a telling one ('leaning', 'pumpkin', 'bright').
- The informal 'happily ever after' reading (marked shaky earlier) is supported: forced, it generates 'happy ... ever
  after', 'prince', 'pumpkin'.
- Czech/Xi'an: NO shared sound visible (no Ch-/Sh-/Xi- sound-alikes) - two detailed, separate clusters (very local Xi'an
  places; Czech/Prague/Slovak). Looks like genuinely packed unrelated meanings; an unseen shared link is not ruled out.
- Random control neurons also have readable expressions when forced: this is a candidate UNBIASED labelling method
  (same procedure for every neuron, no labeller), to run over many neurons.

## 18 Oct 2026: fast forced readout - validation against known answers and across layers (research branch)
(`fast_readout.py`: 32 web + 16 empty backgrounds, target forced at the last position only, next token; weighted words
= mean dlogp, consistency across backgrounds, mean forced probability; per-background top-20 with forced and unforced
probabilities saved; output D:/already-clean-data/fast_readout/; ~110-125 ms per neuron on the 5090.)
- **Known answers (2,577 neurons, run-1 concepts):** for 2,856 'say'-neuron/concept pairs, the concept's answer token is
  in the forced readout's top-10 73% (top-50 87%, median rank 5); for all picked neurons 50% (top-50 64%). The
  weights-only direct push does at least as well (top-10 68% for all picked; 100% for 'say' neurons, by definition)
  - forcing does not identify late neurons' concepts better than the near-free weight reading.
- **Layers (50 random neurons per layer):** the top word is consistent across backgrounds only late on: layers 28-35
  91-100% (x4) / 97-100% (x16); 24-27 50% / 69%; 0-23 2-16% / 6-36%. A single early or mid neuron forced at one
  position has no consistent expression even at 16x (its effect goes through later layers and other positions).
- **Context dependence (James: do the words depend on what else is active?):** yes, substantially. Per-background
  top-10 words overlap little across web backgrounds (1 - mean pairwise Jaccard: layers 32-35 0.59, 28-31 0.83, earlier
  ~1.0): the top word is stable for late neurons, the rest of the list shifts with context.
- **CORRECTION to the pilot's 'low bias' claim:** it held for hand-picked strong neurons; for random neurons the top-10
  from web vs empty backgrounds overlap only 0.1-0.2 (Jaccard) even in late layers.
- **Reading:** the cheap forcing readout gives a reliable job label for late neurons (roughly the top quarter of the
  model), where the weight reading is about as good; early / mid neurons need a different readout - e.g. describe them
  by which late neurons they drive (downstream fan-out), consistent with the composition seen in feeders.py.

## 18 Oct 2026: fan-out readout for early / mid neurons - smoke test (research branch)
James: describe early / mid neurons (no consistent forced word readout) by what they drive: hold the context fixed,
vary only the target (dose series -4..8 x peak; last position or every position), read all layer-24..35 neurons and the
next-token log-probs at the last position; 16 web texts (two halves) and, for concept-specific targets, the concept's own
recall prompts. 29 targets: 11 concept-specific early neurons (e.g. Clovis I L12, Strawberry L4, Volleyball L8), 8 widely
shared early neurons, 10 random (layers 4-23). (`fanout.py`; outputs D:/already-clean-data/fanout/ - smoke.jsonl = v1,
smoke2.jsonl/.txt = v2; v1's text summary was overwritten by a naming bug, its full data is intact.)
- **v1 (raw slopes):** fan-out exists and is partly repeatable - top driven neurons keep their sign in 75-94% of texts,
  roughly linear in dose (R^2 ~0.5-0.7), split-half top-20 Jaccard 0.2-0.5 (shared early neurons most repeatable),
  effects 5-20x larger on the concept's own prompts. But the ranking was swamped by a few generic last-layer hub
  neurons (L35:9510, L35:8724, L35:11333, L35:9490 - junk pushes) that respond to ANY perturbation.
- **v2 (each late neuron's slope in units of its natural spread, minus the random targets' mean):** the hubs drop out;
  driven neurons are consistent (88-100% sign) but mostly in layers 24-27 and themselves unreadable by their pushed
  words. Glimpses: Antigonus I Monophthalmus -> a 'Rome / 罗马' neuron plus 'mon' / 'mat' spelling neurons; widely shared
  early neurons -> 'Title-Case term' neurons (Theory, Approach, Responsibility, '-Based', '-Level') - consistent with
  answer-format machinery. Most concept-specific early neurons get no readable fan-out (Strawberry, Johann Strauss II,
  Monopoly, Marathi).
- **Reading (smoke test):** early neurons' downstream effects are repeatable but do not translate into a word-level
  job one step later - the description problem moves up a layer. Options: restrict to READABLE late endpoints (layers
  28-35) and follow chains; or describe early neurons by the concepts they matter for (from the maps).

## 18 Oct 2026: question-side test + specificity gradient - 'say it' vs meaning, and how specific (research branch)
(Queued after run 1, design and predictions in RESEARCH_STATE.md before running. `specificity.py`; outputs
D:/already-clean-data/specificity/, summary + neighbours in results/specificity/; Luna ~$0.08, GPU ~15 min.)
15 mapped landmarks. Neighbours from Wikidata, 3 per level: L1 notable physical places within 3 km (coordinates; first
pass picked abstract items with Paris coordinates - UNESCO, the French language - so restricted to structures /
mountains / islands / squares / parks via the type hierarchy); L2 same 'instance of' type > 500 km away (falling back to
the source's other types when the first had no distant matches - broader, e.g. St. Peter's -> Eiffel Tower, Burj
Khalifa); L3 same Vital Articles section, different type; L4 a different section. Luna wrote 8 FACT questions per source
(the name in the question, short factual answer, e.g. "In which city is the Eiffel Tower?" -> Paris) and 8 naming
questions per neighbour; only questions Qwen answers were used (P >= 0.2). Zeroed at every position, each against an
as-is copy in the same pass. Median log-odds drop:

| question set | k (breaks naming) | top-5 | top-5 late | top-5 early | random-5 (x2) | another landmark's top-5 |
|---|---|---|---|---|---|---|
| source NAMING (15) | **3.62** | **8.03** | 6.44 | 1.44 | -0.01 / -0.01 | 0.16 |
| source FACTS (15) | **0.03** | 0.68 | 0.30 | 0.97 | -0.04 / -0.05 | 0.02 |
| neighbours L1 same place (25) | 0.02 | 0.17 | 0.17 | -0.01 | 0.02 / 0.00 | 0.10 |
| neighbours L2 same kind (30) | 0.01 | 0.10 | 0.07 | 0.03 | -0.01 / 0.00 | 0.05 |
| neighbours L3 same section (36) | 0.00 | 0.00 | 0.01 | 0.02 | 0.01 / 0.00 | -0.01 |
| neighbours L4 unrelated (37) | 0.00 | -0.02 | -0.02 | 0.00 | 0.00 / -0.01 | -0.03 |

- **'Say it', not meaning (predictions held):** the neurons that break NAMING a landmark (k, median drop 3.62) leave
  questions ABOUT it essentially intact (0.03). Late neurons hurt naming most (6.44) and facts little (0.30); early
  neurons the reverse pattern (naming 1.44, facts 0.97) - the early cluster carries more of the meaning.
- **Exceptions (facts damaged too):** Burj Khalifa (facts 8.10), St. Peter's Basilica (6.73), Machu Picchu (3.86) - to
  check whether their fact answers share tokens with the name or depend on name-linked attributes.
- **Specificity gradient (prediction held):** damage to neighbours falls with distance, L1 0.17 -> L2 0.10 -> L3 0.00 ->
  L4 -0.02 (vs 8.03 on the source itself: neighbours lose ~1-2%). Large exceptions where neighbours share an attribute:
  Sydney Opera House -> Sydney Harbour Bridge / Sydney Tower 5.40; Burj Khalifa -> other skyscrapers 5.98; Golden Gate
  -> other bridges 4.74; St. Peter's -> Sistine Chapel 3.61; Statue of Liberty -> towers 2.22; Sagrada Família -> Park
  Güell / Casa Batlló (Gaudí) 1.68; Machu Picchu -> Huayna Picchu 1.57. Another landmark's top-5 also touches L1/L2 a
  little (0.10 / 0.05) - shared landmark-attribute neurons.
- **Reading:** a concept's map is mostly a specific 'say it' mechanism (naming), not where the knowledge ABOUT it lives;
  the meaning sits more in early neurons and elsewhere. Its spill-over follows shared attributes (place, kind, architect).
- **Limits:** small n per level (many neighbours had < 2 recalled naming questions); Luna-written fact answers; one
  forced 'answer in as few words as possible' format for facts; 15 sources, all landmarks.

## 18 Oct 2026: job-finding for knowledge neurons - an entity x relation causal grid + position test (research branch)
James: naming maps are 'say it' machinery; find mechanisms for the job of earlier neurons, where knowledge seems to live.
(`facts.py`; outputs D:/already-clean-data/facts/, summary / entities / classes in results/facts/; Luna ~$0.01; GPU ~35 min.)
Ground truth: 30 landmarks x 5 relations (city, country, designer, century built, main material), 3 FIXED phrasings per
relation (same wording for every landmark), answers from Luna ('Unknown' excluded), kept where Qwen answers >= 2 of 3:
111 facts (city 30, country 30, century 25, material 14, designer 12). Per fact: attribution of the answer -> top-100 ->
each zeroed alone -> fact map. Grid: each fact's top-3 neurons (drop > 0.2) zeroed on EVERY fact -> mean drop on the same
landmark's other facts / the same relation for other landmarks / neither -> class (threshold max(0.2, 25% of its own-fact
drop), fixed before looking): entity, relation, entity+relation, fact-specific, generic. Position: each zeroed only at the
landmark's name tokens vs only at the last position. Predictions stated before running.
- **(a) 'fact neurons sit earlier than naming neurons': FAILED as stated.** Fact maps' top-5 have the same layer profile
  as naming maps (median layer 30; 19% vs 16% below layer 20). Mapping by the answer's log-prob again finds late neurons
  that SAY the answer ('Paris', '19th').
- **(b) 'entity neurons are early and act at the subject tokens': HELD.**

| fact neurons | n | zeroed everywhere | only at the name tokens | only at the last position | entity / relation / fact-specific / generic |
|---|---|---|---|---|---|
| layers 0-9 | 40 | 2.72 | **2.60** | 0.00 | 15 / 1 / 24 / 0 |
| layers 10-19 | 19 | 1.58 | **1.46** | 0.01 | 8 (+1 both) / 0 / 10 / 0 |
| layers 20-27 | 31 | 1.81 | 0.04 | **1.31** | 0 / 11 / 17 / 3 |
| layers 28-35 | 243 | 2.64 | 0.00 | **1.60** | 16 (+8 both) / 50 / 158 / 11 |

- **(c) 'relation neurons sit mid-to-late': HELD.** E.g. L30:8291 is a 'century built' neuron: zeroing it costs ~3.2
  log-odds on the century fact of every landmark tested (Taj Mahal, Tokyo Tower, CN Tower, Blue Mosque, Versailles...)
  and nothing on their other facts. Early entity examples: L8:9458 (Blue Mosque: its facts drop, other landmarks'
  don't), L1:4818 (Sagrada Família), L1:3960 (Statue of Liberty).
- **Entity neurons are mostly not naming neurons:** 6 of 32 appear in their landmark's naming top-5, essentially one
  neuron - L29:5436, a 'Vatican' neuron (all Sistine Chapel / St. Peter's facts collapse without it, drops 15-28: city,
  country, etc. route through 'Vatican').
- **Reading:** a working job-finding mechanism for early neurons: place a neuron in an entity x relation grid causally
  and find where it acts. Early (layers 0-19) knowledge neurons are ENTITY neurons acting at the entity's own name
  tokens; mid/late ones extract RELATIONS (or say specific answers) at the answer position. This matches the published
  picture of factual recall (subject enrichment in early MLPs at the subject token, attribute extraction late; Meng et al.
  2022, Geva et al. 2023), now neuron by neuron in Qwen3-8B.
- **Limits:** only 59 early neurons in the fact maps' top-3s (maps still favour late answer-sayers); 5 relations, all
  landmarks; class thresholds arbitrary (stated); Luna's answers; fixed templates.

## 18 Oct 2026: direct entity neurons, scaled to three kinds - and how shared / polysemantic they are (research branch)
(`facts2.py` entity stage, `entity_sharing.py`; outputs D:/already-clean-data/facts2/, summary in results/facts2/.)
90 entities (30 landmarks, 30 people, 30 works; two near-duplicate works - 'Odyssey' / 'The Odyssey', 'Divine Comedy' / 'The
Divine Comedy' - kept as separate entries), 5 relations each with fixed phrasings; 333 facts kept (Qwen answers >= 2 of 3).
- **Direct entity neurons:** per entity, attribution summed over ONLY its name-token positions and all its facts, across
  all 36 layers -> top-100 -> each zeroed ONLY at the name tokens -> mean drop over the entity's facts. Top-3 per entity
  with drop > 0.2 (218 neurons, 86 entities): **median layer 8; 89% below layer 20** (landmarks 8, people 8, works 8.5).
  Predicted (median < 20): held.
- **Specific in use:** zeroed at the name tokens of every fact: own entity's facts drop a median 0.58, other entities of
  the same kind 0.015, other kinds 0.008; **209/218 (96%) >= 5x** more for their own entity (predicted: held). Strongest:
  L8:9458 Blue Mosque 9.6; L7:8413 Ulysses 8.7; L0:1550 The Starry Night 6.7; L1:4818 Sagrada Família 6.3; L7:990 Genghis
  Khan 4.7; L8:334 Sistine Chapel 4.1.
- **But shared with a few others (James: towards polysemanticity):** 119/218 (55%) also matter for >= 1 other entity
  (another entity's facts drop > max(0.2, 25% of own); 1 other: 51, 2-3: 46, 4-6: 22); only 36 of those 119 share only
  with the same kind. Coherent groups at layers 7-13: Leonardo -> The Birth of Venus, The Night Watch, The Last Supper,
  Mona Lisa, Van Gogh; War and Peace -> Ulysses, Odyssey, Moby-Dick, Faust; bridges / skyscrapers (L9:410, L12:10025).
  Mixed groups at layers 21-23: L21:1310 -> Brooklyn Bridge, Statue of Liberty, The Scream, Sagrada Família, The
  Brothers Karamazov, Eiffel Tower; L22:5427 -> Bach, the Blue Mosque, Beethoven, Jane Austen. Map overlap: 184 of 1,432
  neurons in the entity maps' top-20 appear for >= 2 entities.
- **Input side - broad:** in web text (atlas, 20 strongest firings) entity neurons are mostly NOT about their entity:
  median 5% of firings mention it; 104/218 never. What they fire on: attributes ('Turkish' / 'Turkey' for the Blue Mosque
  neuron; 'Vatican' / 'conclave' for the Sistine Chapel one), word parts or meanings of the name ('Sac[red]' for Sagrada
  Família; ' night' for The Starry Night; 'Div[ine]'), a few on the name itself (Ulysses 100%).
- **Not simple word detectors:** the Starry Night neuron (fires on ' night' everywhere) does nothing for The Night Watch
  (0.01); the Divine Comedy neuron nothing for 'The Divine Comedy' (-0.02); the Sagrada Família ('sacred') neuron nothing for
  other churches. Exceptions follow shared attributes: Sistine Chapel neuron -> St. Peter's 1.04 ('Vatican'); Blue Mosque
  neuron -> Hagia Sophia 0.29 (Istanbul / 'Turkish').
- **Reading:** early entity neurons are broad on the INPUT side (they respond to attributes and name-parts across web
  text - polysemantic in the usual sense) but specific in USE (in this task each matters for its own entity, plus a few
  sharing an attribute). The same pattern as concept neurons ('packed triggers, one job'), now for knowledge neurons;
  what looks entity-specific is a combination: the right word-part / attribute neuron firing in the right context.

## 18 Oct 2026: fact knowledge as shared machinery - answer-value and relation neurons (facts grid re-read; CPU)
James: by 'knowledge' he meant the FACTS (location, height...), and whether those show up as shared machinery. Re-read of the
landmark facts grid (each fact neuron zeroed on every fact): for each neuron, its mean drop on OTHER facts with the SAME
answer vs same-relation facts with a DIFFERENT answer vs the rest (219 neurons whose fact shares its answer with others).
- **Answer-value machinery (124/219):** neurons that matter for every fact with the same answer and not for the same
  question with other answers (> 3x). A '19th century' neuron, L33:2629: zeroed, the century fact of every 19th-century
  landmark drops (Arc de Triomphe 10.1, Sagrada Família 8.9, Brooklyn Bridge 8.5, Statue of Liberty 8.2) and other
  centuries do not (-0.03). 'Vatican City' L29:5436 (St. Peter's and Sistine Chapel, city AND country, drops ~26); 'marble'
  L31:6383 (Parthenon and another marble landmark, 7.8). Layers 29-33.
- **Relation machinery (45/219):** neurons that matter for one question type whatever the answer - L30:8291 'century
  built' (~3.2 on every landmark's century fact, 12th or 20th alike).
- **Reading:** the fact itself is not one neuron but a factorised lookup: entity neurons early at the name (rows),
  relation neurons mid-late (columns), answer-value neurons late (the values) - and the value neurons are shared by every
  entity with that value. Not yet tested: the chain (does zeroing an entity neuron at the name lower the value neuron's
  activation at the answer position?) and numeric facts (height).

## 18 Oct 2026: scaled facts grid (landmarks, people, works) + the fact route, causally (research branch)
(`facts2.py` map / grid / position; `chain.py`; outputs D:/already-clean-data/facts2/, summaries in results/facts2/.)
**Scaled grid (333 facts, 11 relations, 3 kinds) replicates the landmark picture:**

| fact neurons | n | zeroed only at the name tokens | only at the last position | job classes |
|---|---|---|---|---|
| layers 0-9 | 109 | 2.90 of 2.90 | 0.00 | entity 50, fact-specific 49, relation 9 |
| layers 10-19 | 44 | 1.46 of 2.12 | 0.00 | fact-specific 31, entity 10 |
| layers 20-27 | 139 | 0.01 | 1.84 | **relation 63 (45%)**, fact-specific 43, generic 22 |
| layers 28-35 | 707 | 0.00 | 1.75 | fact-specific 457, relation 182 |

- Fact maps' top-5 again have naming maps' layer profile (median 30; 18% vs 16% below layer 20): the 'fact neurons
  earlier' prediction fails again as stated - answer-ranked maps find answer-writers.
- By kind: landmarks and works match cleanly (early neurons act at the name: 2.21 of 2.30, 2.77 of 2.82); PEOPLE are
  weaker - fewer early fact neurons (25, mostly fact-specific, 4 entity), only about half their effect at the name (1.55 of
  2.90), and relation neurons dominate their mid layers (35 of 58) - another people anomaly (cf. no P4 family signal, fewer
  'say' neurons for historical people).
- Relation neurons by relation: century 124, country 64, medium 22, creator 13, language 10 - the 'century' neuron
  L30:8291 again strongest (same-relation drop ~4.2).
**Fact route (James: entity -> value):** for each shared answer-value neuron V (151, from the scaled grid) and every fact
with V's answer, the entity's direct entity neuron(s) zeroed ONLY at its name; V's activation at the answer position
measured (sign-aligned), vs a random same-layer neuron and another entity's neuron (controls).

| zeroed at the name | V's relative activation drop (median) | answer log-odds drop | pairs where entity > both controls |
|---|---|---|---|
| entity's top-1 entity neuron | 2.2% | 0.59 | 430/647 (66%) |
| entity's top-5 together (follow-up) | 7.3% | 1.78 | 465/657 (71%) |
| random neurons / another entity's | 0.0% | 0.00 | - |

- **Prediction (median V drop >= 20% for the top-1) FAILED.** The route is real and completely specific (controls exactly
  0) but each early entity neuron supplies only a slice of what drives the shared value neuron; five give ~3x one. Large
  cases: van Gogh -> 'painter' (> 100%), Symphony No. 9 -> '19th', Sistine Chapel / Columbus -> '15th', Sagrada Família ->
  'Antoni Gaudí'.
- **Reading:** entity -> value is a causal, specific route, but an entity's knowledge is distributed over many early
  neurons (and the value neuron has other inputs: relation heads, other neurons). Consistent with the factorised-lookup
  picture; not a single-wire one.

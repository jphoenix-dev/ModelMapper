# Results log

Every run gets logged here, failures included. Newest at the bottom.

---

## 15 Oct 2026: setup and pipeline debugging (Qwen3-0.6B; not evidence)

**Setup**
- Environment: `.venv`, PyTorch 2.11 (cu128), transformers 5.20, RTX 5090.
- Hook check: SiLU(gate) × value reproduces the input to down_proj to within 0.17% of max activation, which is
  bfloat16 rounding. So we're reading the right number.
- `huggingface_hub` downloads stalled on this machine (as noted before in C:\dev\CLAUDE.md), so models are now
  fetched with resumable curl (`scripts/download_models.sh`) into `C:/dev/models/hf/`.
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

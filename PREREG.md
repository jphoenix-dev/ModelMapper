# Pre-registration

Written before looking at any data for the experiment concerned. Changes after data are seen get logged
at the bottom with a date and reason; we don't silently edit.

---

## Experiment GG: Golden Gate Qwen from raw neurons (frozen 15 Oct 2026, before step 1 on Qwen3-8B)

**Model:** Qwen3-8B, bf16. Pipeline debugging on Qwen3-0.6B doesn't count as evidence.

### Definitions
- **Selective unit:** a neuron whose contrastive score is ≥ 2 (mean firing on Golden Gate texts minus mean on its
  best-firing control group, in units of its spread), and which fires more on ≥ 80% of Golden Gate texts than on
  every single control text.
- **Clean pole:** a blind labeller (a Claude subagent that sees only examples, never this file or the essay) can give
  one explanation covering ≥ 80% of the top 40 corpus examples. The examples are judged after splitting off
  double-negative (gate−, value−) firings.
- **Explanation survives:** examples written from the explanation alone reach ≥ 30% of the neuron's corpus max at
  least 60% of the time, and near misses reach it at most 20% of the time.
- **Intruder test (Jev):** for each unit, 20 trials. Each trial shows 4 random examples from the unit's top 40 plus
  1 random corpus snippet, positions rotated. Accuracy = mean probability Jev gives the true intruder.
- **Golden-Gate-steered output:** Jev gives ≥ 0.5 that the reply "mentions or is preoccupied with the Golden Gate
  Bridge". Coherence is judged separately: Jev gives ≥ 0.5 that the reply is fluent English.

### Controls
- 20 random neurons from the same layers as the candidates.
- 10 planted mixtures: two neurons from the same layer, each with a clean but unrelated label, blended 50/50.
  This is a known polysemantic unit, to check our tools can see mess when it's there.
- Steering controls: the same number of random neurons from the same layers, clamped to the same multiple of
  *their own* max.

### Predictions

**Support for the essay** (all of these):
1. At least one selective unit exists, and at least one selective unit has a clean positive pole. Its theme may be
   coarser than the bridge (SF landmarks, famous bridges, the Bay Area) and still count.
2. Its explanation survives.
3. Jev intruder accuracy is ≥ 0.8 on the GG unit, while planted mixtures score clearly lower. The second part is a
   check on the tool, not on the essay.
4. Steering ≤ 10 neurons gives Golden-Gate-steered output on ≥ 50% of 20 neutral prompts at some strength where
   ≥ 50% of replies are still coherent. Random-neuron steering at the same strength gives ≤ 5%.

**Against the essay** (any of these):
- The most selective units have top examples split across ≥ 2 unrelated themes. "Unrelated" means the blind
  labeller can't name a shared pattern, and no route (sign regime, summary motif, a sensible step-by-step drift)
  explains the split.
- No selective unit exists, and only a direction spread over many neurons (> 50) steers toward the Golden Gate.
  This counts against "neurons are the right unit", though not as direct evidence of packing.

**Ambiguous:**
- Steering produces off-topic or incoherent text before it produces Golden Gate text, for both GG and random neurons.
  That's a sign our steering method is too crude, not evidence either way.
- Clean but coarse units whose steering gives "San Francisco" rather than the bridge. This is consistent with the
  essay's "coarser, not mixed" claim, but weak, because we went looking for it.

### Known weaknesses (stated in advance)
- **Selection bias:** we're hunting for one concept, so a clean result says nothing about the share of clean neurons.
- The blind labeller and the essay's labeller are both LLMs, and the planted-mixture control only partly addresses
  whether LLM labellers are too lenient.
- Jev's documented weaknesses include position bias in choice questions, which is why positions are rotated.

### Change log
- **15 Oct, after the 0.6B debug runs (no 8B data seen):**
  - The candidate score now measures the gap in units of the *control* texts' spread (the old version had a ceiling).
  - Candidates must reach ≥ 20% of their own web-text peak on GG texts ("fires hard"). The trigger was 0.6B comma
    neurons that looked selective while firing at ~6% of peak.
  - Added 15 "gg_unnamed" texts (bridge described, not named) as a word-vs-concept diagnostic. Reported only; not
    part of the selectivity definition.
- **15 Oct, after seeing 8B search results (exploratory, weaker evidence):**
  - Added a behavioural check (does Qwen name the bridge from the unnamed texts? 14/15).
  - Added an exploratory ranking by selectivity for *unnamed* texts. Its top hits (L26:1311, L25:6245) were added to
    the scan as "exploratory". Any conclusion about them is post hoc.
  - Planted mixtures defined concretely: a top positive candidate blended 50/50 with a random neuron from the same
    layer, each weighted by 1/peak.
  - Labeller gets the essay's reading tips (context, punctuation/summary motif, patterns over nouns). The planted
    mixtures measure any extra leniency this causes.
- **15 Oct, labelling method (decided after a debug run on 0.6B units, before any 8B labelling):**
  - Explanations are written by GPT-6 Luna (cheap and fast) instead of Claude subagents. Jev grades them, so the
    writer never grades its own coverage.
  - New specificity requirement, the essay's "explanations that can lose": the explanation is also graded against 20
    random tokens in random web text. **Clean** now = Luna verdict ONE, AND Jev finds ≥ 16/20 of the pole's examples
    fit, AND ≤ 4/20 random tokens fit. Trigger: in the debug run a random direction got a vague "discourse
    continuity" explanation that covered 17/20 of its examples but also 16/20 random tokens.
  - Jev's own blind verdict (ONE/TWO_RELATED/MIXTURE/NONE) is reported but not used. In the debug run it said ONE for
    everything, even with option order rotated.
  - Known limitation: Jev under-credits syntactic explanations (comma neurons got 9–15/20), so the coverage bar may
    be harsh on syntax-type units.

---

## Experiment CR: concept replication with the automated pipeline (DRAFT, to be frozen before the first run)

**Model:** Qwen3-8B, bf16. **Pipeline:** `src/concept_pipeline.py`. Luna writes 20 recall prompts and 8 controls per
concept; prompts are kept if P(whole answer) ≥ 0.2. Attribution is activation × gradient of log P(answer); ablation
uses k = 1, 2, 3, 5, 10, 20, 50 vs random neurons from the same layers; labels come from the atlas (blind Luna + Jev,
strict "clean" bar); steering uses the top 2/4/6 late-layer neurons × strengths 0.5–2× own peak, judged by Jev on 20
neutral prompts. Steering mode (clamp vs add, reply-only or not) is fixed to whichever wins on the Golden Gate
calibration run, which is chosen *before* the 20 concepts are run.

**Concepts (20):**
- Places/landmarks: Golden Gate Bridge (positive control), Eiffel Tower, Mount Everest, Sahara Desert, Venice, Tokyo
- People: Albert Einstein, William Shakespeare, Napoleon Bonaparte, Wolfgang Amadeus Mozart, Frida Kahlo
- Things: the Titanic, the Moon, coffee, the violin, chess
- Abstract/other: photosynthesis, democracy, the French Revolution, Harry Potter

**Negative controls:**
- 3 made-up concepts (the model can't know them): "Mount Velkaran" (a fictional Andean peak), "Professor Elsabeth
  Quorrin" (a fictional linguist), "the Draxwell Bridge" (a fictional bridge in Ohio). They must fail recall (< 5
  prompts kept). If they pass, the prompts leak the answer.
- Shuffled judge check: steer with concept A's best setting, judge "about concept B?" for 5 random (A, B) pairs.
  It must stay ≤ 0.1, or Jev is being fooled.
- Random-neuron steering control at each concept's best setting (built into the pipeline).

**Predictions (the essay's view):**
1. **Handful:** for ≥ 70% of recalled concepts, k50 ≤ 10 (ablating ≤ 10 neurons halves P(answer)), while 50 random
   neurons leave P(answer) ≥ 80% of baseline.
2. **Specific:** at k = k50, unrelated controls (the 4 "well-known facts") drop by ≤ 10 percentage points on average.
   Related controls may drop (composition predicts shared parts, e.g. bridges for Golden Gate).
3. **Legible:** across concepts, the top-10 neurons' "clean on at least one pole" rate exceeds the random-neuron rate
   from the GG-1 census (positive poles 50%, negative 15%; "either pole" to be computed the same way).
4. **Nudgeable:** ≥ 50% of recalled concepts reach nudgeability ≥ 0.25 (e.g. 50% on-topic × 50% fluent), with the random
   control ≤ 0.05.

**Against the essay:** k50 > 50 for most concepts (recall spread thin); or top neurons no more legible than random
ones; or nudgeability rarely beats random.

**Ambiguous:** concepts where recall uses mostly early-layer prompt-specific word detectors (as with "field" and
"Christopher" in GG-2); low nudgeability with a clean k50 (steering method limits, not the neurons).

**Known weaknesses:** Luna writes the prompts (it may favour some angles); Jev judges steering (known biases,
checked by the shuffled control); 8B only; first-pass atlas = 10M tokens (rare concepts may be under-represented in
labels).

**Added before freezing (15 Oct, ~22:55), all EXPLORATORY: recorded for later analysis, not part of predictions 1–4:**
attention-head attribution, head-ablation curve (with controls) and where the top heads look; top 200 neurons with
per-prompt attributions; 50 neurons pushing *against* the answer; per-token profiles of the top 20 neurons and top 10
heads; each top-20 neuron ablated alone; each top neuron's activation as a fraction of its peak on the recall prompts;
every steered reply with Jev's per-reply scores; code fingerprints (provenance).

**FROZEN 2026-10-15 23:10** (automatically, after the Golden Gate calibration and before any other concept ran).
Steering variant chosen by best Golden Gate nudgeability: `clamp_reply`. Concept list, controls and predictions above
are now fixed; later changes go in the change log.

---

## Experiment CEN: neuron census (frozen 15 Oct 2026 ~23:20, before any census labels were generated)

**Question:** what share of *typical* (randomly chosen) MLP neurons in Qwen3-8B have a clean job? This is the essay's
headline claim (~90% of poles have one main job in Qwen3-32B; ~7% two related jobs; ~2–4% arbitrary).

**Sample:** 1,000 neurons drawn uniformly at random over all 36 layers × 12,288 (seed 2026). Examples come from the
10M-token atlas (`data/atlas_Qwen3-8B.pt`); positive-pole double-negative firings are removed.
**Labeller:** `src/label_batch.py` (Luna writes an explanation and verdict per pole from 15 examples; Jev grades 10
examples per pole and the batched random-token check).
**Clean pole:** Luna verdict ONE, AND ≥ 80% of graded examples fit, AND ≤ 20% of random tokens fit.
**Leniency control (run when the GPU is free):** ≥ 50 planted mixtures, each two random neurons from the same layer
blended 50/50 with weights 1/peak, labelled the same way.

**Reported:** % clean per pole; verdict shares (ONE / TWO_RELATED / MIXTURE / NONE) as essay-comparable numbers;
both broken down by depth (layers 0–11 / 12–23 / 24–35); the planted-mixture clean rate.

**Predictions:**
- **Essay-style** (Luna verdict only): ONE + TWO_RELATED ≥ 85% of poles; MIXTURE ≤ 5%.
- **Strict bar:** I expect 40–70% of positive poles clean (based on GG-1's 20 random neurons: 50%). An honest guess,
  not the essay's claim.
- **Depth (essay: neurons narrow with depth):** clean share in the late third ≥ clean share in the early third.
- **Leniency check:** planted mixtures clean ≤ 10%. If higher, the bar is too lenient and the census number is an
  overestimate.

**Supports the essay:** essay-style ≥ 85% with mixtures mostly caught, and strict clean ≥ 70%.
**Against it:** essay-style < 70%, or strict clean < 40% with mixtures caught (i.e. the tools can see mess, and see
a lot of it).
**Ambiguous:** in between, or mixtures not caught.

**Known weaknesses:**
- Top 15 examples from 10M tokens only.
- Luna and Jev biases (Jev under-credits syntactic jobs, which pushes the strict bar down).
- 8B only (the essay's figure is for 32B).
- Middle-of-range activations are not covered by this step (planned as a follow-up on a subset).
- **15 Oct ~23:45, during Experiment CR (speed only, results-neutral):** more parallel API calls (labels for a
  concept's 10 neurons run concurrently; Jev grading 32 and judging 40 concurrent; OpenAI client retries honour
  rate-limit headers). Same prompts, graders and thresholds. Concepts from Mount Everest onwards carry the new code
  fingerprint.
- **15 Oct ~23:55, known limitation found during CR (no change to the frozen method):** recall requires the *whole*
  answer as Luna phrased it. "Sahara Desert" failed recall (4/20) because the model completes "…the Sahara" and
  usually doesn't add "Desert". Concepts failing recall this way stay "did not recall" in the confirmatory results.
  A post-hoc exploratory re-run with a shortest-unambiguous-answer variant will be reported separately.
- **15 Oct ~23:55 (robustness only):** label_batch.py now caches every API result to disk and resumes;
  concept_pipeline.py skips finished concepts and retries failures. No change to judgements.
- **16 Oct ~00:05, census analysis (post hoc, reported both ways):** ~20,000 neurons (4.5%), all in layers 0–5 and
  mostly in layers 1–3 (56%, 57%, 42% of those layers), are near-silent: their strongest firing in 10M tokens is
  < 0.05 (typical peaks 5–100). They have no job to label. The census reports them as a separate "near-silent"
  category, with % clean given both excluding them (primary) and counting them as not clean.
- **16 Oct ~00:25 (logging only, results-neutral):** concept results now include every recall prompt's P(answer) and
  the model's top-5 next tokens, to diagnose recall failures (ISSUES.md #9, #10) without re-running.

---

## Hypothesis H-DEPTH: stages by depth (registered 16 Oct ~00:35, before the test was run)

**Origin (exploratory, eyeballed):** in the top 10 neurons of 7 concepts, concept "recognisers" appeared around layers
8–10 (Venice L8:3346, Eiffel L9:9406, Golden Gate L9:2918, Everest L9:10107 / L10:11862); concept "parts/associations"
around 24–31; generic answer-formatting neurons (e.g. L33:10933 "finish a proper name", L35:8393 "don't answer
with a blank") at 33–35, shared across most concepts. Counter-example: Tokyo has no dedicated early recogniser in its
top 10.

**Test:** for every recalled concept in Experiment CR (plus the post-run recall-fixed re-runs), label the top 50
attributed neurons with `label_batch.py`, then have Jev classify each as concept-specific / generic / unrelated given
the concept and the neuron's explanation and output words.

**Predictions:**
1. Concept-specific neurons are over-represented, relative to their share of all top-50 neurons, in two depth bands:
   ~layers 6–12 and ~22–32.
2. Generic neurons are over-represented in layers 32–35.
3. The same generic neurons recur across ≥ 50% of concepts.

**Fails if:** concept-specific neurons are spread evenly by depth, or generic neurons aren't concentrated late.
**Known weaknesses:** labels and the Jev classification are noisy; attribution favours some layers by construction
(the output-side gradient is largest late), so the comparison baseline must be the depth distribution of the top-50
set itself, not of the whole model.

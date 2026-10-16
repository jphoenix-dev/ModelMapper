# Prior work: what we found vs what was already known

*Written 16 Oct 2026 from four parallel literature searches (localisation, neuron description, steering, methods).
Every source was opened by the searching agent; a separate agent is re-verifying each link, title, author and year
(status at the bottom). Verdicts are our best reading, not a systematic review; we may have missed work, especially
non-English and very recent preprints. We are not specialists in this field.*

**Bottom line.** Most of what we found is **known or partly known**. Several individual facts have direct
precedents, some in the very model we used. What looks genuinely new is narrower:
1. the **scale and consistency** of the "handful of neurons" effect across 158 concept runs / 139 concepts incl.
   abstract ideas, in Qwen3-8B;
2. a **calibrated, test-retested neuron census** of a modern LLM (we found no prior % single-job figure for a modern
   model, and none for Qwen3);
3. **raw-neuron steering measured like a benchmark** (held-out prompts, strict vs lenient judge, five selection rules,
   per-concept selection) and the observation that **people are harder to steer fluently** than other concepts;
4. **concept-pair part sharing** measured against random pairs (pending a lexical control, below);
5. a candid **methods / agent case study**: which pitfalls we hit, how big they were, and which control caught each
   of the AI agent's own bugs.

We should present the work as **an independent, automated test of a recent claim, on consumer hardware**, that
replicates known effects at larger scale, adds calibration, and documents pitfalls. We should not claim to have
discovered concept neurons or neuron steering.

---

## 1. "A handful of neurons carry each recalled concept" (F1): PARTLY KNOWN

**Precedents.**
- **Knowledge neurons**: Dai et al. 2022, BERT, ~4 neurons per fact; suppressing them lowers the correct probability by ~29%
  ([arXiv 2104.08696](https://arxiv.org/abs/2104.08696)).
- **The closest precedent, in our own model**: Yona, Barzilay, Karasik & Geva 2026, "Friends and Grandmothers in
  Silico", 7 models **including Qwen3-8B** ([arXiv 2604.01404](https://arxiv.org/abs/2604.01404)). A single "entity
  cell" can erase or restore recall of an entity, but they find the effect **weaker in Qwen3-8B** and the cells in
  **early layers (0–5)**.
- Also: Yu & Ananiadou 2024 ([2312.12141](https://arxiv.org/abs/2312.12141)); neuron-level editing in Qwen3-8B, ACE
  ([2510.07896](https://arxiv.org/abs/2510.07896)).
- The essay itself already ablates small groups (8–25 neurons; Santa, Lion King) in Qwen3-32B.

**Contradicting or complicating.**
- Nanda et al. 2023, "Fact Finding" (Pythia-2.8B): athlete→sport recall was **not localisable to neurons**
  ([LessWrong](https://www.lesswrong.com/posts/iGuwZTHWb6DFY3sKB/fact-finding-attempting-to-reverse-engineer-factual-recall)).
- Liu et al. 2025: knowledge is cumulative, with no single relation neuron ([2502.17355](https://arxiv.org/abs/2502.17355)).
- Niu et al. 2024: neurons being *necessary* for an answer does not mean the knowledge is *stored* there; the
  knowledge-neuron thesis is "an oversimplification" ([2405.02421](https://arxiv.org/abs/2405.02421)).
- Chen et al. 2025, "Mission Not Accomplished?" ([2405.14117](https://arxiv.org/abs/2405.14117)).
- Choe et al. 2025: in Qwen, early **attention** matters more than MLPs for recall ([2509.08778](https://arxiv.org/abs/2509.08778)).

**What's ours.** Scale (157/158 runs ≤ 5 neurons, 139 concepts, incl. abstract ideas and generic concepts), a
non-saturating measure, random same-layer controls, and the effect holding in Qwen3-8B where Yona et al. saw weaker
single-cell effects (we use several neurons, found by attribution and verified by ablation, and late layers).

**What we must do.**
- Phrase the claim as **necessity** ("switching these off stops recall"), not storage.
- Report probability and flip rates alongside log-odds, so the result can be compared with Dai's 29%.
- Discuss the **layer tension** with Yona et al. (early entity cells vs our layers 22–32 "parts").
- Say plainly that we did not analyse attention heads.

## 2. "Concepts are built from readable, shared parts" (F2): PARTLY KNOWN

**Precedents.**
- FFN key–value memories with readable keys (Geva et al. 2021, [2012.14913](https://arxiv.org/abs/2012.14913)).
- Multimodal neurons in CLIP (Goh, Cammarata et al. 2021).
- Neurons shared across related relations (Liu et al. 2025).
- Domain-localised neuron regions (QRNCA, [2406.10868](https://arxiv.org/abs/2406.10868)).
- Knowledge neurons shared across languages (Chen et al. 2024, [2308.13198](https://arxiv.org/abs/2308.13198)).

**Contradicting.** Ait Hou & Hwa 2026, "Polysemanticity or Polysemy?" ([2604.00443](https://arxiv.org/abs/2604.00443)):
neuron overlap is driven more by **shared word form than shared meaning**.

**What's ours.** A pair-vs-random measurement with generic neurons removed: related pairs share 2.6 specific neurons
vs 0.13 for random pairs. We found no direct precedent.

**Lexical control, done 16 Oct.** None of the 26 shared non-generic neurons fires mainly on the pairs' shared word (0/26
word-form detectors; RESULTS.md). The result survives this control. Some shared neurons look conceptual (they fire in
moon contexts but on other tokens); others fire on punctuation (steam engine / printing press), where shared topic and
shared formatting can't be told apart. **Claim:** related concepts share specific non-lexical neurons ~20× more than
random pairs. Don't claim that all of them are "meaning" parts.

## 3. Depth structure: recognise → concept parts → say the answer (F3): KNOWN

Three-stage recall is well established:
- Geva et al. 2023, "Dissecting Recall", [2304.14767](https://arxiv.org/abs/2304.14767);
- ROME causal tracing, [2202.05262](https://arxiv.org/abs/2202.05262);
- "Stages of Inference", Lad, Gurnee & Tegmark, [2406.19384](https://arxiv.org/abs/2406.19384);
- late "say X" features in Lindsey et al. 2025, "Biology of an LLM";
- late prediction and suppression neurons, Gurnee et al. 2024, [2401.12181](https://arxiv.org/abs/2401.12181);
- confidence-regulation neurons, Stolfo et al. 2024, [2406.16254](https://arxiv.org/abs/2406.16254).

Our pre-registered H-DEPTH result is a **confirmation in Qwen3-8B at neuron level**, not a new finding.

## 4. Near-certain facts resist; neurons re-route (F4): PARTLY KNOWN

**Precedents.**
- Self-repair: the Hydra effect (McGrath et al. 2023, [2307.15771](https://arxiv.org/abs/2307.15771)) and Rushing &
  Nanda 2024 ([2402.15390](https://arxiv.org/abs/2402.15390)).
- "Degenerate" redundant knowledge neurons (Chen et al. 2024).

**Contradicting the word "re-routing".** Ahmad, Seth & Sankarapu 2026, "Every Ablation Is a Dose"
([2610.02173](https://arxiv.org/abs/2610.02173)): apparent self-repair (in Qwen too) follows a pre-existing gain,
i.e. it isn't re-routing *caused* by the ablation.

**What's ours.** A cross-concept link: the facts that need more neurons to flip are the ones the model is most confident about.

**What we must do.** Avoid "re-routing" as a mechanism claim. Report log-odds drop per neuron, not just flip counts,
because flipping a P = 0.998 answer needs a bigger push whatever the mechanism.

## 5. Neuron census, ~50% verified single-job (C1–C4): PARTLY KNOWN; calibration POSSIBLY NEW

**Precedents.**
- OpenAI's GPT-2 XL explanations: only ~0.3% of neurons scored ≥ 0.8, and "individual neuron scores are generally
  uninformative" (Bills et al. 2023).
- Towards Monosemanticity: the median neuron scored 0 on their rubric, and the paper documents the
  top-examples-only illusion.
- Scaling Monosemanticity (2024).
- EleutherAI's detection/fuzzing scores ([2410.13928](https://arxiv.org/abs/2410.13928)): **neurons barely above
  random** in Llama-3.1-8B and Gemma-2, i.e. *contradicting* a high single-job rate.
- FADE ([2502.16994](https://arxiv.org/abs/2502.16994)): neuron descriptions comparatively clear, i.e. *supporting* us.
- Transluce described every Llama-3.1-8B neuron, but gives no % figure.
- Sanity-check frameworks closest to our spliced controls: Oikarinen et al. 2025 ([2506.05774](https://arxiv.org/abs/2506.05774))
  and FIND ([2309.03886](https://arxiv.org/abs/2309.03886)).
- Activation-range decay (our C2) is known qualitatively from Towards Monosemanticity and Paulo et al.

**What's ours.** A judge-verified % single-job for a modern LLM, with the "clean" bar calibrated on spliced
two-neuron dossiers, plus a test-retest flip rate (~20%) and an inter-grader check. We found no prior Qwen3 neuron
census and no prior % figure for any modern model.

**Reading of our ~50%.**
- It is higher than the 2023 figures (older and smaller models, different scoring) and lower than Goodfire's ~90%.
- Our *uncalibrated* labeller (81%) sits near Goodfire's figure, so most of the gap is probably **how strict the
  judging is** (Goodfire counts "one main job with related offshoots" and uses expert audits), plus model size (32B vs 8B).
- C4 (concept neurons slightly cleaner) used an older rule and is weak: re-run it under the calibrated rule or drop it.

## 6. Steering with a few raw neurons (S1–S4): PARTLY KNOWN; evaluation angle POSSIBLY NEW

**Precedents.**
- The single sentiment neuron (Radford et al. 2017, [1704.01444](https://arxiv.org/abs/1704.01444)).
- **About 3 "expert" neurons steer generation toward a concept** (Suau, Zappella & Apostoloff 2022,
  [2110.02802](https://arxiv.org/abs/2110.02802)), the direct precedent.
- AurA toxicity neurons ([2407.12824](https://arxiv.org/abs/2407.12824)).
- Language-specific neurons ([2402.16438](https://arxiv.org/abs/2402.16438)).
- Transluce Monitor neuron clamping.
- Arora et al. 2026: circuits are sparse in the neuron basis ([2601.22594](https://arxiv.org/abs/2601.22594)).
- Golden Gate Claude (SAE features).
- **"The Eiffel Tower Llama"** (Louapre, Hugging Face, Nov 2025; SAE features on Llama-3.1-8B), **which shares our
  demo's concept name**.

**Steering is modest.**
- AxBench (Wu et al. 2025, [2501.17148](https://arxiv.org/abs/2501.17148)): SAE steering 0.165 vs prompting 0.894,
  **no raw-neuron baseline**.
- Macocco et al. 2026 ([2606.12234](https://arxiv.org/abs/2606.12234)): steering is **far less effective on
  instruction-tuned models, Qwen3-8B included**, consistent with our modest numbers.
- Anthropic's "Evaluating feature steering" (a narrow sweet spot), Tan et al. 2024 (unreliable across inputs), and
  Pres et al. 2024 (CAA works less well than first reported).

**Selection.**
- Input vs output features (Arad, Mueller & Belinkov 2025, [2505.20063](https://arxiv.org/abs/2505.20063)).
- Interpretability only weakly predicts steering usefulness (Wang et al. 2025, [2510.03659](https://arxiv.org/abs/2510.03659)).
- Gradient attribution even *anti-predicts* single-neuron controllers (Liu 2026, "Leverage Is Not Reach",
  [2606.19831](https://arxiv.org/abs/2606.19831)), consistent with our Shakespeare result.

**Prompt + reply steering.** Herbster et al. 2026 ([2604.08169](https://arxiv.org/abs/2604.08169)) found steering
all tokens beats steering only the reply's tokens (known).

**What's ours.**
- Golden-Gate-style chat steering with 2–6 attribution-chosen raw neurons on a modern instruct model, with a random
  control.
- Held-out and strict-judge measurement, including how much a score drops from the prompts it was chosen on to fresh
  prompts.
- **No universal selection rule**, but a stable per-concept preference (prior work proposes one better rule each).
- **People harder to steer fluently**: no study found comparing entities or people against other concept types.

**What we must do.**
- Don't compare our 0.16 with AxBench's numbers unless we use their scale (they score 0–2 and take a harmonic mean).
- Cite Suau 2022 as the direct precedent.
- If the demo keeps the "Eiffel Tower Qwen" name, cite the Eiffel Tower Llama.

## 7. Methods lessons (M1–M5): mostly KNOWN; the safeguards are worth reporting

**Known.**
- **Probability metrics saturate**, so use logit or log-odds measures: Zhang & Nanda 2024,
  [2309.16042](https://arxiv.org/abs/2309.16042) ("we generally recommend avoiding using probability as the metric");
  Heimersheim & Nanda 2024, [2404.15255](https://arxiv.org/abs/2404.15255).
- **Gradient attribution gives false negatives; verify with real ablation**: AtP*, [2403.00745](https://arxiv.org/abs/2403.00745);
  Syed et al. 2023; "When Attribution Patching Lies" 2026, [2606.09899](https://arxiv.org/abs/2606.09899).
- **Exact-match answer scoring underestimates models**: Kamalloo et al. 2023, [2305.06984](https://arxiv.org/abs/2305.06984);
  Jiang et al. 2020.
- **Prompt-format sensitivity**: Sclar et al. 2024, [2310.11324](https://arxiv.org/abs/2310.11324).
- **LLM judges are lenient and noisy**: Thakur et al.; Schroeder & Wood-Doughty; Zheng et al. (MT-Bench).
- **Agents introduce bugs and confirm their own hypotheses**:
  - MAIA, [2404.14394](https://arxiv.org/abs/2404.14394);
  - Anthropic's auditing agents;
  - an evaluation of Sakana's AI Scientist, [2502.14297](https://arxiv.org/abs/2502.14297);
  - MLR-Bench, [2505.19955](https://arxiv.org/abs/2505.19955), where coding agents often fabricate or invalidate results.
- **Interpretability illusions**: Bolukbasi et al. 2021, [2104.07143](https://arxiv.org/abs/2104.07143);
  Makelov et al. 2023, [2311.17030](https://arxiv.org/abs/2311.17030); Hase et al. 2023, [2301.04213](https://arxiv.org/abs/2301.04213).

**Ours, worth reporting.**
- The size of each pitfall in a real study. For example, the old "halve the probability" measure made chat format
  look 4–10× less concentrated than it is, and checking attribution's rankings with real ablation cut one concept's
  count from 50 neurons to 5.
- **Answer discovery guarded by realistic made-up concepts**, which caught a false recall.
- **Calibrating the judge on spliced two-job controls**, with test-retest noise.
- A **catalogue of the agent's own bugs and the control that caught each** (ISSUES.md, CAPS.md). Prior work mostly
  benchmarks agents rather than reporting the safeguards that rescued a real study.

---

## Follow-ups this review creates (before or alongside publication)
1. ~~Lexical control for part sharing (F2)~~: done, survives (0/26 word-form detectors); some shared neurons are
   punctuation-firing, so the claim is worded as "non-lexical".
2. **Report probability and flip rates with log-odds (F1).** The data is already in the log-odds v2 files.
3. **Word changes:** "necessary for recall", not "stores"; "others take over" with the self-repair caveat, not
   "re-routing".
4. **Read in full before publishing:** Yona et al. 2026 (same model, layer tension), Suau et al. 2022, and the AxBench
   and Macocco et al. 2026 evaluation set-ups.
5. Drop or re-run C4 under the calibrated rule.

## Citation verification status
Pending (an independent agent is fetching all 80 links and checking title, authors, year and key claims). Results
will be recorded here; any citation that fails is removed.

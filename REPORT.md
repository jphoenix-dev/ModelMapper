# Are neural networks already clean? An independent, automated test on Qwen3-8B

**James Phoenix** and **Claude** (Anthropic; running as Claude Code)
*Version 1, published 16 October 2026. Independent work, not affiliated with Anthropic, Goodfire or Alibaba (Qwen).*

> **Status:** version 1 (16 Oct 2026). Corrections will be logged in ISSUES.md and dated here.
> - Exploratory unless marked pre-registered (CR, CEN, H-DEPTH, and the confirmatory CR-3; see PREREG.md).
> - Only two small models from one family were tested (Qwen3-8B; Qwen3-0.6B comparison).
> - All 80 citations independently verified (LITERATURE.md).

## Abstract
Cammarata's essay *Neural Networks Are Already Clean* (Goodfire, 2026) argues that most individual MLP neurons in a
large language model do one recognisable job, and that concepts are composed from such neurons. We spent ~24 hours
testing its claims on Qwen3-8B with one consumer GPU (RTX 5090), with an AI agent running the experiments under
human direction. We found:

- **Recall depends on a handful of neurons.** For every concept the model recalls, a handful of specific MLP neurons
  are necessary for it. Switching off at most 5 lowers the answer's odds ≥ 20× in 157 of 158 question-format concept
  runs (139 distinct concepts, incl. abstract ideas). Five neurons remove 61–82% of the answer's probability, while
  five random same-layer neurons change nothing.
- **Concepts share parts.** The neurons have readable labels (gold, bridge, landmark, coffee), and related concepts
  share specific, non-lexical neurons ~20× more often than random pairs.
- **About half of neurons verifiably do one job.** A calibrated census of 1,000 random neurons finds ~50% verified
  single-job, closer to a floor than an estimate, against the essay's ~90% for a 4× larger model.
- **Steering works, modestly.** Nudging 2–6 of these neurons steers free-form chat towards the concept (an
  "Eiffel Tower Qwen"). The effect is real but modest and best chosen per concept. People steer only when the
  question invites a person.
- **Confirmed.** A frozen confirmatory run on 40 never-used concepts (CR-3) passed 6 of 7 predictions: recall,
  concentration, flips, the confidence link and steering selection. It failed one, and the failure revised our
  view of people.

Most individual findings have precedents (knowledge neurons, entity cells, expert-neuron steering, staged recall).
Our contribution is an automated test at scale with explicit controls and calibration, plus a candid account of the
measurement pitfalls we hit, including the agent's own bugs and the controls that caught them.

## 1. Background and question
The essay reports preliminary audits of Qwen3-32B:
- **~90% of neuron poles have one main job** (often with related offshoots).
- **Small neuron groups carry specific behaviours.** Removing ~8–25 neurons changes answers in its examples.
- **Concepts are built compositionally** from such neurons.

Earlier work was more pessimistic about single neurons: polysemanticity and superposition (Bills et al. 2023;
Bricken et al. 2023). We asked whether the essay's claims reproduce, with fresh code, in a smaller model from the same
family, and whether they survive deliberate attempts to break them.

## 2. Setup
- **Model.** Qwen3-8B, 36 layers × 12,288 SwiGLU MLP neurons (442,368), bf16. A "neuron" is the input to down_proj:
  SiLU(gate) × value. Positive and negative "poles" are treated separately, as in the essay.
- **Hardware.** One RTX 5090 (32 GB). Total wall-clock about one day.
- **Tools.**
  - Neuron labels are written by an OpenAI model ("Luna", gpt-6-luna) from top-activating examples.
  - Labels and steered replies are graded by TypeSafe's "Jev" (System One), which returns calibrated yes/no and
    rubric probabilities.
  - A free local fluency check (validated against Jev) is used where noted.
- **Atlas.** The top-40 activations per pole for every neuron over 10M FineWeb tokens.
- **Process.** Human-directed, agent-executed. James set questions and standards (no silent caps; assume the set-up is
  broken; pre-register; test for noise). Claude (Claude Code) wrote and ran all code, controls and analyses. Every
  decision, bug and correction is logged in ISSUES.md, CAPS.md and RESULTS.md.

## 3. Methods (summary; ARCHITECTURE.md has the pipeline)
1. **Recall prompts.**
   - Luna writes 20 questions per concept whose answer is the concept. Recall means P(answer) ≥ threshold on ≥ 5
     prompts.
   - Prompts that give away the answer are filtered.
   - **Answer discovery:** the model answers first, Luna checks that the reply names the concept, and the model's own
     wording is scored. Realistic made-up concepts act as negative controls.
2. **Finding neurons.**
   - Rank all neurons by activation × gradient on the answer.
   - Re-rank the top 200 by measured single-neuron knockout (attribution alone mis-ranks).
3. **Concentration.**
   - Switch off the top-k (k = 1…50) and measure the mean drop in the answer's **log-odds**, then
     k_mean = fewest neurons for a drop ≥ 3.
   - Also the median prompt, and k_flip = fewest neurons to push P below 0.5 on half the prompts the model had right.
   - Control: random same-layer neurons.
4. **Census.**
   - Sample 1,000 random neurons; Luna labels each pole from 15 examples.
   - Jev grades the label on 10 own examples and on 20 random snippets with a 3-level rubric, keeping the soft score.
   - The "single-job" bar is **calibrated on spliced dossiers**: examples from two unrelated neurons mixed together,
     chosen so ≤ 10% of them pass.
5. **Steering.**
   - Clamp or add to 2–6 neurons during reply generation, using one of five selection rules (late-layer attribution;
     any layer; measured knockout; direct effect on the answer token; label-matched).
   - Select the setting on 20 neutral prompts; score it on 20 **held-out** prompts.
   - Score = "on-topic and fluent" on the same reply (Jev lenient; strict "main topic" also reported).
   - Controls: random neurons; unsteered replies.

## 4. Results
### 4.1 Recall depends on a handful of neurons (exploratory; replicated across prompt sets and 80 new concepts)
- **Question format.** k_mean ≤ 5 in **157/158** runs and ≤ 2 in 142/158, all within 10. The median prompt gives the
  same result.
- **Completion format** ("The tallest mountain is Mount"): 8/11, all within 20.
- **Flips.** Five or fewer neurons flip the answer in 137/158 runs. The holdouts are near-certain facts (the Moon,
  Tokyo, DNA, Bitcoin), which start with log-odds of 9–12 and need 20–50 neurons.
- **In plain probability:**
  - The top neuron cuts P(answer) by 28–40%, comparable to Dai et al.'s ~29% with ~4 neurons.
  - The top five cut it by 61–82% and flip 67–84% of correct answers.
  - Five random neurons: no change to three decimals.
- **A measurement artefact we had to fix.** Our first measure ("fewest neurons to halve P") made question format look
  4–10× less concentrated. That was saturation at P ≈ 0.998, a known pitfall (Zhang & Nanda 2024; Heimersheim &
  Nanda 2024).
- **Repeated knockout.** Knocking out, re-finding and knocking out again shows near-certain facts resist because of a
  larger confidence margin and more spread-out support. Other neurons compensate for every concept. This is
  consistent with known self-repair, and we don't claim a mechanism.
- **Recall coverage.**
  - **Every real concept we first called "not recalled" was a fault in our measurement, with one exception.** The
    faults were prompt format, our expected answer wording, or our leak filter. The exception, "the robot", is a
    prompt-design limit.
  - **Recalled:** 139 distinct concepts.
  - **Made-up controls:** not recalled, after one discovery bug was caught.

### 4.2 Readable, shared parts
- **Labels.** Top neurons have readable labels and multilingual parts (山 for mountain, 咖啡 for coffee, 民主 for
  democracy).
- **Sharing.** For 10 related pairs, the top-50 neurons share 2.6 specific (non-generic) neurons on average, against
  0.13 for 200 random pairs.
  - Examples: coffee/tea share the coffee neuron; the Moon / the Moon landing share the Moon neuron.
  - **Lexical control:** none of the 26 shared neurons fires mainly on a word the pair shares.
  - Some shared neurons fire on punctuation, so we claim "non-lexical", not "all meaning".
- **Depth (pre-registered H-DEPTH; replicated, 91% agreement).** Layers 6–12 recognise the question, 22–32 hold
  concept parts, 33–35 hold generic "say the answer" machinery. This is consistent with known three-stage recall.

### 4.3 How clean are neurons? (pre-registered CEN, plus calibration)
- **Verified single-job: ~50% of positive poles, ~47% of negative**, at a bar that passes 8% / 4% of known two-job
  dossiers. The labeller alone says ~81%, but it also calls about half of the two-job dossiers "one job".
- **Robustness:**
  - Fit fades gradually below the top activations: 66 → 51 → 41 → 33% at 100/50/25/10% of peak (random 20%).
  - Single-neuron verdicts flip ~20% on identical re-runs, while population rates stay stable.
  - A second grader agrees (Spearman 0.81), and a blind human spot-check matches Jev's accuracy (88% vs 88%, 40 cards).
  - Longer grading context (3,500 characters) changes nothing.
  - Less labeller reasoning gives vaguer labels.
- **Reading.** Our figure sits between the pessimistic 2023 numbers and the essay's ~90%. The uncalibrated 81% lands
  near the essay, so the gap is plausibly mostly how strictly "one job" is judged, plus model size (8B vs 32B).

### 4.4 Steering ("Eiffel Tower Qwen")
- **Raw-neuron steering works.** Nudging 2–6 neurons steers chat towards the concept. Random neurons and unsteered
  replies score ≈ 0 throughout.
- **Size of the effect.**
  - **Modest.** With per-concept selection, held-out "on-topic and fluent" averages **0.16** (Jev, 59 concepts)
    against **0.09** for a single fixed rule. Per-concept selection is better in 31 concepts and worse in 7.
  - **Stricter judge.** "Main topic" roughly halves scores. Typical steering makes the concept *intrude* (coffee at the
    top of a weekend plan) rather than take over.
  - **Winner's curse.** Scores halve from the prompts a setting was chosen on to fresh prompts.
- **Selection.**
  - No rule wins overall.
  - A concept's preferred rule is stable across independently written prompt sets (same rule 13/19; held-out r = 0.76).
  - Attribution rank is not steering usefulness: Shakespeare steered only through his one name-labelled neuron
    (0.50 fluent vs ≤ 0.10).
- **People.** On generic prompts, people almost never appear. Given questions that invite a person, they steer:
  - Exploratory: person-friendly prompts roughly triple people-steering, and 4 of 5 hard cases reach 0.25–0.55.
  - Confirmatory (CR-3): the four people (Cleopatra, Lincoln, Gandhi, Martin Luther King Jr.) reached 0.43 against
    0.22 for other concepts on generic prompts. Jev agrees (0.40), and none of the names appears unsteered.
  - Some individuals still resist (Shakespeare, Mozart, Harry Potter), and pushing harder tends to garble the text.
- **Prompt steering.** Steering the prompt as well as the reply is stronger (+0.05–0.10), as known from prior work.

### 4.5 Model size: Qwen3-0.6B vs 8B (exploratory; judge-free)
Same 60 concepts (CR-2 + wave 2) and 3 made-up controls, identical saved prompts, the same judge-free pipeline, with
answer discovery re-run per model (`results/scale_compare.txt`). The matched 8B re-run was stopped after 44 concepts
to free the GPU; it recalled all 44.

| | Qwen3-0.6B | Qwen3-8B |
|---|---|---|
| real concepts recalled (question format) | **7/60** | 44/44 covered (and 60/60 in earlier runs) |
| made-up controls recalled | 0/3 | 0/3 |
| recalled concepts with k_mean ≤ 5 | 6/7 | all |
| typical k_flip on the shared concepts | **1–3** | 1–50 (Tokyo, photosynthesis, Bitcoin: 50) |
| steering (judge-free, held-out), mean | 0.057 (n = 7) | 0.226 (n = 44) |

- **The small model knows far fewer of these concepts** in chat-question form.
- **Where it does recall, recall again depends on a handful of neurons.**
- **Its answers flip much more easily** (1–3 neurons), consistent with the confidence-margin reading of §4.1: the less
  sure the model, the fewer neurons it takes to tip it.
- **Steering is weak** in the small model.
- **Only two sizes from one family**, so we make no claim about trends with scale.

### 4.6 Confirmatory replication (CR-3, pre-registered and frozen before data)
40 never-used concepts, picked mechanically (10 categories) plus 4 new made-up controls; the exact final pipeline;
predictions and scoring script committed before any run (commits a0c3702, edf7668).

| prediction | result |
|---|---|
| P1 ≥ 90% real recalled, 0/4 made-up | **PASS**: 40/40, 0/4 |
| P2 ≤ 5 neurons for ≥ 90% (mean), ≥ 80% (median); random null | **PASS**: 98%, 98%, 100% |
| P3a answer flips with ≤ 5 neurons for ≥ 70% | **PASS**: 85% |
| P3b resistant facts are the most confident | **PASS**: p = 0.005 |
| P4a per-concept steering ≥ 1.5× fixed rule; random < 0.05 | **PASS**: 2.44×; 0.006 |
| P4b same direction under Jev | **PASS**: 0.095 vs 0.045 |
| P5 people steerable but below non-people | **FAIL**: 0.43 vs 0.22 |

P5's failure is genuine: there's no unsteered baseline issue, no keyword looseness, and Jev confirms it. The
prediction was also mis-specified, because it compared different prompt sets. We report it as failed and revise the
people claim (§4.4).

## 5. Pitfalls we hit (and what caught them)
Each is logged with its fix in ISSUES.md (#1–33) and CAPS.md:
- **Measurement.**
  - Saturating probability metric (#27).
  - Attribution mis-ranking (re-ranking a concept's neurons by real knockout cut one count from 50 to 5).
  - The answer-wording and leak-filter failures that faked "not recalled" (#9, #10, #26, #29, #32).
  - Answer discovery faking recall for a made-up bridge, caught by the made-up controls (#31).
- **Judging.**
  - Uncalibrated judge thresholds and discarded judge confidence (#23).
  - Lenient judge wording (#14).
  - Display caps that misled the human reading the judge's output (#19, CAPS.md).
- **Steering.**
  - Accidental prompt steering in a comparison script, caught because a re-run of the original rule didn't
    reproduce (#28).
  - Selection on the evaluation prompts (winner's curse).
- **Agent and engineering bugs.** A variable-name collision, line endings in concept names, a stray duplicate GPU job
  (#30), an output-format drift (#33), and several over-statements in the agent's own summaries. All were caught by
  re-running, by assertions, or by an audit against the result files.

Our view: the controls (random neurons, made-up concepts, spliced dossiers, re-runs with independently written
prompts, held-out prompts) mattered more than any single method choice.

## 6. Related work (details and links: LITERATURE.md)
- **Localisation.**
  - Knowledge neurons (Dai et al. 2022) and entity cells, including in Qwen3-8B (Yona et al. 2026).
  - Critiques: "Fact Finding" (Nanda et al. 2023) finds factual recall not localisable to neurons in Pythia; Niu et al.
    2024 argue necessity is not storage.
  - Staged recall (Geva et al. 2023; ROME).
  - Self-repair (Hydra effect; Rushing & Nanda 2024).
- **Neuron description.** OpenAI 2023; Towards Monosemanticity; EleutherAI (Paulo et al.), whose neuron scores are
  barely above chance; Transluce 2024; Oikarinen et al. 2025 (sanity checks).
- **Steering.**
  - Raw neurons: the sentiment neuron (Radford 2017); about 3 "expert" neurons steering generation (Suau et al. 2022).
  - Feature-based: Golden Gate Claude; the Eiffel Tower Llama (SAE features).
  - Evaluations: AxBench (Wu et al. 2025; no raw-neuron baseline); weak steering on instruction-tuned models incl.
    Qwen3-8B (Macocco et al. 2026).
- **Methods.** Activation-patching best practice (Zhang & Nanda; Heimersheim & Nanda); AtP*; answer-equivalence in QA
  evaluation (Kamalloo et al.); agent-research reliability (MAIA; MLR-Bench).

## 7. Limitations
- **Two small models, one family.** Qwen3-8B, plus a 0.6B comparison. The essay's claims concern a 32B model. No
  other families were tested.
- **Necessity, not storage.** Knockouts show neurons are needed for recall, not that knowledge lives in them.
- **No attention-head analysis**, although prior work finds attention matters for recall in Qwen.
- **LLM labellers and judges.** Luna writes both prompts and labels. Grading is by LLM judges (calibrated, but
  single-sample per item). A blind human spot-check (one rater, 40 cards) found Jev as accurate as the human at telling a neuron's real firings from random text (88% vs 88%).
- **Free fluency check.** It inflates absolute levels (it agrees with Jev on ranking, r = 0.65); reported numbers use
  Jev where stated.
- **Noise.** Single-concept steering numbers move ±0.1 between prompt sets. Population numbers are stable.
- **Exploratory vs confirmed.** Most results were found exploratorily. CR-3 then confirmed the central ones on 40
  new concepts (6/7 predictions; one failed, reported in full). The census, part sharing and steering-rule details are
  not covered by CR-3.
- **Not specialists.** We may have missed relevant work.

## 8. Reproducibility
All code, prompts, pre-registrations, logs and raw judge outputs are in the repository. Two kinds of data are
regenerated rather than shipped:
- **Bulk FineWeb text:** the per-neuron example scans and the blind dossiers, rebuilt from the public corpus (README,
  "Data").
- **Large derived data:** the neuron atlas, rebuilt with `src/build_atlas.py`.

Answer keys (`answer_keys/`) let any install check that it rediscovers our marker neurons. Generation is greedy, so steering re-runs reproduce (mean
difference 0.013).

## Acknowledgements and credit
The experiments were designed, run and analysed in about 24 hours by James and Claude working together: James set
the questions and standards and made the calls; Claude (Anthropic) wrote and ran the code, controls and analyses.
Thanks to Nick Cammarata and Goodfire for a clear, testable essay.

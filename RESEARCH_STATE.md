# Where the research stands (research branch; living document, last updated 17 Oct 2026 evening)

One place for the current picture. Everything here is exploratory unless marked as pre-registered. The evidence and
caveats are in the dated RESULTS.md entries named in brackets. Nothing on this branch is published.

## The goal (James, 17 Oct)
A mechanism that tells us something informative and queryable later about how any model maps neurons (and attention
heads) to concepts. It should be scalable, self-verifying and reproducible, and the loop should be fast. **Phase 1
(now):** prove the loop gives meaningful, novel findings, using pre-registered predictions. **Phase 2:** optimise
and scale.

## Working picture: what a concept neuron is and does
How Qwen3-8B computes, in the terms we use:
- Every token carries a running state, the **residual stream** (4,096 numbers). Each of the 36 layers runs
  **attention** first, then the **MLP**, on every token, and **adds** its result to the stream. Nothing is replaced,
  so early information stays available unless something cancels it.
- **Attention** is the only part that moves information between tokens. It reads earlier tokens' streams.
- Each **MLP neuron** (12,288 per layer) reads only its own token's stream, through two fixed directions. One is the
  **gate**, a soft switch; the other is the **value**. If both line up with the stream, it fires, and it adds **one
  fixed output vector** times its activation. Different triggers change only how much it pushes, or the sign; the
  direction never changes.
- **Many triggers.** There are far more features than dimensions, so feature directions overlap. A neuron's fixed
  read directions partly line up with several unrelated features, and any of them can set it off. This is
  polysemanticity on the input side. Some of those features may have no human name.
- **One job.** The output is one push, and the network mostly reacts to it the same way whatever set it off.
- **Telling triggers apart is possible but not shown.** The neuron keeps no memory of its trigger, but the triggering
  features stay in the stream, so later parts *could* react to "push + accounting context" differently from
  "push + Moon context". Whether Qwen does this, or simply tolerates the leak, is open (the interference test below).
- **Better framing (James):** a neuron has a **job**, not a concept. A concept is what a few jobs add up to.
  - Jobs range from narrow (dedicated, close to the concept itself) to broad ingredients ("a person's name comes
    next").
  - They also vary by depth: most key neurons sit in layers 28-35 and look like "deliver the answer" jobs; a smaller
    cluster at layers 8-10 looks more like recognising.
  - "Mono vs poly" is usually defined on the input side. On the output side every neuron is "mono" by construction;
    the open question is whether its one push means something a human can name.

## Established findings (with evidence)
| Finding | Evidence | RESULTS entry |
|---|---|---|
| A handful of neurons carries a recalled concept, famous or obscure | k_mean <= 5 in 71/72 of a random fame-stratified sample; median 1-3 per tier; obscure concepts are not known rather than spread out | 17 Oct night, fame sample |
| These neurons are about the concept, not the prompt wording | **P1 (pre-registered) holds:** top-5 break recall on independent Wikipedia cloze prompts (+5.59 log-odds vs -0.003 random) | 17 Oct day, battery |
| "Dedicated" depends on the comparison set and keeps falling | ~ -9 to -10 points per doubling; **P2 (pre-registered) holds narrowly** out of sample (32.0% vs 29.4% predicted) | 17 Oct morning, day |
| Sharing between concepts is causal | **P3 (pre-registered) holds:** a shared neuron hurts both concepts (+0.835 vs ~0 controls) | 17 Oct day |
| The map's structure predicts unseen concepts | **P4 (pre-registered) holds:** family neurons appear at 9.2x the other-family rate in 191 new concepts (p = 0.0001); shuffles never reach it; holds without people (8.3x). Caveat: family ~ type of answer word | 17 Oct day / P4 |
| Neuron descriptions in web text are stable | **P5 (pre-registered) holds:** rho 0.986 firing rate, 0.883 on-concept share on 2M new tokens | 17 Oct day |
| Concept neurons are polysemantic in ordinary text | Luna on web firings: 12% dedicated, 22% ingredient, 57% mostly unrelated (controls 95% unrelated) | 17 Oct night, Luna judge |
| Their extra meanings are mostly genuinely different, not hidden facets | In Qwen's own space, meanings Luna calls unrelated sit at the unrelated reference; Ur-hubs in only 8/98 (5% by chance): weak | 17 Oct day, stages A/C |
| Packed triggers, one job | A neuron's downstream effect barely differs across its meanings (0.18 on the same-job -> different-neurons scale); **holds on above-noise knockouts** (0.18-0.22 at every threshold) | 17 Oct day stage D; late afternoon noise check |
| Key neurons often share a layer, slightly less than chance | 46% of concepts have >= 2 of top-5 on one layer (chance 54%); ~70% of key neurons in L28-35 | 17 Oct late afternoon |
| A concept's key neurons combine a 'say it' job with legible attribute jobs | 60% have one late neuron with the answer in its top-10 of 152k tokens (null 0.4%); >= 2 neurons raise the answer in 71% (null 2%); e.g. Jimmy Carter = first names + 1970s + presidency + Georgia | 17 Oct evening |
| Attribute neurons are reusable, nameable parts | A neuron's job from the weights picks its users out of distractors: 74.6% vs 55.4% control vs 44% chance (200 neurons, p = 2e-14); pre-set ratio criterion not met (badly set) | 17 Oct evening |
| Attention routes into concept neurons causally; answer staged late | Head -> neuron routes causal 49/53; answer staged at the turn boundary, copied by late heads; no bow-tie | 16 Oct attention study |

## Method lessons (keep)
- bf16 results depend on batch shape (up to ~0.5 log-odds near saturation). Batched knockouts failed their gate
  twice and were only 1.1x faster: **dropped**. Compare only like-shaped passes, or run a null in the same pass.
- A typical single-token, single-neuron knockout is only ~2x the batch-shape noise (gpu-fan-out session). Check that
  conclusions hold on above-noise cases.
- Pre-set rules can fail their own calibration (stage D silhouette): report it, don't rescue it quietly.
- Save everything (per-prompt drops, all 200 candidates, activation maps, head data) so new questions don't need
  re-runs.

## Open questions and next tests
1. **Concept = intersection of jobs? Partly (17 Oct evening).** From the weights (direct pushes): in 60% of concepts
   one late neuron is close to 'say this word'. It usually sits alongside 1-3 legible attribute neurons ('Georgia',
   '1970s', 'Bay Area', 'first name') that also raise the answer (>= 2 raise it in 71% vs 2% by chance). Where no
   single neuron says it, the attributes' overlap alone gets there in about 1 in 6. **Attribute neurons are reusable
   parts:** a neuron's weight-read job picks out which concepts use it (74.6% vs 55.4% for another neuron's job, chance
   44%; 103 vs 21 neurons, p = 2e-14). The concepts sharing a neuron are themselves similar. Next: follow the indirect
   path (later layers).
2. **Interference.** In a neuron's off-topic contexts, does switching it off make Qwen's predictions better (the
   network tolerates a leak), worse (it uses the push there too) or neither?
3. **Family vs answer type** (the P4 caveat): e.g. people named in the answer vs described.
4. **Meaning count vs sample size** (15/30/60/120 firings): do the meanings saturate?
5. **Quantisation gate** (int8/int4 on the 12-concept reproduction set): would it speed up the loop without
   changing the map?
6. Later: a size curve across Qwen3 0.6B -> 8B (do Ur-hubs and dedication change with scale?); cross-family check
   (3080); atlas database and provenance (after Phase 1).

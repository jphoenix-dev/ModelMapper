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
| A concept's key neurons form a feed-forward assembly converging on the 'say it' neuron | 'say' neurons act directly (share 1.05), early ones only indirectly; zeroing an upstream key neuron cuts a later 'say' neuron by ~12% vs ~0 for activation-matched nulls (555 concepts, all pair types p < 1e-5) | 17 Oct evening, indirect path |
| Between words, the assembly runs through a few shared heads | Earlier-word part ~19% of the composition (p = 3e-40 vs null); top-3 heads recover 56% vs 0% random; L34H28 is a top carrier in 173 of 312 concepts; matches the attention study's carriers | 17 Oct evening, cross-position |
| Off-topic firings are harmless at single-firing resolution | Switching a concept neuron off on unrelated text: +0.0003 +- 0.0005 nats/token (neither); on its own topic +0.004 +- 0.002 (helps); 22,415 firings vs an in-batch null | 17 Oct evening, interference |
| Key neurons often share a layer, slightly less than chance | 46% of concepts have >= 2 of top-5 on one layer (chance 54%); ~70% of key neurons in L28-35 | 17 Oct late afternoon |
| A concept's key neurons combine a 'say it' job with legible attribute jobs | 60% have one late neuron with the answer in its top-10 of 152k tokens (null 0.4%); >= 2 neurons raise the answer in 71% (null 2%); e.g. Jimmy Carter = first names + 1970s + presidency + Georgia | 17 Oct evening |
| Attribute neurons are reusable, nameable parts | A neuron's job from the weights picks its users out of distractors: 74.6% vs 55.4% control vs 44% chance (200 neurons, p = 2e-14); pre-set ratio criterion not met (badly set) | 17 Oct evening |
| Attention routes into concept neurons causally; answer staged late | Head -> neuron routes causal 49/53; answer staged at the turn boundary, copied by late heads; no bow-tie | 16 Oct attention study |

## Method lessons (keep)
- bf16 results depend on batch shape (up to ~0.5 log-odds near saturation). Batched knockouts failed their gate
  twice and were only 1.1x faster: **dropped**. Compare only like-shaped passes, or run a null in the same pass.
- A typical single-token, single-neuron knockout is only ~2x the batch-shape noise (gpu-fan-out session). Check that
  conclusions hold on above-noise cases.
- Any knockout that changes anything diverges by ~0.005 nats/token (~3e-4 KL) in bf16, whatever its size: an in-batch
  null is not exact. Read single-neuron effects against a null, never against 0.
- Check gate counters actually count (an edit once put them inside a comment); verify gates from saved records.
- Pre-set rules can fail their own calibration (stage D silhouette): report it, don't rescue it quietly.
- Save everything (per-prompt drops, all 200 candidates, activation maps, head data) so new questions don't need
  re-runs.

## Direction (17 Oct late evening, with James)
Back to the loop. What we capture today is a validated, concept-first map of ONE behaviour (naming an entity), touching
2.8% of neurons. Agreed direction: (a) keep labelling concept neurons, systematically and reproducibly; (b) capture
'all other neurons' bottom-up as stored measurements, NOT imposed categories (the truth may be weirder than our list);
labels only from tests a neuron passes. Before scaling (agreed 'regret' list): one schema with provenance; a null in
every measurement; checks verified from records; save raw ingredients (e.g. residuals at key positions) so the unit can
change later (SAE / transcoder features); a deliberate sampling frame, not only named entities in Q&A; a join key with
the distributed bottom-up pipeline; self-checks in every batch; nothing Qwen-specific. All output to D:/already-clean-data.
**Search instead of grind - feasibility:** cheap profiles rank concept neurons well enough to prioritise (new
coverage-4 concept neurons at the 90th percentile of their layer; top 1% holds 16.7x the chance count; ~1/3 of the
signal is neuron size). Next: the precision of search-then-verify on top-ranked unmapped neurons.

## Open questions and next tests
1. **Concept = intersection of jobs? Partly (17 Oct evening).** From the weights (direct pushes): in 60% of concepts
   one late neuron is close to 'say this word'. It usually sits alongside 1-3 legible attribute neurons ('Georgia',
   '1970s', 'Bay Area', 'first name') that also raise the answer (>= 2 raise it in 71% vs 2% by chance). Where no
   single neuron says it, the attributes' overlap alone gets there in about 1 in 6. **Attribute neurons are reusable
   parts:** a neuron's weight-read job picks out which concepts use it (74.6% vs 55.4% for another neuron's job, chance
   44%; 103 vs 21 neurons, p = 2e-14). The concepts sharing a neuron are themselves similar. **Indirect path (17 Oct
   evening):** 'say' neurons write the answer directly (direct share 1.05); attribute neurons are mostly direct (0.80);
   early and 'late-other' neurons work through later parts. The key neurons form a feed-forward assembly: zeroing an
   attribute neuron cuts a later 'say' neuron's activation by 12.7% vs ~0 for an activation-matched null (p = 9e-34).
   **Between words (17 Oct evening):** about a fifth of that composition runs from earlier words, carried by a small
   shared set of heads (L34H28 in 173 concepts, then L31H3, L34H21, L26H20, L35H22...; top-3 patching recovers 56% vs 0%
   for random heads). These are the meaning carriers and turn-boundary copiers that the attention study had found.
2. **Interference: none detected (17 Oct evening).** On unrelated text, switching a concept neuron off neither helps
   nor hurts on average (+0.0003 +- 0.0005 nats/token vs the null); on its own topic its push helps (+0.004 +- 0.002).
   Single-firing effects are only ~1.5x bf16 rounding noise, so a weak leak below ~0.0005 nats/token would be missed.
3. **Family vs answer type** (the P4 caveat): e.g. people named in the answer vs described.
4. **Meaning count vs sample size** (15/30/60/120 firings): do the meanings saturate?
5. **Quantisation gate** (int8/int4 on the 12-concept reproduction set): would it speed up the loop without
   changing the map?
6. Later: a size curve across Qwen3 0.6B -> 8B (do Ur-hubs and dedication change with scale?); cross-family check
   (3080); atlas database and provenance (after Phase 1).

# Pre-registration: does the mapping loop tell us something meaningful about Qwen3-8B? (17 Oct 2026)

Written before the data for each test exists (James: "first prove the loop yields meaningful, novel findings, then
scale"). Each prediction is stated with its pass/fail rule. Results will be added below each one without editing the
prediction text. Research branch; exploratory until James approves.

Common definitions: a concept's **top-k neurons** = its measured single-knockout ranking (rank_save / map_loop);
**recall drop** = mean drop in the answer's exact log-odds; **random control** = the same number of random neurons from
the same layers; **flip** = a correctly answered prompt falls below p = 0.5.

## P1 - Generalisation to independent prompts
**Claim tested:** the neurons the loop finds are about the concept, not about Luna's question wording.
**Design:** for each recalled concept (sample of up to 100, fixed seed), a second prompt set built WITHOUT Luna: cloze
prompts from the first sentence of the concept's English Wikipedia article, with the article title removed (the model
must produce the title). Only concepts whose cloze prompts the model answers (>= 3 prompts with p(answer) >= 0.2) are
tested. Knock out the concept's top-5 neurons (found on Luna's prompts) vs 5 random same-layer neurons.
**Prediction:** median recall drop on the independent prompts >= 1.0 log-odds AND >= 5x the random control's, and
the top-5 flip at least half the correct cloze prompts in >= 40% of tested concepts.
**Fails if:** median drop < 1.0, or not above 5x random.

## P2 - Dedication curve out of sample (stated in RESULTS 17 Oct morning, repeated here)
**Prediction:** for the 178 hand-picked concepts' neurons, the share firing for none of N other concepts follows
dedicated(N) = 36.5% - 9.3 x log2(N/358), within +-3 points, at the N reached after coverage-3.

## P3 - Shared neurons are causally shared
**Claim tested:** when the map says two concepts share a neuron, that neuron matters to both.
**Design:** pairs of recalled concepts (A, B) whose top-10 lists share exactly one neuron n (not 'always on'); knock
out n alone and measure the recall drop on A's and on B's prompts. Control 1: a neuron in A's top-10 that is NOT in B's
list, measured on B. Control 2: a random same-layer neuron, measured on B.
**Prediction:** median drop on B from the shared neuron >= 3x control 1 and >= 3x control 2, and >= 0.3 log-odds.
**Fails if:** the shared neuron's effect on B is not above both controls by 3x.

## P4 - Structure predicts the neurons of new concepts (coverage-4 as held-out set)
**Claim tested:** the map has inferable structure: a new concept's neurons can be partly predicted from what similar,
already-mapped concepts use.
**Design:** family labels for every concept from Wikidata 'instance of' (P31), no Luna. For each family with >= 5
mapped members before coverage-4, the 'family neurons' = neurons in the top-30 of >= 20% of members (excluding always-on
neurons). Prediction made from the pre-coverage-4 maps only; scored on coverage-4 concepts of those families.
**Prediction:** a new concept's top-30 contains its family's neurons at >= 3x the rate of another family's neurons
(matched for the number of family neurons), median over coverage-4 concepts in families with >= 5 members.
**Fails if:** < 3x, or not significantly above 1x (sign test p < 0.05 over concepts).

## P5 - Corpus descriptions predict new text
**Claim tested:** where a neuron fires in ordinary text is a stable property the atlas can describe.
**Design:** for the concept neurons profiled on the atlas's 10M tokens, run the same profile on 2M NEW FineWeb tokens
(documents not in the atlas). Rank correlation between the two profiles of (a) firing rate, (b) on-concept share.
**Prediction:** Spearman >= 0.8 for firing rate and >= 0.6 for on-concept share.
**Fails if:** below either.

## Novelty ranking (after the tests)
Every finding that survives is checked against LITERATURE.md and new searches, and labelled known / partly known /
apparently new, as in the 16 Oct literature check.

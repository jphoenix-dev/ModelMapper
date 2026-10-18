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
**RESULT (17 Oct 13:42, `p1_p3_run.py`): HOLDS.** 100 concepts sampled; 50 had >= 3 cloze sentences, 32 had >= 3 the
model answers (tested). Median drop with the top-5 off +5.59 log-odds vs random-5 -0.003; top-5 flips >= half the
correct cloze prompts in 66% of tested concepts. Caveat: some cloze sentences keep partial name clues (e.g. "Maldive
Islands" after blanking "Maldives"), which raises baselines but not the knockout comparison.

## P2 - Dedication curve out of sample (stated in RESULTS 17 Oct morning, repeated here)
**Prediction:** for the 178 hand-picked concepts' neurons, the share firing for none of N other concepts follows
dedicated(N) = 36.5% - 9.3 x log2(N/358), within +-3 points, at the N reached after coverage-3.
**RESULT (17 Oct 13:03, `p2_check.py`): HOLDS, narrowly.** N = 609 other concepts (610 concepts); observed 32.0%
(368/1150) vs predicted 29.4%: +2.6 points (limit +-3). The deviation is towards slight flattening (the last doubling
lost ~9.0 points vs 9.3 predicted); the added concepts are mostly people and countries, so the population mix changed too.

## P3 - Shared neurons are causally shared
**Claim tested:** when the map says two concepts share a neuron, that neuron matters to both.
**Design:** pairs of recalled concepts (A, B) whose top-10 lists share exactly one neuron n (not 'always on'); knock
out n alone and measure the recall drop on A's and on B's prompts. Control 1: a neuron in A's top-10 that is NOT in B's
list, measured on B. Control 2: a random same-layer neuron, measured on B.
**Prediction:** median drop on B from the shared neuron >= 3x control 1 and >= 3x control 2, and >= 0.3 log-odds.
**Fails if:** the shared neuron's effect on B is not above both controls by 3x.
**RESULT (17 Oct 13:42): HOLDS.** 150 pairs from 251 concepts with saved prompts. Median recall drop on B from the shared
neuron +0.835 (on A +0.894); control 1 (A's non-shared top-10 neuron) +0.002; control 2 (random, same layer) +0.000.

## P4 - Structure predicts the neurons of new concepts (coverage-4 as held-out set)
**Claim tested:** the map has inferable structure: a new concept's neurons can be partly predicted from what similar,
already-mapped concepts use.
**Design:** family labels for every concept from Wikidata 'instance of' (P31), no Luna. For each family with >= 5
mapped members before coverage-4, the 'family neurons' = neurons in the top-30 of >= 20% of members (excluding always-on
neurons). Prediction made from the pre-coverage-4 maps only; scored on coverage-4 concepts of those families.
**Prediction:** a new concept's top-30 contains its family's neurons at >= 3x the rate of another family's neurons
(matched for the number of family neurons), median over coverage-4 concepts in families with >= 5 members.
**Fails if:** < 3x, or not significantly above 1x (sign test p < 0.05 over concepts).
**RESULT (17 Oct 16:08, `p4_family.py score`): HOLDS.** 191 coverage-4 concepts fall in predicted families (human 154,
sovereign state 23, film 6, others 1-2 each). Median own/other family hit-rate ratio 9.18 (needs >= 3); own > other in
118/185 (6 ties), one-sided sign test p = 0.00011. Median own-family hit rate 0.250 vs other-family 0.013. 'Matched for
the number of family neurons' is done by comparing hit RATES (share of the family's neurons present), not counts.
**Post-hoc checks (not pre-registered; `p4_checks.py`):** (1) shuffled family labels, 2,000 times: median ratio 0.41,
99th percentile 1.71, max 3.79 - no shuffle reaches 9.18; (2) without the human family (37 concepts): ratio 8.27, 29/37,
p = 0.0004 (sovereign state 19/23, film 6/6); (3) the 4 human-family neurons (L25:11777, L26:4584, L30:8643, L35:9346)
are in 21% of new people's top-30 vs 2.6% of new non-people.
**Caveats:** families here largely coincide with the TYPE of answer word (a person's name, a country's name), so family
neurons may partly be 'a name of this kind comes next' rather than knowledge about the family - this test cannot tell
those apart. 72/191 new concepts contain none of their family's neurons. The family-share rule is loose for small
families (20% of 5 members = 1 member), giving big neuron sets for some families (country 136, war 125).

## P5 - Corpus descriptions predict new text
**Claim tested:** where a neuron fires in ordinary text is a stable property the atlas can describe.
**Design:** for the concept neurons profiled on the atlas's 10M tokens, run the same profile on 2M NEW FineWeb tokens
(documents not in the atlas). Rank correlation between the two profiles of (a) firing rate, (b) on-concept share.
**Prediction:** Spearman >= 0.8 for firing rate and >= 0.6 for on-concept share.
**Fails if:** below either.
**RESULT (17 Oct 13:51, `p5_score.py`): HOLDS.** 2.0M new FineWeb tokens (the 4,102 documents after the atlas's 20,359;
alignment with the atlas asserted). Firing rate Spearman 0.986 (1,100 neurons); on-concept share Spearman 0.883 (1,024
neurons with >= 20 firings in both texts).

## Novelty ranking (after the tests)
Every finding that survives is checked against LITERATURE.md and new searches, and labelled known / partly known /
apparently new, as in the 16 Oct literature check.

# Layers and jobs in Qwen3-8B: what we can defensibly say (draft, 18 Oct 2026)

This is a synthesis of the experiments on the research branch that bear on one question: **does a neuron's layer
predict its job?** Each claim is graded by how many independent methods support it, how large and how well-controlled
the effects are, and what would overturn it. Details and numbers are in RESULTS.md (dated entries named in brackets).
Scope: one model (Qwen3-8B, 36 layers x 12,288 MLP neurons, bf16) and mostly one task (answering a question whose
answer is a name or a short fact). Layer bands are bins over what is a gradient, not hard boundaries.

## Summary table

| layer band | what neurons here do (best current reading) | strength of evidence |
|---|---|---|
| **28-35** (late) | Write specific output tokens: 'say X' (the answer word), attribute words, relation answers; act at the answer position; mostly directly on the output | **Strong** - five independent methods agree |
| **20-27** (mid-late) | Mixed: relation extraction at the answer position, task/format machinery, intermediate composition; little readable output of their own | **Moderate** |
| **10-19** (mid-early) | Act at the entity's own name tokens; carry entity-specific knowledge; work only through later neurons | **Strong** for entity neurons (S4); moderate otherwise |
| **0-9** (early) | Same as 10-19, more so: entity-specific neurons at the entity's name tokens (median layer of direct entity neurons: 8); no direct output | **Strong** for entity neurons (S4) |
| 16-24 (cross-cutting) | Hosts 'always-on' answer-format machinery shared by hundreds of concepts | Moderate |

## Strongly supported

**S1. Late neurons (about layers 28-35) write output tokens: their job is readable as the words they push.**
- *Where key neurons sit:* in 4,194 concept maps, 68% of top-5 neurons are in layers 28-35 (run 1).
- *How they act:* 'say' neurons put their effect on the answer score directly (direct share 1.05; 408 neurons, 555
  concepts); early neurons have no direct effect at all (0.00) (indirect path).
- *Forced readout:* forcing one neuron hard gives a consistent word list only for late neurons - top word consistent
  across backgrounds 91-100% in layers 28-35, vs 2-36% in layers 0-23, at 4x and 16x its natural peak (fast readout).
- *Facts:* late fact neurons act at the answer position, not at the subject (zeroed only at the last position: drop
  1.60; only at the name tokens: 0.00; 243 neurons) and include relation neurons (facts grid).
- *Naming vs knowledge:* zeroing a landmark's naming neurons (mostly late) cuts naming (3.62) but leaves facts about it
  intact (0.03); the late subset alone: naming 6.44, facts 0.30 (question-side, 15 landmarks).
- *Would be overturned by:* late neurons whose forced output is unreadable yet who carry entity knowledge at the
  subject position - not seen in 243 late fact neurons or the 1,800-neuron layer sweep.

**S2. Early neurons (about layers 0-19) do not write words; they work through later neurons.**
- *Direct effect:* 0.00 for early key neurons (516; indirect path).
- *Composition:* zeroing an early key neuron cuts a later 'say' neuron's activation by 11.7% vs ~0 for an
  activation-matched null (312 pairs, p < 1e-50) (indirect path).
- *Forced readout:* no consistent expression even at 16x (fast readout); one step downstream (layers 24-27) the neurons
  they drive are themselves unreadable (fan-out smoke test).
- *Would be overturned by:* a population of early neurons with a consistent forced word output - none found among 800
  random early neurons.
- **Revised 18 Oct (J-lens pilot):** too strong as worded. Early entity neurons do not write the NEXT token, but their
  output directions carry their entity's words: 42% under the J-lens, 29% even under the plain logit lens, vs 1% for random
  neurons (strict own-name match 16% / 8%). Read S2 as "early neurons do not write the next word; what they write is
  concept / entity content readable through a lens". (J-lens readouts are noisy before layer 20 at 32 windows.)

**S3. Late is the readable end of a feed-forward assembly.** Every type of key neuron feeds later key neurons of the
same concept (early -> say, attribute -> say, say -> later say; all p < 1e-5), mostly within the answer position and
~1/5 between words through a small set of heads (mid-layer meaning carriers L23H10, L26H20; late turn-boundary copiers
L34H28, L35H22); the answer is staged at layers 28-29 at the turn boundary (indirect path; cross-position; attention study).

**S4. Early layers (median layer 8) hold entity-specific neurons that act at the entity's own name tokens and support
all of that entity's facts** (scaled run, 18 Oct; facts2.py).
- *Direct search:* for 86 entities of three kinds (landmarks, people, works; 333 facts), neurons were ranked by
  attribution summed over ONLY the entity's name-token positions, across ALL 36 layers, then each zeroed only at the name
  tokens. Of the top-3 per entity with a mean drop > 0.2 (218 neurons): median layer 8, 89% below layer 20 - for all
  three kinds (landmarks 8, people 8, works 8.5). The layer result is not built in: late neurons at the name tokens could
  have been selected and were not.
- *Specificity:* zeroed at the name tokens of every fact, own entity's facts drop a median 0.58 log-odds, other entities
  of the same kind 0.015, other kinds 0.008; 209 of 218 (96%) are >= 5x more specific to their own entity.
- *Examples:* L8:9458 Blue Mosque (own 9.6, others 0.03); L7:8413 Ulysses (8.7); L0:1550 The Starry Night (6.7, layer 0);
  L1:4818 Sagrada Família (6.3); L7:990 Genghis Khan (4.7); L8:334 Sistine Chapel (4.1) - the same L8:334 verified for
  'Vatican City' in the concept atlas.
- *Why strong:* three entity kinds, a direct search not biased to late layers, a 96% specificity rate against two control
  groups, and agreement with the position test (M1) and the question-side test. *Limits:* one model; median per-neuron
  effect is moderate (0.58; a few are very large); 86 entities, 5 relations each.

**S4b. ...but early entity neurons are broad on the input side and partly shared** (entity_sharing.py). Their web-text
firings are mostly not about their entity (median 5%): they fire on attributes ('Turkish', 'Vatican') or word parts /
meanings of the name ('sacred', 'night'); 55% also matter causally for >= 1 other entity, coherently at layers 7-13
(Renaissance paintings, literary epics, bridges) and more mixed at 21-23. They are not simple word detectors (the
Starry Night neuron does nothing for The Night Watch). Polysemantic on input, specific in use - the same pattern as
concept neurons. (Moderate as a general claim: 218 neurons, one task.)

## Moderately supported

**M1. Early neurons (0-19) that matter for facts act at the entity's own name tokens and carry entity-level knowledge.**
(Replicated on 333 facts / 3 kinds: layers 0-9 2.90 of 2.90 at the name; people weaker.)
- *Position:* zeroing them only at the landmark's name tokens reproduces nearly all of their effect (layers 0-9: 2.60 of
  2.72; 10-19: 1.46 of 1.58), at the last position nothing (0.00 / 0.01) (facts grid).
- *Job class:* 23 of 59 are entity neurons (all of one landmark's facts drop, other landmarks' do not), 1 is a relation
  neuron, the rest fact-specific (facts grid).
- *Meaning, not naming:* the early subset of naming maps hurts facts more than naming relative to the late subset
  (facts 0.97 vs naming 1.44; late 0.30 vs 6.44) (question-side).
- *Why only moderate on its own:* 59 early neurons, landmarks only, found via answer-ranked maps; class thresholds chosen
  by us (stated before looking). Now backed by the direct entity search (S4).

**M2. Mid-to-late neurons (20-35) include relation neurons that act at the answer position.** (Scaled: 45% of layer-20-27
fact neurons are relation neurons, 255 overall across 11 relations.) The cleanest example,
L30:8291 ('century built'), costs ~3.2 log-odds on the century fact of every landmark tested and nothing on their other
facts; 61 relation-class neurons, 11 of them in 20-27 and 50 in 28-35 (facts grid). *Why only moderate:* 5 relations,
one entity type, one strong example dominating.

**M2b. Fact knowledge appears as shared late machinery of two kinds** (facts grid re-read): answer-value neurons shared by
every fact with the same answer (124/219; e.g. a '19th century' neuron across four 19th-century landmarks, 'Vatican City',
'marble'; layers 29-33) and relation neurons shared by question type (45/219; 'century built'). The fact itself is a
factorised lookup - entity (early, rows) x relation (mid-late, columns) -> value (late, shared by all entities with that
value). The entity -> value route is causal and specific: zeroing an entity's top-5 early neurons at its name lowers the
shared value neuron at the answer (median 7.3%; answer -1.78 log-odds; controls exactly 0; 71% of 657 pairs) - but
distributed (top-1 alone: 2.2%; the pre-set 20% prediction failed). Moderate.

**M2c. Numeric facts (years, heights) follow the same layer/position pattern, and late shared neurons include era
neurons that cross formats** (numeric.py, 18 Oct; 86 facts: 75 years, 11 heights; Qwen writes numbers one digit per
token; measure = exact float64 log-odds per digit, summed; random-neuron noise floor 99th pct 0.65, threshold 1.0).
- *Position replicates:* layers 0-19 act at the name tokens (16.55 of 16.83) and nothing at the answer positions (0.10);
  layers 20-27 and 28-35 act at the answer positions (10.94 of 12.41; 9.64 of 10.52) and nothing at the name (0.01 / 0.00).
- *Early neurons support more of the number:* a median 50% of its digits per early neuron vs 25% (one digit) for mid/late.
- *Era neurons:* L31:8930 hurts 18 of 27 years 18xx (median 3.96) and nothing else (0 of 5 17xx, 0 of 54 others) - and it
  is the same neuron facts2 found as the '19th century' answer-value neuron (word answers). L33:12258 does the same for 17xx
  (4 of 5, median 27.7). One neuron, one era, either format.
- *But clean digit/prefix neurons are rare:* of 99 neurons hurting >= 3 facts, 5 are digit neurons (>= 80% of hurt facts
  share one digit at one position) and 0 are 3-digit-prefix (decade) neurons; overall the facts a neuron hurts share a
  digit only slightly more than random sets (33% vs 26%). 16 broad neurons hurt >= 20 facts (L35:9490: 82 of 86).
- *No magnitude neurons* (numeric_mag.py dose series, -4x to 4x, plus 43 less-certain numbers): lost probability goes to
  neighbouring digits, but equally for random neurons (76% vs 81%); graded dose responses (36-51% of top neurons vs 0-2%
  random) move towards the neuron's own digit in 158 of 159 - digit supporters, not a bigger/smaller axis. The number line
  is in the output embedding (digit-token cosine falls steadily with distance, 0.78 -> 0.67).
- *Era-concept vs digit neurons* (era.py; 280 entities, 14th-21st century, 4 question formats + plain text): era neurons
  fire for their era in every format (median AUC 0.87-0.94 vs controls 0.51-0.60) and weakly when the entity is merely
  named (0.60); L28:9936 (18th) is near-perfect in plain text too. Their direct output is cross-lingual history (L31:8930:
  'Victorian', 'Civil', Taiping, Cixi; L28:8912: 'Bolshevik', 'Lenin', 民国). numeric.py's 'era' neurons L33:12258 /
  L33:12280 FLIP in the century format ('17xx' is '18th') - they are digit neurons ('7' / 'seven' / 七). Activation is
  format-free but necessity is format-bound: L31:8930 is needed for year / decade answers (1.43 / 2.52), not '19th century'
  (0.12); L33:2629 the reverse. Moderate.

**M3. Layers ~16-24 host shared answer-format machinery.** 35 neurons in 16-24 fire for every concept (cross-activation,
16 Oct); the most widely shared neurons (hundreds of concepts, e.g. L20:7473) are fed by the same upstream neurons
whatever the concept (feeder Jaccard 0.45 vs 0.11 for topic neurons; feeders) and drive 'Title-Case term' neurons at
layers 24-25 (fan-out). *Why only moderate:* defined by sharing, not tested for meaning; specific to the 'Answer with just
the name' format.

**M4. The on-topic usefulness of concept neurons is a late-layer property.** In web text, switching a concept neuron off
on its own topic hurts prediction in layers 24-35 (+0.004-0.005 nats/token), not in earlier layers (interference).
*Why only moderate:* single-firing effects are only ~1.5x bf16 noise.

## Not established (do not claim yet)
- **Clean boundaries.** All bands are bins; layers 20-27 are a genuine mix (relation, machinery, composition).
- **Early neurons 'are' entity representations in general.** Shown for 86 entities of three kinds in one question format;
  not yet for other kinds of entity, other tasks, or how many early neurons are entity neurons overall.
- **The input side** is tested only for entity neurons (S4b) and concept neurons (earlier: polysemantic in web text);
  how input polysemanticity varies by layer in general is untested.
- **Generality.** One model, bf16, mostly one prompt format; other tasks (reasoning, generation) may use layers
  differently.
- **That 'job' means the same thing at every depth.** Late jobs are word-like; early jobs are entity- or fact-level and
  only visible through causal tests on facts, not through any word readout.

## Methodological cautions that affect every claim
- **Selection bias towards late neurons:** ranking neurons by their effect on the answer's probability favours neurons
  that write the answer. Early neurons are under-sampled unless ranked at the subject position.
- **The 'say'/'attribute'/'early' roles are partly definitional:** 'early' was defined as layer < 20, so role-by-layer
  tables restate that cut; the position and readout tests are the non-circular evidence.
- **bf16 noise:** single-neuron effects in web text are ~1.5x rounding noise; prompt-level knockouts on recall/facts are
  far above it (null drops ~0.03 vs effects of 1-10 log-odds).
- **Predictions that failed:** 'fact neurons sit earlier than naming neurons' failed as stated (answer-ranked fact maps
  are as late as naming maps) - the layer/job link shows up in where neurons act and how, not in where answer-ranked
  maps put them. Numeric facts: 'late writers are prefix / digit neurons' held only for a few era neurons (M2c).
- **Saturation:** log-probabilities near certainty hide knockouts (numeric facts first measured with log-prob showed
  drops of 0.000 where log-odds show 10-40). Use exact log-odds (ISSUES #35) and attribute log-odds, not log-prob.

## Next tests that would most strengthen or overturn the picture
1. **Done (scaled grid):** replicates for landmarks and works; people weaker (fewer early entity neurons, relation neurons
   dominate their mid layers) - investigate why.
2. **Relation neurons beyond 'century':** do 'nationality', 'creator', 'language' neurons exist, and where?
3. **Input side by layer:** what tokens / contexts each class fires on (the atlas), to connect entity neurons to the
   entity's name tokens on the input side as well.
4. **A second model** (e.g. a smaller Qwen3, or another family) to test whether the layer/job profile is general.

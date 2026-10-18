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
| **10-19** (mid-early) | Act at the entity's own name tokens; carry entity-level knowledge; work only through later neurons | **Moderate** (small n) |
| **0-9** (early) | Same as 10-19, more so: act only at the entity's name tokens; entity or fact-specific jobs; no direct output | **Moderate** (small n) |
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

**S3. Late is the readable end of a feed-forward assembly.** Every type of key neuron feeds later key neurons of the
same concept (early -> say, attribute -> say, say -> later say; all p < 1e-5), mostly within the answer position and
~1/5 between words through a small set of heads (mid-layer meaning carriers L23H10, L26H20; late turn-boundary copiers
L34H28, L35H22); the answer is staged at layers 28-29 at the turn boundary (indirect path; cross-position; attention study).

## Moderately supported

**M1. Early neurons (0-19) that matter for facts act at the entity's own name tokens and carry entity-level knowledge.**
- *Position:* zeroing them only at the landmark's name tokens reproduces nearly all of their effect (layers 0-9: 2.60 of
  2.72; 10-19: 1.46 of 1.58), at the last position nothing (0.00 / 0.01) (facts grid).
- *Job class:* 23 of 59 are entity neurons (all of one landmark's facts drop, other landmarks' do not), 1 is a relation
  neuron, the rest fact-specific (facts grid).
- *Meaning, not naming:* the early subset of naming maps hurts facts more than naming relative to the late subset
  (facts 0.97 vs naming 1.44; late 0.30 vs 6.44) (question-side).
- *Why only moderate:* 59 early neurons, landmarks only, found via maps ranked by the answer's probability (which favour
  late neurons); class thresholds were chosen by us (stated before looking). The scaled run (people, works, and entity
  neurons ranked directly at the name tokens) is in progress and will strengthen or weaken this.

**M2. Mid-to-late neurons (20-35) include relation neurons that act at the answer position.** The cleanest example,
L30:8291 ('century built'), costs ~3.2 log-odds on the century fact of every landmark tested and nothing on their other
facts; 61 relation-class neurons, 11 of them in 20-27 and 50 in 28-35 (facts grid). *Why only moderate:* 5 relations,
one entity type, one strong example dominating.

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
- **Early neurons 'are' entity representations in general.** Shown only for neurons that fact maps select, for landmarks.
- **The input side.** Everything above is about what neurons DO (output, causal role, position). What makes them fire, and
  how polysemantic that is by layer, is untested.
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
  maps put them.

## Next tests that would most strengthen or overturn the picture
1. **Scaled grid + direct entity neurons** (running): people, works and landmarks; entity neurons ranked at the name
   tokens. Strengthens M1 if early, entity-specific neurons appear for all three kinds.
2. **Relation neurons beyond 'century':** do 'nationality', 'creator', 'language' neurons exist, and where?
3. **Input side by layer:** what tokens / contexts each class fires on (the atlas), to connect entity neurons to the
   entity's name tokens on the input side as well.
4. **A second model** (e.g. a smaller Qwen3, or another family) to test whether the layer/job profile is general.

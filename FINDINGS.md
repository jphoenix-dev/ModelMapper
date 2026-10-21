# Findings

What Model Mapper has found so far about the MLP neurons of **Qwen3-8B** (with a Qwen3-0.6B comparison), how sure we
are, and what we got wrong. Every number comes from switching neurons off and measuring the change in the model's
log-odds for the right answer, against random neurons switched off the same way. The full record of how each result
was obtained, failures included, is the Zenodo snapshot: [doi:10.5281/zenodo.23459926](https://doi.org/10.5281/zenodo.23459926).

Status: exploratory unless marked *pre-registered*. One model family, one main task (naming a concept from questions
about it). Neuron names are reading aids, not proof of what a neuron does.

## 1. A handful of neurons carry each concept

Run 1 mapped **4,194** Wikipedia Vital Articles concepts (of 5,000 attempted; Qwen must answer >= 5 questions
correctly for a concept to be mapped).

| | 3,262 concepts mapped on RTX 3090s | 932 on an RTX 5090 |
|---|---|---|
| 5 or fewer neurons lower the answer's log-odds by >= 3 | 99.0% | 99.0% |
| Median drop from the top neuron alone (log-odds) | 3.29 | 3.29 |
| Top 5 switched off vs 5 random neurons (median drop) | 8.31 vs 0.000 | 8.34 vs 0.000 |
| The answer flips with 5 or fewer neurons | 92.8% | 92.4% |

The same result, on different concepts, on two GPU types. The earlier 158-run study (16 Oct) found the same: <= 5
neurons in 157/158 runs, and a pre-registered replication on 40 new concepts passed 6 of 7 predictions.

**What this does and doesn't show.** The neurons are *necessary* for the answer, not proven to *store* the
knowledge; the network partly compensates when neurons are removed. And each concept's list is incomplete (section 3).

**Are the links specific?** The map links each neuron to the concepts it was found to matter for, but the search only
tested a neuron where a fast estimate flagged it. So 100 mapped neurons were also switched off on the questions of 200
random concepts, flagged or not (one GPU configuration, 20 random neurons per concept as the floor). Their own known
concepts still broke in 99.9% of 3,132 pairs; never-flagged concepts broke in 1.25% of 19,585 pairs, against 1% for
random neurons by construction. Specific neurons (needed by 1-3 concepts) behave like random neurons off their map
(0.76%); shared neurons are broader (4.0%). One never-flagged pair dropped the answer by more than 1 log-odds (a shared
neuron on Polar vortex).

## 2. Most of a concept's key neurons are late

About **two thirds** of each concept's top-5 neurons are in the last eight layers (28-35): 68% as mapped, 66% with the
missed neurons of section 3 added, 65% ranked by change in probability rather than log-odds. A smaller cluster sits
around layers 8-10, where the model takes in what the question is about. Caveat: finding neurons by their effect on
the answer favours the layers that write it; early neurons are under-counted.

## 3. The first search missed some neurons

Run 1 picked 200 candidates per concept with a fast estimate and switched each off; that estimate ranked candidates
slightly differently from the measure they were judged by, so some strong neurons were never tested. Re-checking every
missed candidate (3,159 of the 4,194 concepts so far), judged only on *stable* calls (section 6): **66%** of concepts gain at least
one neuron that clearly beats run 1's 30th; another 28% gain only borderline ones. Those clear entrants change the
probability of the right answer by a median of **0.78 percentage points** (17% by 5 points or more). So the published
top-30 lists are incomplete, and "a handful is enough" is a lower bound.

## 4. Neurons stay on across answer formats

Asking for the answer in a different form ("in as few words as possible" instead of "just the name"): **95%** of a
concept's top-30 neurons stay at least half as active, and for **13%** switching them off matters less than half as
much in the short format (scored on the same answer in both formats).

## 5. Some neurons have readable jobs

Examples, each checkable on the [map](https://eiffel-tower-qwen.vercel.app/map/):
- **A cycling neuron (layer 31, neuron 10454).** It is the single most important neuron for Bicycle, Cycling and Cycle
  sport, and pushes the model towards "cycle, cycling, cycles". Switching it off alone takes the right answer from
  19-96% likely to about 0% on all 11 Bicycle questions.
- **An 18th-century neuron (layer 28, neuron 9936)** fires for 18th-century people, places and works whether the date
  is asked as a year, century, decade or period (AUC 0.98-0.99 in most formats, 0.78 in one; also in plain text). Beside it, a neuron that looks like one
  (layer 33, neuron 12258) turns out to track the digit 7 in "17xx".
- **A north-south scale (layer 32, neuron 11585).** Pushing it one way moves answers towards "north", the other way
  towards "south", with a random neuron pushed the same amount as a control. Only gentle pushes give clean results.
- **About half of neurons verifiably do one job** in a calibrated census of 1,000 random neurons (~50% of positive
  poles, ~47% of negative; the bar passes only 8% / 4% of known two-job neurons). Concepts share readable parts: related
  pairs share 2.6 specific top-50 neurons vs 0.13 for random pairs.
- **1,827 neurons have a short convenience name** that passed a blind check: a second AI, given only the name, picks out
  a hidden concept that needs the neuron from five options.

## 6. Hardware: what reproduces and what doesn't

The model runs in 16-bit arithmetic, and different GPUs and software versions round in slightly different orders.
- **Same GPU type and software: bit-identical.** Two separate RTX 3090 machines agreed on all 169,937 numbers of a test
  set; the 5090 reproduced its own earlier measurements to the last stored digit.
- **Across GPU types: small differences.** Per-neuron effects typically differ by 0.02-0.04 log-odds; whole concepts can
  shift by up to ~0.2. Big effects (a concept's top neuron typically drops the answer by over 3) are unaffected; calls that sit
  within a few hundredths of a threshold are not reliable across cards.
- **So every measurement records its GPU and software, thresholds are re-measured in the same pass, and any call within
  0.13 log-odds of its threshold** (the 99th percentile of 3090-vs-5090 differences) is labelled *uncertain* rather than
  silently in or out. A test of that band agreed on 98.3% of stable calls across cards (predicted >= 99%, so it narrowly
  failed; all the misses came from one concept whose whole set shifted). Re-measuring the threshold in the same pass is
  designed to remove such shifts; that is still to be tested on new data.
- **Even the PyTorch build matters a little.** On one CPU, Qwen3-0.6B, the same code and questions: torch 2.11.0+cu128 vs
  2.11.0+cpu gave per-neuron differences of median 0.017 and at most 0.078 log-odds, the same top-30 set and the same
  marker neuron in 2nd place (one concept; GPU builds not yet compared).
- **The 0.13 band was calibrated on Qwen3-8B.** A smaller model has smaller effects (Qwen3-0.6B's 30th neuron for
  Bitcoin drops the answer by 0.27), so the same band marks most of its list uncertain; each model needs its own
  calibration.

## 7. Reading early neurons with the J-lens

Reading a neuron by passing its output straight to the vocabulary works for late layers and gives noise early on. The
J-lens (after Anthropic's J-space work) measures, on real text, what the later layers turn a nudge at each layer into.
Estimated from 256 pieces of web text, it gives early neurons readable words (e.g. layer 10 neuron 6062: "barcelona,
catalonia, spain" for Park Güell) and is stable between two halves of the text (overlap 0.68-0.80 in layers 0-15). It
finds the entity's own words for 45% of entity neurons vs 29% by the direct reading. It is worse than the direct reading
around layers 21-29, and it is an average effect, not proof.

## Corrections

- **19 Oct 2026: "42% of neurons are format-bound" (and an earlier 38%) withdrawn.** The short format was scored on
  Qwen's own short answer ("Capital stock." rather than "Capital") while the original format used the run's answer, so
  a change of target looked like a change of format. Measured on the same answer: 13% (section 4).
- **19 Oct 2026: the missed-neuron figures were qualified.** They compared new measurements against a threshold measured
  earlier, on a different GPU for 78% of concepts. Re-scored on stable calls against a threshold measured on the same
  configuration: 66% (section 3), not 81%.
- **19 Oct 2026: run 1's hardware was misstated** as "RTX 3090s"; 22% of concepts ran on an RTX 5090.

## Limits

One model family and size pair; one main task; a bias towards late layers; neurons that matter for only one concept
can't be named yet (the blind check needs a second concept); attention heads are mapped far less thoroughly than
neurons; names and pushed words are informal readings.

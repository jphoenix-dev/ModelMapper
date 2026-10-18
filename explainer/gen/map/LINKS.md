# Links between a concept's key neurons: method and pass rule (written 18 Oct 2026, before any data)

**Purpose.** A graph view on the neuron map: for a chosen concept, its 30 key neurons by layer, with arrows where
switching one neuron off measurably weakens a later one. James (18 Oct): run it on rented GPUs through the gpu-fan-out
session, and register the links in the research ledger.

**Status.** Exploratory. Generalises `src/indirect_path.py` (research/attention, 555 concepts, top 5 neurons, whose
composition result held: e.g. attribute -> later 'say' neuron, median relative drop 0.127 vs null -0.003) from the
top 5 to the top 30, on all run-1 concepts. The rule below is fixed before the run; results are reported whatever
they show.

## Scope

All 4,194 mapped concepts of loop v2 run 1 (D:/already-clean-data/loop_v2/run1/results/results/map/*/records.jsonl):
each concept's top 30 neurons in run 1's rank order (`rank.top`), its recalled prompts (`rank.prompts`), and the
first token of its answer (`rank.answer_ids[0]`). Model: Qwen3-8B at the pinned revision, bf16, as run 1.

## Measurement (per concept, per recalled prompt; answer position = the last prompt token)

1. Pass 1, batch of 1: activations at the answer position for every layer holding a top-30 neuron, to pick nulls.
2. Pass 2, one batch of 61 identical rows:
   - row 0 as is;
   - rows 1-30: one top-30 neuron zeroed at **every** position (as in the maps);
   - rows 31-60: for each of those, an **activation-matched null**: the neuron in the same layer, outside the
     concept's top 30, whose |activation| at the answer position is closest to it, zeroed the same way.
3. Record per row: the activations of all 30 top neurons at the answer position, and the answer's first-token
   log-odds. Record the null ids and their base activations.
4. Gate (as indirect_path.py): a zeroed neuron must read exactly 0 at the answer position in its own row; row 0's
   first-token log-odds must be >= the saved whole-answer baseline - 0.5 (bf16 batch-shape tolerance). Report the
   count of prompts failing either.

## Link strength

For an earlier neuron i and a later neuron j (layer(i) < layer(j); neurons in the same layer run in parallel and
cannot affect each other), per prompt:

    drop_ij = (a_j(base) - a_j(i zeroed)) * sign(a_j(base)) / |a_j(base)|     (relative drop in j's own direction)
    null_ij = the same with i's matched null zeroed instead of i

averaged over the concept's prompts (prompts where |a_j(base)| < 1e-3 are skipped for that pair).

## Pass rule (fixed before the run)

A link i -> j is drawn on the map if **all** hold:
- |mean drop_ij| >= 0.05 (a 5% change in j);
- |mean drop_ij| >= 3 x |mean null_ij|;
- the sign of drop_ij agrees with the mean in at least 2/3 of the concept's prompts (and in at least 3 prompts).

Positive drop = i **supports** j (switching i off weakens j); negative = i **suppresses** j. Both are kept; the map
draws them differently. Every pair's numbers are stored, passing or not.

## Outputs

One record per concept: units (30), types (run-1 roles), per prompt the 61 x 30 activation matrix, log-odds per row,
null ids; plus a links file: concept, i, j, mean drop, mean null, n prompts, sign agreement, passes. Drained to
`D:/already-clean-data/loop_v2/links_run1/`. The research ledger imports the links as unit -> unit2 evidence
(experiment `links_run1`), and the existing indirect_path.jsonl (555 concepts, top 5) as a second experiment.

## Cost estimate

Per concept ~10-17 prompts x one 61-row forward pass of ~40 tokens, plus the small pass 1: a few seconds on a 3090.
4,194 concepts: roughly 2-4 GPU-hours, about $1-2 on rented RTX 3090s at $0.22/h, plus boot overhead.

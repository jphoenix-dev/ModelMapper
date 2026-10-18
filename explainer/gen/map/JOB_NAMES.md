# Convenience names for neurons: method and pass rule (written 18 Oct 2026, before any names were generated)

**Purpose.** Give the neuron map a short, plain-English name for what a neuron does, where the evidence supports
one, so a lay reader can hover over a dot and get its gist. James (18 Oct): Luna labelling is fine, "but only for
high certainty jobs; we don't want to overclaim, and all labels should be clear they are for convenience rather than
a 'true meaning'". Neurons without a clear job are shown as contributing to their concepts, with no name.

**Status.** Exploratory. The pass rule below was fixed before the run; results are reported whatever they show.

## Who gets a name

Neurons in at least one concept map (top 30) that matter for **at least two distinct concepts**: one concept is
held out for the check below. Neurons used by a single concept get no name on the map (the check isn't possible).

## Naming (GPT-6 Luna)

For each neuron, Luna sees the evidence we have, **with one concept held out** (chosen at random, seeded by the
neuron id):
- its layer, and whether it is early (layer < 20) or late;
- the words it pushes, read from its weights (flagged as unreliable before layer 20, LAYERS_AND_JOBS S1/S2);
- the other concepts that need it, with how much the answer suffers without it;
- blind web-text labels (census / H-DEPTH) where they exist, with whether they passed the calibrated one-job bar;
- fact, entity, relation and era findings about it, minus any that name the held-out concept.

Luna returns a name of 2-6 words (or none), a confidence (high / medium / low) and the evidence it rests on. It is
told to return no name when the evidence has no clear common thread, and not to name a single concept unless the
evidence is about that concept.

## The check (Jev, blind)

Jev sees only the name, and is asked which of five concepts the neuron most likely helps the model name: the held-out
concept plus four distractors that do not use the neuron (from the same Wikipedia section as the held-out concept
where possible, else any mapped concept), in shuffled order. Chance is 20%.

**Control.** The same question with the name of a different, randomly chosen named neuron. Its pass rate is the
false-pass rate of the check.

## Pass rule (fixed before the run)

A name is shown on the map as a **checked convenience name** only if:
1. Luna's confidence is **high**, and
2. Jev puts **p >= 0.5** on the held-out concept (chance 0.2).

Everything else (medium / low confidence, no name, or failing the check) is kept in the data but not shown as a
name. The map says of every name: "a short label to help you read the map, not the neuron's true meaning".

## Outputs

`D:/already-clean-data/job_names/names.cache.jsonl` (every Luna and Jev result, resumable) and `summary.json`
(counts, pass rates by layer band and role, control false-pass rate, spend). Budget cap: $5.

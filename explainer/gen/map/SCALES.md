# Two-ended neurons: one scale, or two jobs? Method and pass rule (written 18 Oct 2026, before any data)

**Question.** A neuron pushes the model along one fixed arrow; firing positively pushes one way, negatively the
other. Most mapped neurons matter for concepts in one direction only. In run 1, 1,567 of 21,510 neurons that matter
for some concept (drop >= 0.5 log-odds) matter for some concepts firing positively and for others firing negatively.
An exploratory word-match (explainer/gen/map/bipolar_check.py) found examples that look like **one scale with two
poles** (south <-> north, orange <-> blue, moon <-> planet, birth <-> pregnancy) and a few that look like **two
unrelated jobs** (information <-> sports). This test classifies all 1,567 and checks the classification blind.
James approved it on 18 Oct ("Agree further experiments"). Exploratory; the rule below is fixed before the run.

## Inputs (per neuron)

- Each end's top-10 pushed tokens, read from the weights (data/push, push_words.py), without the concepts' names.
- The concepts that use each end (run 1, drop >= 0.5, sign of the activation at the answer position).

## Step 1: Luna describes and classifies, from the pushed words only

GPT-6 Luna sees only the two word lists ("end A", "end B"; which is positive is randomised per neuron, since the sign
is a convention) and returns: a short description of each end (or none), and a relation: `opposites` (two ends of
one scale, e.g. south / north), `related` (different but connected), `unrelated`, or `unclear`; plus a confidence.

## Step 2: usage check (Jev, blind)

For each end, one concept that uses it (drawn at random, seeded by the neuron id). Jev sees Luna's two descriptions
and the concept's name, and says which end the concept belongs to (two options, order shuffled). Pass: p(correct
end) >= 0.8 for both concepts. This checks that Luna's descriptions, written from weights alone, predict how the
model actually uses each end.

## Step 3: second opinion on the relation (Jev, blind)

Jev sees only the two descriptions and picks opposites / related / unrelated.

## Control

200 shuffled pairs: one neuron's end A with a different neuron's end B, through steps 1 and 3. The rate at which
these are called `opposites` is the false-positive rate of the classification.

## Pass rule (fixed before the run)

- **Two-ended scale:** Luna `opposites` with confidence high, Jev's top choice `opposites` (p >= 0.5), and the usage
  check passes.
- **Two unrelated jobs (output-side polysemantic):** Luna `unrelated` with confidence high, Jev's top choice
  `unrelated` (p >= 0.5), and the usage check passes.
- Everything else is recorded but not claimed.

## Step 4 (causal, GPU; run by the research session)

For a sample of passing neurons (up to 20 scales and 10 unrelated pairs): on each end's concept's recall questions,
add the neuron at +k and at -k times its peak activation (k in 1, 2, 4) and measure the change in each end's answer
log-odds. Prediction for a scale: one sign raises end A's answers and lowers end B's, the other sign the reverse;
a random same-layer neuron at the same strengths moves neither.

## Outputs

D:/already-clean-data/scales/ (resumable cache, summary.json); imported into the ledger as experiment `scales`.

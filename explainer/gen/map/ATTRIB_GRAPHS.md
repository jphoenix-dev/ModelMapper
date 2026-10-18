# Attribution graphs on Qwen3-8B's own neurons: method and checks (written 18 Oct 2026, before any data)

**Goal.** An attribution-graph view on the neuron map, like Neuronpedia's circuit tracer and the figure in "Neural
Networks Are Already Clean": for one question, labelled groups of neurons in rows from the clues in the question up to
the answer, with weighted links, where hovering a group shows its neurons and clicking it removes them and shows what
happens to the answer. James asked for this on 18 Oct. The difference from Neuronpedia's graphs: nodes are the model's
own MLP neurons, not transcoder features, so there is no reconstruction error to account for.

**Status.** Exploratory pilot. Checks below are fixed before the data.

## Pilot questions

18 run-1 concepts (D:/already-clean-data/attrib_graphs/pilot_prompts.json): for each, the recall question the model
answers most confidently (chat template, thinking off, as in run 1), the answer form and its token ids.

## 1. The graph (GPU)

For one question, at every position:
- **Target:** the answer's first token at the last position, scored as its logit minus the mean logit (as in
  src/indirect_path.py), which does not saturate the way log-odds do for near-certain answers.
- **Linearisation:** freeze the attention patterns, the RMSNorm scales and each neuron's gate (SiLU(gate)) at their
  values on this question. The model is then linear in the neurons' value parts, so contributions add up exactly.
- **Nodes:** (layer, position, neuron) with attribution = activation x gradient of the target; the prompt's token
  embeddings (one node per position); the target. Keep the nodes with the largest |attribution| that together
  account for 80% of the total (cap 300).
- **Edges:** for each kept node, its direct contribution to each later kept node's activation (and to the target),
  through the frozen model. Keep the edges with the largest |weight| that together account for 98% of each node's
  inflow (circuit-tracer's thresholds).
- **Completeness:** report the share of the target's score the kept nodes explain directly and through each other,
  and the share carried by paths with no kept neuron (attention-only, embeddings straight through).
- **Implementation:** try circuit-tracer (MIT) first with an exact "transcoder" that is the SwiGLU MLP itself
  (features = neurons, value part as the encoder with the frozen gate as the activation's slope, down_proj as the
  decoder, error nodes zero). If it can't express that, implement the same algorithm directly (precedent: Arora et al.
  2026, neuron-basis circuit tracing).
- **Gate:** the linearised target must match the model's real target on the unmodified question (difference < 1%),
  and zeroing a kept node must change the target by roughly its attribution (rank correlation over the top 30
  nodes >= 0.7, measured by actual single-node knockouts, against 30 random active nodes).

## 2. Groups (Luna)

Luna groups each graph's kept neurons into at most 10 labelled groups by shared meaning, using for each neuron: its
layer, the token it sits on, its checked convenience name and tested jobs, its blind labels and the words it pushes
(both directions). Each group gets a short label and a row: **Answer** (writes the answer), **The concept** (parts of
what is being asked about), **Clues** (neurons at the question's words that pick up what it is asking). Neurons
Luna can't place go to "other" and are drawn as one faint box. Labels are convenience labels, as for neuron names.

## 3. Checks on the groups (GPU)

For each group: zero all its neurons at their positions and measure the answer's probability; the same with a
random set of the same size from the same layers and positions (3 draws). A group is drawn as it is only if removing
it changes the answer more than all 3 random sets. Also: the sum of a group's attributions vs its measured effect
(reported, not a pass rule). These numbers power "click to remove".

## Outputs

D:/already-clean-data/attrib_graphs/<slug>/graph.json (nodes, edges, completeness, gate results),
groups.json (Luna), ablations.json (step 3). The ledger imports them as experiment `attrib_graphs`.

## Amendments (18 Oct 2026, after a Qwen3-0.6B validation and before any Qwen3-8B data)

The research session built steps 1 and 3 (src/loop_v2/attrib_graph.py, research/attention) and tested them on 3 pilot
questions with Qwen3-0.6B. Gates 1 and 2 passed (the linearised forward matches the model; all direct contributions
sum to the real target within 0.6%). Gate 3 failed (Spearman 0.22-0.35) because most top-attributed nodes sat at
position 0, the `<|im_start|>` attention sink, which frozen attention and norms over-credit. Changes:

1. **circuit-tracer is not used.** Its features are act_fn(encoder . x); a SwiGLU neuron with a frozen gate has a
   different slope at each position, which a fixed encoder can't express. The same algorithm is implemented directly
   (the stated fallback).
2. **Position 0 (the attention sink) is left out of the neuron nodes**, as circuit-tracer leaves out BOS. On 0.6B,
   gate 3 becomes 0.70 / 0.71 / 0.58. The gate's pass mark stays 0.7; a graph below it is still shown, with a visible
   note that its check failed, rather than hidden.
3. **Node budget and coverage.** Attribution is spread over 36 layers x ~40 positions x 12,288 neurons, so 300 nodes
   hold only ~10% of total |attribution| and the 80% target is unreachable at a viewable size. Keep the cap of 300 and
   report two numbers: that share, and a **joint knockout**: switch off all kept neurons together and measure the
   answer, against 3 random sets of the same size from the same layers and positions. The second says how much of the
   answer the drawn graph actually carries.
4. **Edges.** All edges passing the 98%-of-inflow rule are stored; the viewer draws group-to-group sums, so no further
   pruning is needed in the data.

## What the graph does not show

One question at a time; first answer token only; attention is frozen, so heads appear only as the routes edges take
(not as nodes); first-order (linearised) attributions, checked by knockouts but not exact for large changes.

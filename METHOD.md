# Method

How `map-concept` measures which neurons a concept depends on, the controls it runs, and how `check` knows an install
measures the same thing. The code is in `modelmapper/questions.py` (steps 1-3) and `modelmapper/mapping.py` (4-9).
It is the code that made run 1 of the Qwen Neuron Map, copied unchanged.

A "neuron" here is one MLP neuron: one input to a layer's `down_proj` (Qwen3-8B has 36 layers × 12,288 of them;
Qwen3-0.6B has 28 × 3,072). "Switching it off" means setting that input to 0 at every token position.

## The procedure

**Questions (the labeller).**
1. The labeller (default `gpt-6-luna`) writes 20 sentence fragments whose natural next words are the concept's
   name, plus 8 control fragments with other answers. It is told never to include the answer's words.
2. It rewrites each fragment as a question: "What is the name of ...?". Each is asked as a chat turn ending
   "Answer with just the name.", with thinking switched off. A model whose tokenizer has no chat template (a base
   model) is refused before any labeller call, so nothing is spent. `MODEL_MAPPER_PROMPT=plain` asks it
   "Question: ... Answer with just the name." then "Answer:" on the next line instead. That is experimental and
   uncalibrated, and every record says which format it used (`prompt_format` in its provenance).
3. The model under study answers each question (greedy, 16 tokens). The labeller judges which of those answers name
   the concept. The ones that do become extra answer forms ("BTC", "the Bitcoin network").

**Measurement (the model, no API calls).**
4. **Answer form and recall.** Each answer form is scored on every question: P(answer) is the probability of the
   whole answer's tokens. The form with the most questions at P ≥ 0.2 is used. The questions it gets right are kept,
   except any that contain the answer. Fewer than 5 kept: the concept is `not_recalled` and nothing more runs.
5. **Baseline.** The answer's exact log-odds, log P − log(1 − P), on each kept question. It is computed in float64
   from the logits, so it doesn't saturate near certainty.
6. **Attribution.** Activation × gradient of log P(answer), for every neuron, summed over positions and averaged over
   the questions. The 200 neurons with the largest attribution are the candidates.
7. **Single knockouts.** Each candidate is switched off alone. Its **drop** is the baseline log-odds minus the
   knocked-out log-odds, averaged over the questions. The 30 largest drops are the concept's map (`top`).
8. **Controls.**
   - *Random same-layer neurons:* one random neuron from the layer of each of the top 2, switched off the same way
     (seeded by the concept's name). Their drops are the noise floor for this concept.
   - *k curve and its random twin:* the top 1, 2, 3, 5, 10, 20 and 50 by drop switched off together, then as many
     random neurons (one from each candidate's layer). k_mean is the first k with a mean drop of 3 or more; k_flip
     the first k that takes at least half of the right answers below P = 0.5.
   - *Heads:* the 50 attention heads with the most attribution, each switched off (its slice of `o_proj`'s input set
     to 0), with their drops.
9. **Roles, from the weights.** For each top-30 neuron, its direct push on the output: its activation at the answer
   position × its `down_proj` column × the final norm, read through the unembedding. `say`: the answer's first token
   is in its top 10. `attribute`: the answer's z-score among all tokens is 2 or more. `early`: the neuron is in the
   first 20/36 of the layers (its effect on the output is mostly indirect, so this reading says little). Otherwise
   `late-other`. These are readings, not measurements of behaviour.

   Some models normalise the MLP's output before adding it to the residual (OLMo-2, Gemma 2 and 3:
   `post_feedforward_layernorm`). There the reading also multiplies by that norm's gain. The norm's overall scale
   depends on the input, so the reading is approximate: each such entry is marked `"approximate": true`, and the
   command shows `[roles approximate]`. Qwen3 has no such norm, and its readings are unchanged. The knockout drops
   (steps 5-8) are measured on the model's real output, so they are exact for every model.

Per-concept arrays (all 442,368 attribution scores, activations, residuals at the answer position, attention rows)
go to `arrays/<concept>.npz` next to the record.

## Steering

`steer` holds neurons at a value while the model writes a reply (the question itself is read normally unless
`--whole`). `x8` means 8 × the neuron's largest activation on 2M tokens of FineWeb web text; `x-8` means 8 × its
most negative one. Those come from the shipped range tables (`modelmapper/data/ranges/`, built by `modelmapper ranges`).

`--judge "question"` asks a judge model for P(yes) on each reply, plus P(fluent) as a check that steering hasn't just
broken the text. The published steering results used TypeSafe's Jev (`JUDGE_BACKEND=jev`). Any OpenAI-compatible
endpoint that returns top_logprobs also works (`JUDGE_BACKEND=llm`); its probabilities are not calibrated against Jev,
so compare like with like.

## What makes a record trustworthy

1. **Provenance.** Every record says what it was measured on: model and its revision, dtype, device, GPU and its
   memory, torch, CUDA, cuDNN, transformers, numpy, Python, OS, package version, code commit, and start and finish
   times. Nothing is left out: a field that can't be found says `unknown`. Drops differ slightly between GPUs, software
   versions and dtypes.
2. **In, uncertain or out.** Every knockout in one run is measured in the same pass, so if another card moves a whole
   concept's drops up or down, the order hardly changes. What can change is which neurons sit either side of the 30th
   place. So each of the 200 candidates gets a **margin**: its drop minus the drop of the neuron ranked 30th in this
   run. Its **call** is `in` if the margin is above the band, `out` if it is below minus the band, and `uncertain`
   otherwise (the 30th neuron is always `uncertain`). The ranking and the drops are not changed. The record counts
   `n_in` and `n_uncertain` in the top 30, and `n_out_but_close`: candidates below 30th place but within the band.

   **The band is per model.** For **Qwen3-8B it is 0.13 log-odds**: the 99th percentile (0.1285) of the per-neuron
   difference in mean drop between an RTX 3090 and an RTX 5090, on 10 calibration concepts (395 neurons; torch
   2.12.1+cu129 against 2.11.0+cu128). No other model has a band yet. Their calls are `uncalibrated` (the margins
   are still recorded), never a silent `in` or `out`. A band can be given with `--band` or `MODEL_MAPPER_BAND`, and the
   record always says which band was used and where it came from. A new model's band should come from paired runs
   of the same concepts on two configurations, added to `guards.BANDS`.

   *How well it holds.* On 10 other concepts, the stable calls agreed between the two cards 98.3% of the time. We
   predicted at least 99%, so the test narrowly failed. All 6 disagreements came from one concept whose whole set of
   drops moved by about 0.15 between cards.
   So the comparison point is the 30th neuron of the same run, re-measured in the same pass. That is designed to
   remove such whole-concept shifts, but it is still to be tested on new data. Comparing a drop with a stored number
   from another run (another card, another day) is not covered by the band, and `lookup --records` says so when it
   lists drops from several runs. (FINDINGS.md, section 6.)

   *Why not reuse 0.13 for other models.* A smaller model has smaller effects. On Qwen3-0.6B, Bitcoin's 30th drop is
   0.268, so the 8B band would mark 20 of the top 30 `uncertain`. We have no paired-hardware runs on 0.6B, only one
   comparison of two torch builds on one CPU (CALIBRATION.md: differences up to 0.078), which is not enough to set a
   band.

3. **Answer keys** (`modelmapper/answer_keys/`). For Qwen3-0.6B (3 concepts) and Qwen3-8B (10), a marker neuron that
   was needed for recall in two independent question sets, with the question set pinned. Tier 1 re-runs each marker's
   concept on its pinned questions. A marker passes at rank ≤ 2 × the key's rank + 3 with a drop ≥ half the key's
   drop. 90% of markers must pass.

## How `check` is calibrated

`modelmapper check` runs everything on Qwen3-0.6B and compares with what our machine got (CALIBRATION.md has the
runs). Each step is PASS, FAIL, WARN (works, but differs from ours or varies from run to run) or SKIPPED.

| Step | What runs | PASS when |
|---|---|---|
| key1 | answer key tier 1 | the tier 1 rule above |
| map | the whole `map_concept`, offline, on the key's pinned Bitcoin questions | L23:2392 is in the top 5 |
| steer | L23:2392 held at x8 on two prompts, twice | every reply changes, and the two runs match exactly |
| live (`--apis`) | `map-concept` for the key's 3 concepts with fresh labeller questions | 2 of 3 markers in their concept's top 20 of 200 (else WARN: fresh questions vary) |
| judge (`--apis`) | one judged reply, if a judge is set up | a probability in [0, 1] |

## Limits

- A drop shows a neuron is needed for the answer, not what it represents. Redundant neurons can each show small drops;
  the k curve shows when a concept needs many neurons at once.
- The map depends on the questions. Fresh questions move neurons near the bar; the strongest few are stable (the
  Bitcoin marker came 2nd, 2nd, 3rd, 2nd and 2nd in five runs with fresh questions).
- Only the 200 most-attributed neurons are knocked out. A neuron that matters but has low attribution is missed.
- Knockouts set activations to 0. That is not the same as the neuron's typical value; mean-ablation can differ.
- Tested on Qwen3-0.6B and Qwen3-8B. The code expects gated MLPs at `model.model.layers[i].mlp.down_proj`. OLMo-2 runs
  (0.1.1 handles its missing `head_dim` and its post-MLP norm), but no other family has been calibrated.
- Recall is scored on the answer's tokens straight after the prompt. Small chat models often answer in a sentence
  ("The mathematician is Fibonacci"), which this doesn't credit: on OLMo-2-1B-Instruct, 22 of 36 concepts came out
  `not_recalled`. *Proposed, not implemented:* take each question's greedy reply, and when it starts with a lead-in
  before the answer, score the answer after that lead-in (prompt + lead-in), rejecting lead-ins that contain the
  answer's words. That would be recorded per question, and it would stay off by default, because it changes what a
  recalled question means.
- Error messages kept in reports are cut to 300 characters, and endpoint error bodies are left out, so a report never
  carries a key or a private URL.

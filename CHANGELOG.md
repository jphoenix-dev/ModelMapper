# Changelog

## 0.2.0

Labelling: what each part of a neuron does, with every label checked (LABELLING.md). This is the method of the v3 run
on Qwen3-8B, ported from the research code and checked against that run's own files.

- **`modelmapper sweep`**: the vocabulary sweep. Every token in 12 template sentences, read for every neuron in one
  pass, gives each neuron's favourite tokens by gate, by + side and by − side, plus glitch (hub) tokens.
- **`modelmapper pseudo`**: the fake-neuron sets that labelling must fail on.
- **`modelmapper exam`**: describe (labeller) and blind exam (judge) against surface-matched decoys and control labels,
  with the list exam, bins and verdicts.
- **`modelmapper gencheck`**: the generative check of verified labels against surface-only control words.
- **`modelmapper jobtype`**: amplifier / eraser / redirector per side, from the weights.
- **`modelmapper label`**: all of it for a list of neurons, resumable, with spend caps, ending in a scorecard.
  The plan gives the cost as a measured range (Qwen3-8B to Qwen3-0.6B), the sweep's time on a 32 GB and a 6 GB GPU,
  and includes the fake neurons' own sweep; without `--budget` it caps spend at twice the upper estimate. With
  `--with-fakes` it ends with the falsifier's PASS / WARN / FAIL per part (exit code 4 on FAIL). From the laptop trial
  (21 Oct 2026: RTX 1000 Ada 6 GB, Python 3.10, behind a proxy).
- Labels whose exam passed with a control above 0.65 (bin `decoys-too-easy (intelligible)`) count as **verified at
  the edge**, with lower confidence (`at_edge`); decided after the v3 / v3x runs, on the corpus exam's evidence
  (LABELLING.md, Verdicts). The scorecard, `label`, `check --labels` and `lookup` use this rule.
- `export-pack` reads the model from a label run's results and leaves out a merged exam's describe-only and slice
  files. `check` WARNs (not PASS) on a custom labeller endpoint with no key; the version line and the hub message no
  longer show stale or empty values.
- **`modelmapper realtext`** (experimental): the corpus exam and graded simulation, the research run's code with its
  paths and model made arguments (`modelmapper/experimental/`; LABELLING.md, "Experimental"). Tested offline on
  Qwen3-0.6B, not re-run or calibrated here.
- **`modelmapper export-pack`**: labelling runs and map records as a contribution pack for the research ledger (its
  third-party JSON-lines format), with the same evidence rows and settings hashes as the ledger's own importers and the
  run's provenance (`modelmapper/pack.py`).
- **`modelmapper check --labels`**: labelling's self-calibration (`modelmapper/labelcheck.py`): `sweep1` (a sweep
  slice against our stored profiles), `exam1` (20 exams re-judged from shipped inputs, `exam.rejudge`) and
  `falsifier` (real and fake neurons labelled for real on the model under check). References for Qwen3-0.6B and
  Qwen3-8B ship in `modelmapper/data/check/`. The sweep's `.json` now records `intermediate_size`, so the exam's
  per-layer hub-token shares use the model's own width (before, Qwen3-8B's 12,288 for every model).
- Checked against the research run (tests marked `repro`):
  - pseudo-neuron sets and job types bit for bit;
  - exam decoys bit for bit, and 5 / 5 verdicts on a live re-judge;
  - the scorecard reproduces all 1,000 v3 records;
  - the sweep and the generative check are checked on the CPU against direct forward passes.
  - The sweep's and the generative check's Qwen3-8B values were not re-run here (no GPU).
- **`modelmapper lookup`** shows the published labels for the 11,000 labelled Qwen3-8B neurons of the v3 and v3x runs (gate, + side,
  − side with verdicts and scores, job type, push words, knockouts), shipped in the package
  (`data/labels/Qwen3-8B_v3.json.gz`, built by `scripts/build_label_data.py`). A word also lists the labelled neurons
  whose labels mention it. `--records` also reads your own `label` output (`*_results.json`).
- `judge.py` gains the exam's question types (`rate`, `rate_examples`); `labeller.py` gains `chat_json_n`. Existing
  APIs are unchanged; the 0.1.x commands and their results are unchanged.

## 0.1.1 (unreleased)

Portability fixes, found running 0.1.0 on OLMo-2-1B-Instruct on a second machine. Qwen3 results are unchanged, bit for bit (CALIBRATION.md).

- **Base models fail fast.** A model whose tokenizer has no chat template is refused before any labeller call
  (before, every concept failed after the labeller had been paid). `MODEL_MAPPER_PROMPT=plain` asks
  "Question: ..." / "Answer:" instead: experimental, uncalibrated, recorded as `prompt_format` in each record.
- **No `head_dim` in the config** (OLMo-2): worked out as hidden size / heads, as transformers does.
- **Roles on models with a post-MLP norm** (OLMo-2, Gemma 2/3): the reading includes that norm's gain and is marked
  `"approximate": true`. Gemma-style norms (scale 1 + weight) are read as such. Knockout drops are unaffected.
- Limits: small chat models that answer in a sentence are often `not_recalled` (METHOD.md, with a proposed fix).
- Tests: model-free unit tests for the prompt-format check, the `head_dim` fallback and the norm gains.

## 0.1.0 (19 Oct 2026)

The clean package: `map-concept`, `map-batch`, `steer`, `lookup`, `check`, the Python API, guards and docs.

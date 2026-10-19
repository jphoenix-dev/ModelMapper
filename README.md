# Model Mapper

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23459925.svg)](https://doi.org/10.5281/zenodo.23459925)

Model Mapper finds which neurons of a language model a concept depends on. Give it a concept ("Bitcoin"). It writes
questions whose answer is that concept, keeps the ones the model gets right, and switches neurons off one at a time
to see which ones the answer needs. Random neurons are switched off the same way, as a control.

It is the method behind the [Qwen Neuron Map](https://eiffel-tower-qwen.vercel.app/map/) (about 4,200
concepts mapped on Qwen3-8B). A record you make has the same format as the map's, so your results can add to it.

Version 0.1.0. Tested on Qwen3-0.6B and Qwen3-8B only.

**Tested so far** (`modelmapper check --apis`, Qwen3-0.6B; details in [CALIBRATION.md](CALIBRATION.md)):

| Machine | Python | Device | Result |
|---|---|---|---|
| Windows 10 desktop, fresh clone | 3.11.9 | CPU, float32 | PASS: the marker 2nd, live 3/3, judge OK |
| Windows 11 work laptop, RTX 1000 Ada 6 GB, corporate proxy | 3.10.11 | GPU, bfloat16 | PASS in 309 s: the same top 5, live 3/3 |

## Quick start

You need Python 3.10-3.12 and about 10 GB of disk. Qwen3-0.6B runs on a CPU. Qwen3-8B needs a GPU with 24 GB or more.

```bash
git clone https://github.com/jphoenix-dev/ModelMapper.git
cd ModelMapper
python -m venv .venv
source .venv/bin/activate                    # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu      # NVIDIA GPU: .../whl/cu128
pip install .

modelmapper check                            # does this install measure what ours did? (no API calls, ~10 min on a CPU)
modelmapper map-concept "Bitcoin" --model Qwen/Qwen3-0.6B
```

On a CPU, set `MODEL_MAPPER_DTYPE=float32` (in `.env` or the shell): it is faster there, and it is what `check` uses
when there is no GPU. The default is bfloat16, as in every published run.

`map-concept` needs a **labeller**: an LLM that writes the questions. Put `OPENAI_API_KEY=...` in a `.env` file in
the folder you run from, or point `OPENAI_BASE_URL` at any OpenAI-compatible endpoint (SETUP.md). A concept costs well under a
cent of labeller calls with the default labeller; `--budget` caps it (default $0.50).

From Python:

```python
from modelmapper import map_concept

r = map_concept("Bitcoin", model="Qwen/Qwen3-0.6B")
for n in r.top[:5]:
    print(n.neuron, round(n.drop, 2), n.role)      # e.g. L23:2392 0.85 attribute
print(r.nulls)                                     # random neurons switched off: drops near 0
```

## Commands

| Command | What it does |
|---|---|
| `modelmapper check` | Checks the install against our numbers and writes a report you can send back. `--apis` adds live labeller checks. |
| `modelmapper map-concept "X"` | Maps one concept. Saves `maps/<model>/<concept>.json`. `--questions FILE` reuses a saved question set. |
| `modelmapper map-batch FILE` | Maps a list (one concept per line). Loads the model once. Resumable: re-run it and it skips what is done. |
| `modelmapper steer L23:2392=x8` | Holds neurons at a value while the model answers, and shows the replies with and without. `--judge "question"` scores them. |
| `modelmapper lookup L31:8930` | What the published map already knows about a neuron or a concept. `--records maps/Qwen3-0.6B` searches your own. |
| `modelmapper ranges MODEL` | Builds the neuron range table `steer`'s `x` values need, for another model. |
| `modelmapper settings` | Where every setting points (never shows keys). |

## What the numbers mean

It comes down to one measurement: switch a neuron off (set its activation to 0) and see how much less likely the
model is to give the right answer.

- **log-odds.** How sure the model is of the answer: log(P / (1 - P)). 0 means a coin flip; +2 means about 88%.
- **drop.** How far the answer's log-odds fall when one neuron is switched off, averaged over the questions the model
  got right. A drop of 1 cuts the odds by about 2.7 times; a drop of 3 by about 20 times.
- **recall.** How many questions the model answers correctly (P(answer) at least 0.2). Below 5, the concept is
  `not_recalled` and nothing is mapped: the model doesn't know it well enough to test.
- **nulls.** Two random neurons from the same layers, switched off the same way. They are the noise floor; they are
  close to 0 (0.00 and 0.01 for Bitcoin on Qwen3-0.6B). A drop well above them is a real effect.
- **k curve.** The top 1, 2, 3, 5, 10, 20 and 50 neurons switched off together, and the same for random neurons.
  **k_mean** is the smallest k that cuts the log-odds by 3 or more. **k_flip** is the smallest k that turns at least
  half the right answers wrong. Small numbers mean the concept depends on a few neurons.
- **role.** Read from the neuron's weights (no extra runs): `say` (the answer is among the 10 words it pushes most),
  `attribute` (it pushes the answer clearly more than average), `late-other`, or `early` (in the first 20/36 of the
  layers, where the push on the output is indirect). `pushes` lists its top 30 words.
- **heads.** The 50 attention heads with the most attribution, each switched off, with their drops.
- **call.** `in`, `uncertain` or `out`: whether a neuron would still be in the top 30 on other hardware. On Qwen3-8B
  it is `uncertain` when its drop is within 0.13 of the 30th neuron's drop, the largest difference we see between
  two GPUs (99th percentile). The 30th neuron is always `uncertain`. Other models (Qwen3-0.6B included) have no
  calibrated band yet, so their calls say `uncalibrated` rather than guess.

METHOD.md has the full procedure and its controls. FORMAT.md has every field of a record.

## Limits

1. **A drop shows a neuron is needed, not what it means.** Neurons that back each other up can each show small drops.
   The k curve is there to catch that.
2. **Different questions give a somewhat different map.** The labeller writes new questions each time. In five
   independent runs for Bitcoin on Qwen3-0.6B, the marker neuron L23:2392 came 2nd, 2nd, 3rd, 2nd and 2nd. Use
   `--questions` to repeat a run on the same questions.
3. **Hardware and software change the small numbers.** The same run on another GPU, another torch build (even on
   the same CPU), or in float32 instead of bfloat16, moves drops by a few hundredths. That can swap neurons near the
   30th place, which is what the `uncertain` call is for. Every record says what it ran on.
4. **Two models from one family.** Other models with gated MLPs should work, but we haven't tested them.
5. **The labeller matters.** Every published map used `gpt-6-luna`. Another labeller writes different questions.

## Documents

- [SETUP.md](SETUP.md): install step by step, with checks (written so a coding agent can follow it).
- [METHOD.md](METHOD.md): the procedure, its controls and how `check` is calibrated.
- [FORMAT.md](FORMAT.md): the record format, and the published map's format that `lookup` reads.
- [CAPS.md](CAPS.md): every cap and threshold, and which ones you can change.
- [CALIBRATION.md](CALIBRATION.md): the runs behind the values `check` compares with, and the proof that this code gives
  the same numbers as the research code.
- [FINDINGS.md](FINDINGS.md): what we have found with it on Qwen3-8B, how sure we are, and what we got wrong.

## Cite

James Phoenix, *Model Mapper*, Zenodo, [doi:10.5281/zenodo.23459925](https://doi.org/10.5281/zenodo.23459925).
GitHub's "Cite this repository" button reads [CITATION.cff](CITATION.cff).

## Credit and licence

James Phoenix set the questions and the standards. Claude (Anthropic's Claude Code) wrote the code. The judge used for
the published steering results is TypeSafe's Jev; the labeller is OpenAI's `gpt-6-luna`. This is independent work,
not affiliated with Anthropic, Goodfire, OpenAI, TypeSafe or Alibaba (Qwen).

[MIT](LICENSE). Qwen model outputs are under the Qwen3 licence (Apache 2.0).

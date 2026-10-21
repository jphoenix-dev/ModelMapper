# Notes for coding agents

- **Setting up or checking an install:** follow [SETUP.md](SETUP.md), including its "Rules for an agent", and finish
  with the report from its step 8.
- **Layout:** one package, `modelmapper/`. `api.py` is the public API (`map_concept`, `map_batch`, `steer`, `lookup`,
  `check`), `cli.py` the command. The measuring code (`model.py`, `measure.py`, `mapping.py`, `questions.py`) is the
  code the published maps used, copied unchanged: change it only to change the method, and say so in METHOD.md.
  Labelling (LABELLING.md) is `sweep.py`, `pseudo.py`, `exam.py`, `gencheck.py`, `jobtype.py`, `scorecard.py` and
  `label.py` (the runner); they were ported from the v3 research run and are checked against its files (`-m repro`).
- **Tests:** `pytest -m "not model"` runs in seconds with no model; `pytest` also loads Qwen3-0.6B (about 10 minutes on
  a CPU). Model tests use the real Qwen3-0.6B against calibrated values. `modelmapper check` is the end-to-end test.
- Settings come from environment variables or `.env` in the working folder (`.env.example`).
- Never print or commit API keys. `.env`, `OAI.env` and `jevkey.env` are gitignored.

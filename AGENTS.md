# Notes for coding agents

- **Setting up or checking an install:** follow [SETUP.md](SETUP.md) step by step, including its "Rules for an agent"
  section, and finish with the report from its step 9.
- **Running experiments:** see README.md ("Main pipelines" and "The `modelmapper` command") and ARCHITECTURE.md.
  Run everything from the repo root through the CLI: `modelmapper <command>` (after `pip install -e .`) or
  `python -m modelmapper <command>`. Old `python src/...` commands (RESULTS.md, ledger rows) map to new ones in
  SCRIPTS.md; `modelmapper run src/<old path> ...` runs them as written. `scripts/check_imports.py` and
  `scripts/smoke_cli.py` check the package without running any experiment. Settings come from environment
  variables or `.env` (see `.env.example`); `modelmapper settings` shows what they resolve to.
- Never print or commit API keys. `.env`, `OAI.env` and `jevkey.env` are gitignored.

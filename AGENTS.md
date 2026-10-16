# Notes for coding agents

- **Setting up or checking an install:** follow [SETUP.md](SETUP.md) step by step, including its "Rules for an agent"
  section, and finish with the report from its step 9.
- **Running experiments:** see README.md ("Main pipelines") and ARCHITECTURE.md. Run scripts from the repo root as
  `python src/<script>.py`. Settings come from environment variables or `.env` (see `.env.example`).
- Never print or commit API keys. `.env`, `OAI.env` and `jevkey.env` are gitignored.

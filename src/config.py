"""Machine-specific settings, read from environment variables.

Each setting is looked up in the process environment first, then in an optional `.env` file at
the repo root (gitignored; see `.env.example`). Nothing here imports torch, so it is cheap to use
from the API clients too.

    ALREADY_CLEAN_MODELS  folder holding local model copies, one subfolder per model
                          (e.g. <dir>/Qwen3-8B/config.json). Default: <repo>/models.
                          If a model is not there, it is loaded from the Hugging Face hub.
    ALREADY_CLEAN_CORPUS  FineWeb parquet shard streamed by scan.py / build_atlas.py etc.
                          Default: <repo>/data/fineweb_000_00000.parquet.
    OPENAI_API_KEY        used by oai.py (Luna labeller); fallback file OAI.env.
    JEV_API_KEY           used by jev.py (TypeSafe's Jev judge); TYPESAFE_API_KEY also accepted;
                          fallback file jevkey.env.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOTENV = ROOT / ".env"
_dotenv: dict[str, str] | None = None


def _read_dotenv() -> dict[str, str]:
    global _dotenv
    if _dotenv is None:
        _dotenv = {}
        if DOTENV.exists():
            for line in DOTENV.read_text(encoding="utf-8-sig").splitlines():
                line = line.strip().removeprefix("export ").strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    _dotenv[k.strip()] = v.strip().strip("'\"").strip()
    return _dotenv


def env(name: str, default: str | None = None) -> str | None:
    """$name if set and non-empty, else the value in <repo>/.env, else default."""
    v = os.environ.get(name, "").strip()
    if v:
        return v
    return _read_dotenv().get(name) or default


MODELS_DIR = env("ALREADY_CLEAN_MODELS", str(ROOT / "models"))
CORPUS = env("ALREADY_CLEAN_CORPUS", str(ROOT / "data" / "fineweb_000_00000.parquet"))

"""Machine-specific settings, read from environment variables.

Each setting is looked up in the process environment first, then in an optional `.env` file at
the repo root (gitignored; see `.env.example`). Nothing here imports torch, so it is cheap to use
from the API clients too.

    MODEL_MAPPER_MODEL    Hugging Face id of the model under study. Default: Qwen/Qwen3-8B.
    MODEL_MAPPER_MODELS   folder holding local model copies, one subfolder per model
                          (e.g. <dir>/Qwen3-8B/config.json). Default: <repo>/models.
                          A model that is not there is loaded from the Hugging Face hub.
    MODEL_MAPPER_CORPUS   FineWeb parquet shard streamed by scan.py / build_atlas.py etc.
                          Default: <repo>/data/fineweb_000_00000.parquet.

API endpoints (labeller = oai.py, judge = jev.py) are configured in those modules; see .env.example.
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


DEFAULT_MODEL = env("MODEL_MAPPER_MODEL", "Qwen/Qwen3-8B")
MODELS_DIR = env("MODEL_MAPPER_MODELS", str(ROOT / "models"))
CORPUS = env("MODEL_MAPPER_CORPUS", str(ROOT / "data" / "fineweb_000_00000.parquet"))


def short_name(name: str) -> str:
    """'Qwen/Qwen3-8B' -> 'Qwen3-8B' (used in data/ and results/ file names)."""
    return name.split("/")[-1]


def model_path(name: str) -> str:
    """Local copy <MODELS_DIR>/<short name> if it exists, else the hub id (for from_pretrained)."""
    local = os.path.join(MODELS_DIR, short_name(name))
    return local if os.path.exists(os.path.join(local, "config.json")) else name

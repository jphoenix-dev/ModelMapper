"""Settings: read from the environment first, then from a `.env` file, then the defaults below.

The `.env` file is the one in the folder you run `modelmapper` from (MODEL_MAPPER_DOTENV names another). Nothing here
imports torch. `modelmapper settings` prints what each setting resolves to and where it came from.

    MODEL_MAPPER_MODEL    model under study (Hugging Face id)                      default Qwen/Qwen3-8B
    MODEL_MAPPER_MODELS   folder of local model copies (<dir>/Qwen3-8B/config.json) default: none (the Hugging Face cache)
    MODEL_MAPPER_DEVICE   cuda, cuda:1, mps or cpu                                 default CUDA, else MPS, else CPU
    MODEL_MAPPER_DTYPE    bfloat16, float16 or float32 (float32 is faster on CPU)   default bfloat16
    MODEL_MAPPER_CORPUS   web-text parquet, only for `modelmapper ranges`          default ./data/fineweb_000_00000.parquet
    MODEL_MAPPER_MAP_URL  the published map (for `modelmapper lookup`)              default the Qwen Neuron Map
    OPENAI_BASE_URL, OAI_MODEL, OPENAI_API_KEY, OAI_KEY_FILE      the labeller (modelmapper/labeller.py)
    JUDGE_BACKEND, JUDGE_BASE_URL, JUDGE_MODEL, JUDGE_API_KEY, JEV_API_KEY, JEV_KEY_FILE   the optional judge (judge.py)
Keys are read only by the client that needs them and are never printed.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

WORK_DIR = Path.cwd()
DOTENV = Path(os.environ.get("MODEL_MAPPER_DOTENV", "").strip() or WORK_DIR / ".env")
_dotenv: dict[str, str] | None = None


def _read_dotenv() -> dict[str, str]:
    global _dotenv
    if _dotenv is None:
        _dotenv = {}
        if DOTENV.exists():
            for line in DOTENV.read_text(encoding="utf-8-sig").splitlines():
                line = line.strip().removeprefix("export ").strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1); v = v.strip()
                    if v[:1] in ("'", '"') and v[0] in v[1:]:      # quoted: the value is what is inside the quotes
                        v = v[1:1 + v[1:].index(v[0])]
                    else:                                         # unquoted: a ' #' starts a comment
                        v = re.split(r"\s+#", v, maxsplit=1)[0].strip().strip("'\"").strip()
                    _dotenv[k.strip()] = v
    return _dotenv


def env(name: str, default: str | None = None) -> str | None:
    """$name if set and non-empty, else the value in .env, else default."""
    v = os.environ.get(name, "").strip()
    if v:
        return v
    return _read_dotenv().get(name) or default


def source(name: str) -> str:
    """Where a setting's value comes from: 'environment', '.env' or 'default'."""
    if os.environ.get(name, "").strip():
        return "environment"
    return ".env" if _read_dotenv().get(name) else "default"


DEFAULT_MODEL = env("MODEL_MAPPER_MODEL", "Qwen/Qwen3-8B")
MODELS_DIR = env("MODEL_MAPPER_MODELS", "")
CORPUS = env("MODEL_MAPPER_CORPUS", str(WORK_DIR / "data" / "fineweb_000_00000.parquet"))
MAP_URL = env("MODEL_MAPPER_MAP_URL", "https://eiffel-tower-qwen.vercel.app/map/data").rstrip("/")
CACHE_DIR = Path(env("MODEL_MAPPER_CACHE", str(Path.home() / ".cache" / "modelmapper")))


def short_name(name: str) -> str:
    """'Qwen/Qwen3-8B' -> 'Qwen3-8B' (used in file and folder names)."""
    return name.split("/")[-1]


def model_path(name: str) -> str:
    """Local copy <MODELS_DIR>/<short name> if it exists, else the hub id (for from_pretrained)."""
    if not MODELS_DIR:
        return name
    local = os.path.join(MODELS_DIR, short_name(name))
    return local if os.path.exists(os.path.join(local, "config.json")) else name


def describe() -> list[tuple[str, str, str]]:
    """(env var, value, source) for every setting except keys."""
    rows = [("MODEL_MAPPER_MODEL", DEFAULT_MODEL), ("MODEL_MAPPER_MODELS", MODELS_DIR or "(none: the Hugging Face cache)"),
            ("MODEL_MAPPER_CORPUS", CORPUS), ("MODEL_MAPPER_MAP_URL", MAP_URL), ("MODEL_MAPPER_CACHE", str(CACHE_DIR))]
    for var in ("MODEL_MAPPER_DEVICE", "MODEL_MAPPER_DTYPE", "OPENAI_BASE_URL", "OAI_MODEL", "JUDGE_BACKEND", "JUDGE_MODEL"):
        rows.append((var, env(var) or "(unset: default)"))
    return [(var, value, source(var)) for var, value in rows]

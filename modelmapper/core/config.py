"""Machine-specific settings (kept under its old name: 50+ modules import from here).

Since the 18 Oct 2026 refactor the values live in modelmapper.core.settings, the one module for every data path and
env var; this module re-exports the names it always had.

    MODEL_MAPPER_MODEL    Hugging Face id of the model under study. Default: Qwen/Qwen3-8B.
    MODEL_MAPPER_MODELS   folder holding local model copies, one subfolder per model
                          (e.g. <dir>/Qwen3-8B/config.json). Default: <repo>/models.
                          A model that is not there is loaded from the Hugging Face hub.
    MODEL_MAPPER_CORPUS   FineWeb parquet shard streamed by scan.py / build_atlas.py etc.
                          Default: <repo>/data/fineweb_000_00000.parquet.

API endpoints (labeller = oai.py, judge = jev.py) are configured in those modules; see .env.example.
"""
from __future__ import annotations

from modelmapper.core.settings import (CORPUS, DEFAULT_MODEL, DOTENV, MODELS_DIR, _read_dotenv, env,  # noqa: F401
                                       model_path, short_name)
from modelmapper.core.settings import REPO_ROOT as ROOT  # noqa: F401

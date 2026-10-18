"""Every machine setting and data path, in one place. `modelmapper settings` prints what they resolve to.

Each value is read from the process environment first, then from an optional `.env` file at the repo root (gitignored;
see `.env.example`), then falls back to the default below. Nothing here imports torch, so any module can use it.
The defaults are the research machine's paths, so the original runs reproduce unchanged; set the variables to move
them.

Paths and machine settings (resolved once, at import):
    MODEL_MAPPER_MODEL    model under study (Hugging Face id)                      Qwen/Qwen3-8B
    MODEL_MAPPER_MODELS   folder of local model copies (<dir>/Qwen3-8B/config.json) <repo>/models, else the hub
    MODEL_MAPPER_CORPUS   FineWeb parquet shard (atlas, scans, random snippets)    <repo>/data/fineweb_000_00000.parquet
    AC_DATA_DIR           research outputs of the loop / experiments / ledger     D:/already-clean-data
    LEDGER_DIR            the ledger (packs/, ledger.duckdb, export/)             D:/already-clean-data/ledger
    AC_REPO_RESULTS       results/ of the main checkout (earlier results)          C:/dev/already-clean/results
    AC_ATLAS_DOCS         the 8B atlas's document index (J-lens windows)          C:/dev/already-clean/data/atlas_Qwen3-8B_docs.npz
    AC_STATS              the 8B neuron ranges (scales steering)                   C:/dev/already-clean/data/stats_Qwen3-8B.pt
    MAP_WORKTREE          the map-page worktree (ledger importers read its specs)  C:/dev/already-clean-map
    MAP_JOB_NAMES         job-name outputs (ledger importer)                       D:/already-clean-data/job_names
Read by modelmapper.core.neurons and friends at import (listed here so they are all in one place):
    MODEL_MAPPER_DEVICE / MODEL_MAPPER_DTYPE / MODEL_MAPPER_ATLAS_BATCH / MODEL_MAPPER_GEN_BATCH (see README)
API endpoints: OPENAI_BASE_URL, OAI_MODEL, OPENAI_API_KEY (labeller, core/oai.py); JUDGE_BACKEND, JUDGE_BASE_URL,
JUDGE_MODEL, JUDGE_API_KEY, JUDGE_EXTRA_BODY, JEV_API_KEY / TYPESAFE_API_KEY (judge, core/jev.py). Keys are never printed.

Per-run knobs (KNOBS below) are still read where they always were, at the moment the experiment reads them (several are
set by a caller just before an import), so their behaviour is unchanged; KNOBS only lists them.

The ledger's own modules (ledger/core.py, ledger/importers/) are kept byte-identical because their bytes feed pack ids;
they read LEDGER_DIR / AC_DATA_DIR / AC_REPO_RESULTS / MAP_* straight from the environment, so modelmapper.ledger copies
the values resolved here into the environment before they load (see export_to_environ).
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOTENV = REPO_ROOT / ".env"
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


def source(name: str) -> str:
    """Where a setting's value comes from: 'environment', '.env' or 'default'."""
    if os.environ.get(name, "").strip():
        return "environment"
    return ".env" if _read_dotenv().get(name) else "default"


# name -> (env var, default). The defaults are the strings the scripts hard-coded before the refactor.
PATHS = {
    "DATA_DIR": ("AC_DATA_DIR", "D:/already-clean-data"),
    "LEDGER_DIR": ("LEDGER_DIR", "D:/already-clean-data/ledger"),
    "REPO_RESULTS": ("AC_REPO_RESULTS", "C:/dev/already-clean/results"),
    "ATLAS_DOCS": ("AC_ATLAS_DOCS", "C:/dev/already-clean/data/atlas_Qwen3-8B_docs.npz"),
    "STATS": ("AC_STATS", "C:/dev/already-clean/data/stats_Qwen3-8B.pt"),
    "MAP_WORKTREE": ("MAP_WORKTREE", "C:/dev/already-clean-map"),
    "MAP_JOB_NAMES": ("MAP_JOB_NAMES", "D:/already-clean-data/job_names"),
}

DEFAULT_MODEL = env("MODEL_MAPPER_MODEL", "Qwen/Qwen3-8B")
MODELS_DIR = env("MODEL_MAPPER_MODELS", str(REPO_ROOT / "models"))
CORPUS = env("MODEL_MAPPER_CORPUS", str(REPO_ROOT / "data" / "fineweb_000_00000.parquet"))
DATA_DIR = Path(env(*PATHS["DATA_DIR"]))
LEDGER_DIR = Path(env(*PATHS["LEDGER_DIR"]))
REPO_RESULTS = Path(env(*PATHS["REPO_RESULTS"]))
ATLAS_DOCS = Path(env(*PATHS["ATLAS_DOCS"]))
STATS = Path(env(*PATHS["STATS"]))
MAP_WORKTREE = Path(env(*PATHS["MAP_WORKTREE"]))
MAP_JOB_NAMES = Path(env(*PATHS["MAP_JOB_NAMES"]))


def short_name(name: str) -> str:
    """'Qwen/Qwen3-8B' -> 'Qwen3-8B' (used in data/ and results/ file names)."""
    return name.split("/")[-1]


def model_path(name: str) -> str:
    """Local copy <MODELS_DIR>/<short name> if it exists, else the hub id (for from_pretrained)."""
    local = os.path.join(MODELS_DIR, short_name(name))
    return local if os.path.exists(os.path.join(local, "config.json")) else name


def export_to_environ():
    """Put the resolved paths into os.environ (setdefault) for modules that read the environment directly."""
    for name, (var, _) in PATHS.items():
        os.environ.setdefault(var, str(globals()[name]).replace("\\", "/"))


# Per-run knobs: env var -> (where it is read, what it does). Read in place, not here (see the docstring).
KNOBS = {
    "RECALL_VARIANT": ("core/pipeline_fast, core/wave_free, core/markers, loop/map_loop, several experiments",
                       "recall-prompt format (V0, V3, ...); read when pipeline_fast is imported"),
    "CONCEPT_OUT_DIR": ("core/concept_pipeline, core/markers, loop/prepare", "where concept results / spec files go"),
    "CONCEPT_MIN_P": ("core/concept_pipeline", "recall threshold for a prompt to count"),
    "ANSWER_DISCOVERY": ("core/pipeline_fast, core/markers, several experiments", "let the model name its own answer form"),
    "MODEL": ("core/wave_free, loop/map_loop", "model id for those two scripts"),
    "KNOCK_BATCH": ("loop/map_loop", "single knockouts per forward pass (0 = one per pass, the validated path)"),
    "ENRICH_STAGES": ("loop/enrich", "comma list of enrichment stages to run (E0,E2,...); `modelmapper enrich --stages`"),
    "ENRICH_NAME": ("loop/enrich", "output folder name under <AC_DATA_DIR>/loop_v3/"),
    "ENRICH_CHEAP": ("loop/enrich", "cheap mode (see loop/enrich.py)"),
    "ENRICH_ALL": ("loop/enrich", "run every concept, not only the sample"),
    "AG_ROWS": ("experiments/attrib_graph", "rows per backward batch (32 on 48 GB cards)"),
    "JL_ROWS": ("experiments/jlens", "output dimensions per backward batch"),
    "JL_TOKENS": ("experiments/jlens", "tokens per window"),
    "JL_MAX_DIMS": ("experiments/jlens", "cap on output dimensions (tests only)"),
    "CHAIN_K": ("experiments/chain", "neurons per chain step"),
    "FANOUT_TAG": ("experiments/fanout", "output tag"),
    "NB": ("experiments/concept_atlas, experiments/fast_readout", "batch size"),
    "MULT": ("experiments/fast_readout", "activation multiplier"),
    "VF_TAG": ("experiments/concept_atlas", "run tag"), "VF_LIMIT": ("experiments/concept_atlas", "first N concepts"),
    "AUTO_DIRS": ("experiments/auto_steer_scale", "input folders"), "AUTO_OUT": ("experiments/auto_steer_scale", "output file"),
    "LABEL_MATCH_FILE": ("experiments/label_rule", "label-matched neuron file"), "ONLY": ("experiments/label_rule", "subset"),
    "OUT_NAME": ("experiments/label_rule", "output name"), "SKIP_R1": ("experiments/label_rule", "skip the R1 rule"),
    "REPLY_ONLY": ("experiments/steer_select", "steer the reply only"), "STEER_MODE": ("experiments/people_rules", "clamp / add"),
    "MAP_FINDINGS": ("ledger/importers/curated_claims", "the map page's findings file"),
    "DIST_STORE": ("dist", "run store: s3://bucket/prefix or a local folder"),
    "DIST_QUOTA / DIST_MAX_GB / DIST_MAX_CLASS_A / DIST_MAX_CLASS_B": ("dist/quota, dist/worker", "bucket caps"),
    "DIST_CLAIM_TTL_S / DIST_HEARTBEAT_S": ("dist/manifest, dist/worker", "task claim timing"),
    "DIST_DTYPE / DIST_TRUST_LOCAL / GIT_COMMIT": ("dist/runtime", "pod runtime"),
    "RUNPOD_API_KEY / RUNPOD_POD_ID / RUNPOD_GPU_TYPE": ("dist", "RunPod (keys never printed)"),
    "S3_ENDPOINT_URL / S3_ACCESS_KEY_ID / S3_SECRET_ACCESS_KEY / S3_REGION": ("dist/storage, dist/launcher", "bucket (keys never printed)"),
}


def describe() -> list[tuple[str, str, str, str]]:
    """(setting, env var, value, source) for every path / machine setting (no keys)."""
    rows = [("DEFAULT_MODEL", "MODEL_MAPPER_MODEL", DEFAULT_MODEL, source("MODEL_MAPPER_MODEL")),
            ("MODELS_DIR", "MODEL_MAPPER_MODELS", MODELS_DIR, source("MODEL_MAPPER_MODELS")),
            ("CORPUS", "MODEL_MAPPER_CORPUS", CORPUS, source("MODEL_MAPPER_CORPUS"))]
    for name, (var, _) in PATHS.items():
        rows.append((name, var, str(globals()[name]), source(var)))
    for var in ("MODEL_MAPPER_DEVICE", "MODEL_MAPPER_DTYPE", "MODEL_MAPPER_ATLAS_BATCH", "MODEL_MAPPER_GEN_BATCH",
                "OPENAI_BASE_URL", "OAI_MODEL", "JUDGE_BACKEND", "JUDGE_BASE_URL", "JUDGE_MODEL"):
        rows.append(("", var, env(var) or "(unset: module default)", source(var)))
    return rows

"""jlens:<start>-<end>:r<rows>: one shard of the research branch's J-lens estimate (src/loop_v2/jlens.py estimate),
run as a subprocess on the pinned model. The worker does not load the model for these runs (config kind "jlens"),
so the subprocess has the GPU to itself.

The pod needs: the pinned model as a local folder (downloaded or verified via runtime.model_dir, exposed as
<tmp>/models/<short name> for jlens's model_path check), the atlas docs file (store data/atlas_<model>_docs.npz, ~80 MB,
fetched once per pod), JL_ROWS. Output j_sums_<a>_<b>.pt (~4.8 GB) is uploaded in parts, with jlens_log.jsonl.
rows >= 8 needs a card with >= 40 GB (24.6 GB peak measured on the 5090; spilling past VRAM made it 4.6x slower).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
_cache = {}


def _docs(store, cfg):
    if "docs" not in _cache:
        p = Path(tempfile.gettempdir()) / "mm_atlas_docs.npz"
        if not p.exists() or p.stat().st_size == 0:
            store.get_file(cfg["jlens"]["docs_key"], p)
        _cache["docs"] = p
    return _cache["docs"]


def _model_root(cfg):
    if "models" not in _cache:
        from config import short_name
        from dist import runtime
        path, source, fetch = runtime.model_dir(cfg["model"], cfg["revision"])
        p = Path(path)
        if p.name == short_name(cfg["model"]) and (p / "config.json").exists():
            root = p.parent                                      # local copy already named <short name>
        else:
            root = Path(tempfile.mkdtemp(prefix="mm_models_"))
            (root / short_name(cfg["model"])).symlink_to(p, target_is_directory=True)
        _cache["models"] = (root, source, fetch)
    return _cache["models"]


def jlens_task(store, task_id, ctx, key):
    _, rng, rows = task_id.split(":")
    a, b = (int(x) for x in rng.split("-")); R = int(rows.lstrip("r"))
    total = torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0
    if R >= 8 and total < 40e9:
        raise RuntimeError(f"JL_ROWS={R} needs a >= 40 GB card (this one has {total / 2**30:.0f} GiB)")
    cfg = ctx.cfg
    t0 = time.time(); root, source, fetch = _model_root(cfg); t_model = time.time() - t0
    t0 = time.time(); docs = _docs(store, cfg); t_docs = time.time() - t0
    out = Path(tempfile.mkdtemp(prefix="mm_jlens_"))
    env = dict(os.environ, MODEL_MAPPER_MODELS=str(root), AC_DATA_DIR=str(out), AC_ATLAS_DOCS=str(docs), JL_ROWS=str(R),
               PYTHONIOENCODING="utf-8")
    cmd = [sys.executable, "-u", "src/loop_v2/jlens.py", "estimate", "--windows", str(cfg["jlens"]["windows"]),
           "--start", str(a), "--end", str(b)]
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(REPO), env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    t_run = time.time() - t0
    store.put(f"{key}/stdout.txt", (r.stdout[-20000:] + "\n--- stderr ---\n" + r.stderr[-20000:]).encode("utf-8"))
    if r.returncode != 0:
        raise RuntimeError(f"jlens.py exited {r.returncode}: {r.stderr[-1500:]}")
    shard = out / "jlens" / f"j_sums_{a}_{b}.pt"
    logf = out / "jlens" / "jlens_log.jsonl"
    t0 = time.time(); store.put_file(f"{key}/{shard.name}", shard); t_up = time.time() - t0
    log = []
    if logf.exists():
        store.put(f"{key}/jlens_log.jsonl", logf.read_bytes())
        log = [json.loads(x) for x in logf.read_text(encoding="utf-8").splitlines() if x.strip()]
    size = shard.stat().st_size
    shard.unlink(missing_ok=True)
    return {"tokens": 0, "windows": b - a, "jl_rows": R, "gpu_total_gib": round(total / 2**30, 1),
            "model_s": round(t_model, 1), "docs_s": round(t_docs, 1), "estimate_s": round(t_run, 1),
            "upload_s": round(t_up, 1), "shard_gb": round(size / 1e9, 2), "upload_MBps": round(size / 1e6 / t_up, 1) if t_up else None,
            "s_per_window_wall": round(t_run / max(b - a, 1), 1), "jlens_log": log[-(b - a) - 2:]}

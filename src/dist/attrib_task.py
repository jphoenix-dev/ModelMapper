"""attrib:<step>[:<limit>]: the research branch's attribution graphs (src/loop_v2/attrib_graph.py; method ATTRIB_GRAPHS.md)
run as a subprocess on the pinned model. step = graphs | ablate; limit = --limit N for graphs (0 = all).

Inputs come from the store under data/attrib_graphs/ (pilot_prompts.json; for ablate also <slug>/graph.json and
<slug>/groups.json) into <tmp>/attrib_graphs/, AC_DATA_DIR=<tmp>. Every file the step writes or changes there is
uploaded to results/attrib/<step>[/<limit>]/<relative path>, so the layout under it is attrib_graphs' own
(<slug>/graph.json, <slug>/ablations.json). AG_ROWS from the run config (32 on 48 GB cards; 16 if it runs out).
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from dist.jlens_task import REPO, _model_root


def _digest(p):
    return hashlib.sha1(p.read_bytes()).hexdigest()


def attrib_task(store, task_id, ctx, key):
    parts = task_id.split(":")
    step = parts[1]; limit = int(parts[2]) if len(parts) > 2 else 0
    cfg = ctx.cfg
    t0 = time.time(); root, source, fetch = _model_root(cfg); t_model = time.time() - t0
    tmp = Path(tempfile.mkdtemp(prefix="mm_attrib_")); data = tmp / "attrib_graphs"
    for k in store.list("data/attrib_graphs/"):
        store.get_file(k, data / k[len("data/attrib_graphs/"):])
    before = {p: _digest(p) for p in data.rglob("*") if p.is_file()}
    env = dict(os.environ, MODEL_MAPPER_MODELS=str(root), AC_DATA_DIR=str(tmp), AG_ROWS=str(cfg["attrib"]["rows"]),
               PYTHONIOENCODING="utf-8")
    cmd = [sys.executable, "-u", "src/loop_v2/attrib_graph.py", step] + (["--limit", str(limit)] if limit else [])
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(REPO), env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    t_run = time.time() - t0
    store.put(f"{key}/stdout.txt", (r.stdout[-50000:] + "\n--- stderr ---\n" + r.stderr[-20000:]).encode("utf-8"))
    if r.returncode != 0:
        raise RuntimeError(f"attrib_graph.py {step} exited {r.returncode}: {r.stderr[-1500:]}")
    written = [p for p in data.rglob("*") if p.is_file() and before.get(p) != _digest(p)]
    for p in written:
        store.put_file(f"{key}/{p.relative_to(data).as_posix()}", p)
    lines = [x for x in r.stdout.splitlines() if "| pass" in x or "pass " in x]
    return {"tokens": 0, "step": step, "limit": limit, "ag_rows": cfg["attrib"]["rows"], "model_s": round(t_model, 1),
            "run_s": round(t_run, 1), "files_written": len(written), "summary_lines": lines[-40:]}

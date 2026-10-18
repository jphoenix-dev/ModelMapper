"""Where each pod's time and money went: one row per boot step, per task, and teardown.

Reads logs/launcher.json (the launcher's ledger: requested / created / first seen running / terminate sent / gone, and
each pod's $/h) and logs/<pod>/boot.json, task_*.json, exit.json (the worker's epochs and timings). Times between a
desktop event and a pod event mix two clocks (both NTP-synced, so expect about a second of error).

Steps (seconds):
  create call          launcher request -> RunPod answers
  schedule + image     pod created -> container starts (host picked, image pulled, container started)
  install boto3        container start -> boto3 installed (boot script)
  fetch code           -> worker code downloaded from the bucket
  install packages     -> requirements-dist.txt installed
  python + imports     -> worker.py running (torch / transformers import, environment probe)
  model download       pinned weights from the Hugging Face hub (GB, MB/s)
  model to GPU         load onto the GPU
  until first task     model ready -> first task claimed (store listing, quota check)
  tasks                each: compute, input download, result upload, tokens/s, $ at the pod's rate
  teardown             worker exit -> pod gone (self-termination or the launcher's backstop)
Cost per step = pod $/h x step seconds / 3600. Billed totals come from `launcher.py billing` (RunPod's own numbers).

Usage: python src/dist/timeline.py <store> [--json out.json]
"""
import argparse
import json
import sys
from pathlib import Path

from modelmapper.dist.storage import open_store  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("store"); ap.add_argument("--json")
    a = ap.parse_args()
    st = open_store(a.store)
    led = (st.get_json("logs/launcher.json") or {"pods": {}})["pods"]
    workers = sorted({k.split("/")[1] for k in st.list("logs/") if k.count("/") >= 2})
    out = {}
    for w in workers:
        boot = st.get_json(f"logs/{w}/boot.json") or {}
        ex = st.get_json(f"logs/{w}/exit.json") or {}
        tasks = [st.get_json(k) for k in sorted(st.list(f"logs/{w}/task_"))]
        e = led.get(w, {})
        rate = e.get("rate") or 0.0
        s = boot.get("steps", {})
        m = boot.get("model", {})
        rows = []

        def step(name, t0, t1, note=""):
            if t0 is not None and t1 is not None:
                rows.append({"step": name, "seconds": round(t1 - t0, 1), "usd": round(rate * (t1 - t0) / 3600, 5), "note": note})
        step("create call", e.get("requested"), e.get("created"))
        step("schedule + image", e.get("created"), s.get("container_start"), "two clocks")
        step("install boto3", s.get("container_start"), s.get("boto_done"))
        step("fetch code", s.get("boto_done"), s.get("code_done"))
        step("install packages", s.get("code_done"), s.get("pip_done"))
        step("python + imports", s.get("pip_done"), s.get("python_start"))
        if "download_s" in m:
            t = s.get("python_start")
            step("model download", t, t + m["download_s"], f"{m.get('download_gb')} GB at {m.get('download_MBps')} MB/s")
            step("model to GPU", t + m["download_s"], s.get("model_ready"))
        else:
            step("model load (local)", s.get("python_start"), s.get("model_ready"), m.get("weights_source", ""))
        step("until first task", s.get("model_ready"), s.get("first_task_start"))
        for t in tasks:
            if t.get("skipped"):
                continue
            note = (f"compute {t.get('compute_s')} s, in {t.get('io_get_mb')} MB / {t.get('io_get_s')} s, "
                    f"out {t.get('io_put_mb')} MB / {t.get('io_put_s')} s, {t.get('tokens_per_s')} tok/s, "
                    f"GPU peak {t.get('gpu_peak_allocated_gb')} GB")
            step(f"task {t['task']}", t.get("start_epoch"), t.get("end_epoch"), note)
        step("teardown", ex.get("epoch"), e.get("gone"), "two clocks")
        life = (e.get("gone") or ex.get("epoch") or 0) - (e.get("requested") or e.get("created") or s.get("container_start") or 0)
        work = sum(t.get("seconds", 0) for t in tasks if not t.get("skipped"))
        out[w] = {"gpu": (boot.get("env") or {}).get("gpu"), "rate_usd_h": rate, "lifetime_s": round(life, 1),
                  "lifetime_usd": round(rate * life / 3600, 4), "task_s": round(work, 1),
                  "task_share_of_lifetime": round(work / life, 3) if life > 0 else None, "steps": rows}
        print(f"\n== {w}  {out[w]['gpu']}  ${rate}/h  lifetime {life/60:.1f} min  ~${out[w]['lifetime_usd']}  "
              f"(tasks {work/60:.1f} min = {100 * (out[w]['task_share_of_lifetime'] or 0):.0f}%)")
        for r in rows:
            print(f"  {r['step']:<34} {r['seconds']:>8.1f} s  ${r['usd']:<9.5f} {r['note']}")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

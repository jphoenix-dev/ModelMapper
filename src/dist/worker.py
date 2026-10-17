"""Worker loop (the container entrypoint): run pending tasks until none remain, then terminate this pod.

Loop: list the _DONE markers once, work out which manifest tasks are pending and ready (dependencies done), prefer
ones nobody has claimed recently, pick one at random, run it, repeat. There are no locks: two workers may now and
then run the same task, which is harmless (same outputs, _DONE written last). Claims only steer workers apart.

When nothing is pending, the worker logs its summary and, on RunPod, TERMINATES its own pod (not stop): first
`runpodctl pod delete $RUNPOD_POD_ID`, else DELETE /v1/pods/$RUNPOD_POD_ID with the pod-scoped $RUNPOD_API_KEY.
The launcher also terminates any pod that has finished, as a backstop.

Logs (in the store): logs/<worker>/boot.json (start, model download/load times, first task start),
logs/<worker>/task_<n>.json (one per task), logs/<worker>/heartbeat.json, logs/<worker>/exit.json.

Usage: python src/dist/worker.py --store s3://bucket/run1   (or DIST_STORE; --once runs one task; --no-terminate)
"""
import argparse
import os
import random
import socket
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
T_START = time.time()

from dist.storage import open_store  # noqa: E402
from dist import quota  # noqa: E402
from dist.manifest import state  # noqa: E402
from dist.tasks import Ctx, run_task  # noqa: E402

IDLE_POLL, IDLE_MAX = 30, 15 * 60


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def terminate_self(log):
    pod = os.environ.get("RUNPOD_POD_ID")
    if not pod:
        log("not on RunPod (no RUNPOD_POD_ID): exiting without terminating anything")
        return
    # What works on RunPod's images (seen 2026-10-17): `runpodctl remove pod` (older runpodctl; `pod delete` is unknown
    # there). The REST DELETE with the pod-scoped $RUNPOD_API_KEY is refused (403). Never raise: the launcher's
    # backstop terminates any pod whose worker wrote exit.json.
    for cmd in (["runpodctl", "remove", "pod", pod], ["runpodctl", "pod", "delete", pod]):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            log(f"{' '.join(cmd[:3])} -> {r.returncode} {r.stdout.strip()[:200]} {r.stderr.strip()[:200]}")
            if r.returncode == 0 and "removed" in (r.stdout + r.stderr).lower():
                return
        except Exception as e:  # noqa: BLE001
            log(f"{' '.join(cmd[:3])} failed: {e!r}")
    try:
        from dist import runpod_api
        runpod_api.terminate(pod, key=os.environ.get("RUNPOD_API_KEY"))
        log("terminated via REST")
    except Exception as e:  # noqa: BLE001
        log(f"REST terminate failed ({e}); the launcher will terminate this pod")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store")
    ap.add_argument("--once", action="store_true", help="run one task and exit")
    ap.add_argument("--task", help="run exactly this task id (no loop)")
    ap.add_argument("--no-terminate", action="store_true")
    a = ap.parse_args()
    store = open_store(a.store)
    wid = os.environ.get("RUNPOD_POD_ID") or f"{socket.gethostname()}-{os.getpid()}"
    lp = f"logs/{wid}"

    def log(msg):
        print(f"[{now()}] {msg}", flush=True)

    store.put_json(f"{lp}/started.json", {"epoch": time.time(), "boot_steps": {k: v for k, v in os.environ.items() if k.startswith("T_")}})
    beat = {"tasks_run": 0, "phase": "loading model"}

    def heartbeat():
        while True:
            time.sleep(120)
            try:
                store.put_json(f"{lp}/heartbeat.json", {"epoch": time.time(), "t": now(), "uptime_s": round(time.time() - T_START),
                                                        **{k: v for k, v in beat.items() if k != "claim"}})
                if beat.get("claim"):                    # keep this worker's claim fresh while it works
                    store.put_json(beat["claim"], {"worker": wid, "t": time.time()})
            except Exception as e:  # noqa: BLE001 - a missed heartbeat must never kill the worker
                print(f"heartbeat failed: {e!r}", flush=True)
    threading.Thread(target=heartbeat, daemon=True).start()
    ctx = Ctx(store, wid)
    if os.environ.get("RUNPOD_POD_ID") and not ctx.env["device"].startswith("cuda"):
        raise SystemExit(f"refusing to run on {ctx.env['device']} on a GPU pod: torch cannot see the GPU ({ctx.env})")
    # epoch seconds of each boot step: container start, boto3 installed, code fetched, packages installed (boot script),
    # python started, model downloaded + on the GPU, first task started
    steps = {k[2:].lower(): float(v) for k, v in os.environ.items() if k.startswith("T_") and k[2:].replace("_", "").isalpha()
             and v.replace(".", "", 1).isdigit()}
    steps["python_start"] = T_START
    boot = {"worker": wid, "process_start": T_START, "process_start_iso": now(), "env": ctx.env, "steps": steps}
    t0 = time.time(); _ = ctx.model                      # download (if needed) + load the pinned model
    steps["model_ready"] = time.time()
    boot.update(model=ctx.model_meta, model_ready_s=round(time.time() - T_START, 1), model_load_s=round(time.time() - t0, 1))
    store.put_json(f"{lp}/boot.json", boot)
    log(f"model ready after {boot['model_ready_s']} s ({ctx.model_meta['weights_source']})")

    tasks = store.get_json("manifest.json")["tasks"]
    n_run, idle_since, summary = 0, None, []
    while True:
        if a.task:
            pending, ready, claims = [a.task], [a.task], {}
        else:
            pending, ready, claims = state(store, tasks)
        if not pending:
            log("no pending tasks"); break
        free = [t for t in ready if t not in claims or claims[t]["worker"] == wid]
        pick = random.choice(free) if free else None
        if pick is None:
            idle_since = idle_since or time.time()
            if time.time() - idle_since > IDLE_MAX:
                log(f"idle {IDLE_MAX}s with {len(pending)} pending, all claimed or waiting: exiting"); break
            time.sleep(IDLE_POLL); continue
        idle_since = None
        if os.environ.get("DIST_QUOTA", "1") != "0":
            c = ctx.model.config
            need = quota.task_output_bytes(ctx.cfg, pick, c.num_hidden_layers, c.intermediate_size, c.num_attention_heads,
                                           c.hidden_size, store.get_json(f"promptsets/{pick.split(':')[-1]}.json") if pick.startswith("ablate:") else None)
            try:
                q = quota.check(store, extra_bytes=need, extra_a=200, extra_b=200)
            except quota.QuotaExceeded as e:
                log(f"QUOTA: {e} - not starting {pick}; stopping this worker")
                summary.append({"quota_stop": str(e)}); break
            log(f"quota ok: {q['gb_held']} GB held, {q['class_a_month']:,} A / {q['class_b_month']:,} B this month")
        claim_key = "claims/" + pick.replace(":", "/")
        store.put_json(claim_key, {"worker": wid, "t": time.time()})
        beat["claim"] = claim_key
        if "first_task_start" not in boot:
            boot["first_task_start_s"] = round(time.time() - T_START, 1); boot["first_task_start"] = now()
            steps["first_task_start"] = time.time()
            store.put_json(f"{lp}/boot.json", boot)
        log(f"start {pick} ({len(pending)} pending, {len(ready)} ready)")
        beat.update(phase=f"running {pick}")
        try:
            meta = run_task(store, pick, ctx)
        except Exception:
            err = traceback.format_exc(); log(f"FAILED {pick}\n{err}")
            store.put_json(f"{lp}/error_{n_run}.json", {"task": pick, "error": err, "t": now()})
            beat.pop("claim", None); store.delete(claim_key)
            if a.task or a.once:
                raise
            time.sleep(5); continue
        beat.pop("claim", None); store.delete(claim_key)
        keep = ("seconds", "tokens", "tokens_per_s", "start_epoch", "end_epoch", "compute_s", "io_get_s", "io_get_mb",
                "io_put_s", "io_put_mb", "gpu_peak_allocated_gb", "gpu_peak_reserved_gb")
        row = {"task": pick, "skipped": meta is None, **({k: meta[k] for k in keep if k in meta} if meta else {})}
        summary.append(row); store.put_json(f"{lp}/task_{n_run:04d}.json", row)
        quota.save_ops(store, wid)
        beat.update(tasks_run=n_run + 1, phase="between tasks")
        log(f"done {pick}: {row}")
        n_run += 1
        if a.once or a.task:
            break
    store.put_json(f"{lp}/exit.json", {"t": now(), "epoch": time.time(), "uptime_s": round(time.time() - T_START, 1),
                                       "tasks": summary, "io_total": store.io, "requests": store.ops})
    quota.save_ops(store, wid)
    if not a.no_terminate:
        terminate_self(log)


if __name__ == "__main__":
    main()

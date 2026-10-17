"""Launcher (runs on James's desktop): keep N Community pods (on-demand; --interruptible for spot) working until every task is done.

Every --interval seconds:
  1. list this run's pods (name prefix) and update the spend ledger: each pod's costPerHr x the time since it was
     created (until it is seen gone or terminated). The ledger is saved locally and to logs/launcher.json;
  2. terminate pods that need it: stopped ones (EXITED: interrupted, or a worker that exited without terminating),
     and running ones whose worker wrote logs/<pod>/exit.json (backstop for self-termination);
  3. if the spend so far plus one more interval of the live pods reaches --max-spend: terminate everything and stop;
  4. start replacements until live pods = min(--pods, --max-pods, pending tasks);
  5. when no task is pending: terminate what is left and wait until the pod list shows zero pods.
Ctrl-C terminates every pod of the run before exiting. Whatever happens, it ends by listing pods and printing the count.

Usage:
  python src/dist/launcher.py run   --store s3://bucket/run1 --image <registry>/model-mapper:<tag> --gpu "NVIDIA GeForce RTX 4090" \
                                    --pods 3 --max-pods 3 --max-spend 5
  python src/dist/launcher.py status    --store ... [--name ...]
  python src/dist/launcher.py terminate --store ... [--name ...]       (terminate every pod of the run, confirm zero)
  python src/dist/launcher.py billing   --store ...                      (actual charges from /v1/billing/pods)
Credentials: RUNPOD_API_KEY, S3_* (env or .env); passed to the pods as env vars, never baked into the image.
"""
import argparse
import base64
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import env  # noqa: E402
from dist import quota  # noqa: E402
from dist import runpod_api as rp  # noqa: E402
from dist.storage import open_store  # noqa: E402
from dist.manifest import state  # noqa: E402

POD_ENV_KEYS = ["S3_ENDPOINT_URL", "S3_ACCESS_KEY_ID", "S3_SECRET_ACCESS_KEY", "S3_REGION", "HF_TOKEN"]
STOCK_IMAGE = "runpod/pytorch:1.4.0-cu1281-torch2121-ubuntu2204"   # RunPod's own (often cached on hosts); torch 2.12.1
CODE_PATHS = ["requirements-dist.txt", "src/config.py", "src/neurons.py", "src/dist",
              # loop v2 (map tasks) and its import chain; no keys are read at import, and Atlas/stats load lazily
              "src/loop_v2", "src/atlas.py", "src/concept_pipeline.py", "src/jev.py", "src/oai.py", "src/pipeline_fast.py",
              "src/steer_test.py"]
REPO = Path(__file__).resolve().parents[2]

# Boot script for the stock image (no custom image or registry). It travels base64-encoded in $BOOT_B64 and the start
# command only decodes and runs it, so it works whether RunPod passes the command in exec form or as one string.
# Every step is timestamped and the whole log is PUT to the bucket after each step ($BOOTLOG_URL: a presigned URL,
# no keys needed), so a failed boot is visible. On any failure the pod terminates itself (REST, pod-scoped key).
BOOT = r"""#!/bin/bash
set -o pipefail
LOG=/tmp/boot.log
step() { echo "$(date +%s.%N) $*" >> $LOG; curl -s -m 30 -X PUT -T $LOG "$BOOTLOG_URL" > /dev/null || true; }
fail() {
  step "FAILED: $1"; tail -c 4000 /tmp/step.out >> $LOG 2>/dev/null
  step "self-terminate via runpodctl: $(runpodctl remove pod $RUNPOD_POD_ID 2>&1 | tail -c 300)"
  step "self-terminate via REST: HTTP $(curl -s -m 30 -o /tmp/del.out -w '%{http_code}' -X DELETE -A model-mapper-pod/1.0 \
       -H "Authorization: Bearer $RUNPOD_API_KEY" "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID") $(head -c 200 /tmp/del.out)"
  sleep infinity
}
export T_CONTAINER_START=$(date +%s.%N)
step "container start: pod $RUNPOD_POD_ID; $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>&1); $(python --version 2>&1)"
pip install -q --no-cache-dir boto3==1.43.100 > /tmp/step.out 2>&1 || fail "pip install boto3"
export T_BOTO_DONE=$(date +%s.%N); step "boto3 installed"
python - > /tmp/step.out 2>&1 <<'PY' || fail "fetch code"
import io, os, tarfile, boto3
bucket, _, prefix = os.environ["DIST_STORE"][5:].partition("/")
key = (prefix.strip("/") + "/" if prefix.strip("/") else "") + os.environ["DIST_CODE"]
s3 = boto3.client("s3", endpoint_url=os.environ.get("S3_ENDPOINT_URL"), region_name=os.environ.get("S3_REGION", "auto"),
                  aws_access_key_id=os.environ["S3_ACCESS_KEY_ID"], aws_secret_access_key=os.environ["S3_SECRET_ACCESS_KEY"])
tarfile.open(fileobj=io.BytesIO(s3.get_object(Bucket=bucket, Key=key)["Body"].read())).extractall("/app")
PY
export T_CODE_DONE=$(date +%s.%N); step "code fetched"
cd /app
pip install -q --no-cache-dir -r requirements-dist.txt > /tmp/step.out 2>&1 || fail "pip install requirements"
export T_PIP_DONE=$(date +%s.%N)
step "packages installed: $(python -c 'import torch, transformers; print("torch", torch.__version__, "cuda", torch.version.cuda, "transformers", transformers.__version__)' 2>&1)"
step "gpu check: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-unset} NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES-unset} devices: $(ls /dev/nvidia* 2>&1 | tr '\n' ' ')"
python - > /tmp/step.out 2>&1 <<'PY' || fail "torch cannot use the GPU"
import torch
print("is_available", torch.cuda.is_available(), "count", torch.cuda.device_count())
x = torch.ones(1024, 1024, device="cuda", dtype=torch.bfloat16)
print("matmul ok on", torch.cuda.get_device_name(0), float((x @ x).sum()))
PY
step "gpu ok: $(tail -c 300 /tmp/step.out | tr '\n' ' ')"
python src/dist/worker.py > /tmp/step.out 2>&1; rc=$?
step "worker exited rc=$rc"; tail -c 4000 /tmp/step.out >> $LOG; step "worker log tail saved"
[ $rc -eq 0 ] || fail "worker"
sleep infinity
"""
START_CMD = ["bash", "-c", "echo${IFS}$BOOT_B64|base64${IFS}-d>/tmp/boot.sh;bash${IFS}/tmp/boot.sh"]   # no spaces or quotes


def bootlog_url(store, pod_name, hours=4):
    """Presigned PUT URL for logs/bootlog-<pod name>.txt (the pod has no id yet when it is created)."""
    key = store.prefix + f"logs/bootlog-{pod_name}.txt"
    return store.s3.generate_presigned_url("put_object", Params={"Bucket": store.bucket, "Key": key}, ExpiresIn=hours * 3600)


def upload_code(store) -> tuple[str, str]:
    """git archive of the committed worker code -> code/<commit>.tar.gz in the store. Refuses uncommitted changes, so
    the commit recorded with every result is exactly the code that ran."""
    dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--", *CODE_PATHS], capture_output=True, text=True).stdout
    if dirty.strip():
        raise SystemExit(f"commit the worker code first; uncommitted changes:\n{dirty}")
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    tar = subprocess.run(["git", "-C", str(REPO), "archive", "--format=tar.gz", commit, *CODE_PATHS], capture_output=True).stdout
    key = f"code/{commit}.tar.gz"
    if not store.exists(key):
        store.put(key, tar)
    return key, commit


def run_name(store_url):
    return "mm-" + store_url.rstrip("/").split("/")[-1][:40]


class Ledger:
    def __init__(self, path):
        self.path = Path(path)
        self.d = json.loads(self.path.read_text()) if self.path.exists() else {"pods": {}}

    def see(self, pods, now):
        live = {p["id"] for p in pods}
        for p in pods:
            e = self.d["pods"].setdefault(p["id"], {"created": now, "rate": p.get("costPerHr") or 0, "gone": None})
            e["rate"] = p.get("costPerHr") or e["rate"]; e["status"] = p.get("desiredStatus")
            if p.get("desiredStatus") == "RUNNING" and "first_seen_running" not in e:
                e["first_seen_running"] = now
            for f in ("lastStartedAt", "machineId", "gpu", "memoryInGb", "vcpuCount"):
                if p.get(f) is not None:
                    e[f] = p[f]
            if p.get("desiredStatus") == "EXITED" and not e.get("stopped"):
                e["stopped"] = now                 # a stopped pod with volumeInGb 0 bills nothing
        for pid, e in self.d["pods"].items():
            if pid not in live and e["gone"] is None:
                e["gone"] = now

    def spend(self, now):
        return sum(e["rate"] * ((e.get("stopped") or e["gone"] or now) - e["created"]) / 3600 for e in self.d["pods"].values())

    def save(self, store=None):
        self.path.write_text(json.dumps(self.d, indent=1))
        if store is not None:
            store.put_json("logs/launcher.json", self.d)


def terminate_all(name, why, led=None):
    pods = rp.list_pods(name)
    for p in pods:
        print(f"terminating {p['id']} ({p.get('desiredStatus')}): {why}", flush=True)
        rp.terminate(p["id"])
        if led is not None and p["id"] in led.d["pods"]:
            led.d["pods"][p["id"]].setdefault("terminate_sent", time.time())
    for _ in range(30):
        left = rp.list_pods(name)
        if not left:
            break
        time.sleep(10)
    left = rp.list_pods(name)
    print(f"pods remaining for {name}: {len(left)} {[p['id'] for p in left]}", flush=True)
    all_pods = rp.list_pods()
    print(f"pods on the whole account: {len(all_pods)}", flush=True)
    return len(left)


def drain(store, tasks, pending, dest):
    """Copy finished results that no pending task still needs to `dest`, then delete them from the bucket (keeping
    the small meta.json and _DONE markers there, so the run's state is unchanged). Frees bucket storage as we go."""
    import fnmatch
    pend = set(pending)
    needed = {t["id"] for t in tasks if any(fnmatch.fnmatch(t["id"], pat) for p in tasks if p["id"] in pend for pat in p.get("after", []))}
    moved = 0
    for t in tasks:
        tid = t["id"]
        if tid in pend or tid in needed:
            continue
        pre = "results/" + tid.replace(":", "/") + "/"
        for k, size in store.list_sizes(pre).items():
            if k.endswith(("/_DONE", "/meta.json")):
                continue
            out = Path(dest) / k
            out.parent.mkdir(parents=True, exist_ok=True)
            data = store.get(k)
            if data is None or len(data) != size:
                continue
            out.write_bytes(data)
            if out.stat().st_size == size:           # delete from the bucket only once the local copy is complete
                store.delete(k); moved += size
    if moved:
        print(f"drained {moved/1e9:.3f} GB to {dest}", flush=True)
    return moved


def cmd_run(a, store, name):
    if a.pods > a.max_pods:
        raise SystemExit(f"--pods {a.pods} > --max-pods {a.max_pods}")
    pod_env = {k: env(k) for k in POD_ENV_KEYS if env(k)}
    pod_env["DIST_STORE"] = a.store
    start_cmd = None
    if not a.custom_image:
        pod_env["DIST_CODE"], pod_env["GIT_COMMIT"] = upload_code(store)
        pod_env["BOOT_B64"] = base64.b64encode(BOOT.encode()).decode()
        start_cmd = START_CMD
        print(f"stock image {a.image}; code {pod_env['DIST_CODE']}", flush=True)
    boot_failed = set()
    no_cap_since, gave_up = None, False
    me = f"launcher-{os.getpid()}-{int(time.time())}"
    led = Ledger(a.ledger or f"launcher_{name}.json")
    bad_hosts = set(led.d.get("bad_hosts", [])) | set(filter(None, (a.bad_hosts or "").split(",")))
    tasks = store.get_json("manifest.json")["tasks"]
    launched = 0
    try:
        while True:
            now = time.time()
            pods = rp.list_pods(name)
            led.see(pods, now)
            pending, ready, claims = state(store, tasks)
            for pid, e in led.d["pods"].items():   # gone or stopped before the worker ever got going = a failed boot
                if (e.get("gone") or e.get("stopped") or e.get("boot_timeout") or e.get("boot_failed")) \
                        and not store.exists(f"logs/{pid}/boot.json") and not e.get("bad_host") \
                        and pid not in boot_failed:
                    boot_failed.add(pid); print(f"pod {pid} ended before its worker booted ({len(boot_failed)} so far)", flush=True)
            if len(boot_failed) >= a.max_boot_failures:
                terminate_all(name, f"{len(boot_failed)} pods failed to boot: stopping (see the pod logs in the RunPod console)")
                break
            for p in pods:
                e = led.d["pods"].get(p["id"], {})
                if p.get("machineId") in bad_hosts and p.get("desiredStatus") == "RUNNING":
                    print(f"terminating {p['id']}: landed on known-bad machine {p['machineId']}", flush=True)
                    rp.terminate(p["id"]); e.setdefault("terminate_sent", time.time()); e["bad_host"] = True
                    continue
                if p.get("desiredStatus") == "RUNNING" and not a.custom_image:
                    bl = (store.get(f"logs/bootlog-{p.get('name')}.txt") or b"").decode(errors="replace")
                    if "FAILED" in bl:
                        why = bl.split("FAILED:", 1)[1].splitlines()[0].strip()
                        print(f"terminating {p['id']}: boot failed ({why}) on machine {p.get('machineId')}", flush=True)
                        rp.terminate(p["id"]); e.setdefault("terminate_sent", time.time()); e["boot_failed"] = why
                        if "GPU" in why and p.get("machineId"):
                            bad_hosts.add(p["machineId"]); led.d.setdefault("bad_hosts", []).append(p["machineId"])
                        continue
                age = (now - e.get("created", now)) / 60
                if (p.get("desiredStatus") == "RUNNING" and age > a.boot_timeout_min
                        and not store.exists(f"logs/{p['id']}/started.json")):
                    print(f"terminating {p['id']}: no worker after {age:.1f} min (image pull or boot stuck)", flush=True)
                    rp.terminate(p["id"]); e.setdefault("terminate_sent", time.time()); e["boot_timeout"] = True
                    continue
                hb = store.get_json(f"logs/{p['id']}/heartbeat.json") or store.get_json(f"logs/{p['id']}/started.json")
                if p.get("desiredStatus") == "RUNNING" and hb and time.time() - hb.get("epoch", time.time()) > a.stale_min * 60 \
                        and not store.exists(f"logs/{p['id']}/exit.json"):
                    print(f"terminating {p['id']}: no heartbeat for {(time.time() - hb['epoch']) / 60:.1f} min", flush=True)
                    rp.terminate(p["id"]); e.setdefault("terminate_sent", time.time()); e["stale"] = True
                    continue
                finished = store.exists(f"logs/{p['id']}/exit.json")
                if p.get("desiredStatus") == "EXITED" or (finished and p.get("desiredStatus") == "RUNNING"):
                    print(f"terminating {p['id']} ({p.get('desiredStatus')}, worker exited: {finished})", flush=True)
                    rp.terminate(p["id"])
                    led.d["pods"].setdefault(p["id"], {}).setdefault("terminate_sent", time.time())
            live = [p for p in pods if p.get("desiredStatus") == "RUNNING" and not store.exists(f"logs/{p['id']}/exit.json")]
            if a.drain_to:
                drain(store, tasks, pending, a.drain_to)
            try:
                q = quota.check(store); quota.save_ops(store, me)
            except quota.QuotaExceeded as e:
                terminate_all(name, f"R2 free-tier cap: {e}")
                break
            try:
                bal = rp.balance()
            except Exception as e:  # noqa: BLE001 - the balance is a backstop; never let it crash the launcher
                bal = None; print(f"balance check failed: {e!r}", flush=True)
            if bal is not None:
                led.d.setdefault("balance_log", []).append([round(now), bal["clientBalance"], bal["currentSpendPerHr"]])
                if a.min_balance is not None and bal["clientBalance"] < a.min_balance:
                    terminate_all(name, f"account balance ${bal['clientBalance']:.2f} below the floor ${a.min_balance}", led)
                    break
            spent = led.spend(now)
            rate = sum(p.get("costPerHr") or 0 for p in live)
            print(f"[{time.strftime('%H:%M:%S')}] live {len(live)}, pending {len(pending)} (ready {len(ready)}, claimed {len(claims)}), "
                  f"spend ~${spent:.3f}, ${rate:.3f}/h, balance {('$%.2f' % bal['clientBalance']) if bal else '?'}, bucket {q['gb_held']} GB, {q['class_a_month']:,} A / {q['class_b_month']:,} B this month", flush=True)
            led.save(store)
            if spent + rate * a.interval / 3600 >= a.max_spend:
                terminate_all(name, f"spend cap: ~${spent:.2f} + next interval reaches ${a.max_spend}")
                break
            if not pending:
                print("all tasks done", flush=True)
                for k in store.list("claims/"):          # leftover soft claims (harmless, but tidy up)
                    store.delete(k)
                if a.drain_to:
                    drain(store, tasks, [], a.drain_to)
                break
            want = min(a.pods, a.max_pods, len(pending))
            for _ in range(want - len(live)):
                if launched >= a.max_launches:
                    print(f"--max-launches {a.max_launches} reached; not starting more", flush=True)
                    break
                guess = max([e["rate"] for e in led.d["pods"].values()] + [a.rate_guess])
                if spent + (rate + guess) * a.interval / 3600 >= a.max_spend:
                    print(f"not starting a pod: ~${spent:.2f} + one more interval would reach ${a.max_spend}", flush=True)
                    break
                rate += guess
                t_req = time.time()
                pod_name = f"{name}-{launched:03d}-{int(t_req) % 100000}"
                env_i = dict(pod_env)
                if not a.custom_image:
                    env_i["BOOTLOG_URL"] = bootlog_url(store, pod_name)
                try:
                    p = rp.create_pod(pod_name, a.image, [g.strip() for g in a.gpu.split(",")], env_i, a.disk_gb, a.registry_auth, start_cmd,
                                      a.interruptible, a.cuda_versions.split(","))
                except rp.NoCapacity:
                    no_cap_since = no_cap_since or time.time()
                    waited = (time.time() - no_cap_since) / 60
                    print(f"no {a.gpu} available on Community (CUDA {a.cuda_versions}); waited {waited:.1f} of "
                          f"{a.max_wait_min} min, retrying next interval", flush=True)
                    if waited >= a.max_wait_min:
                        gave_up = True
                    break
                no_cap_since = None
                launched += 1
                led.d["pods"][p["id"]] = {"requested": t_req, "created": time.time(), "rate": p.get("costPerHr") or 0,
                                          "gone": None, "gpu_type": (p.get("gpu") or {}).get("id") or a.gpu, "name": p.get("name")}
                print(f"started {p['id']} on {(p.get('gpu') or {}).get('displayName') or a.gpu} at ${p.get('costPerHr')}/h", flush=True)
                led.see(rp.list_pods(name), time.time())
            if gave_up and not live:
                print(f"giving up: no {a.gpu} capacity for {a.max_wait_min} min", flush=True)
                break
            time.sleep(a.interval)
    except KeyboardInterrupt:
        print("interrupted", flush=True)
    finally:
        led.see(rp.list_pods(name), time.time()); led.save(store)
        n = terminate_all(name, "launcher finished", led)
        led.see(rp.list_pods(name), time.time()); led.save(store)
        print(f"estimated spend ~${led.spend(time.time()):.3f}; pods left: {n}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "status", "terminate", "billing"])
    ap.add_argument("--store", default=env("DIST_STORE"), required=env("DIST_STORE") is None)
    ap.add_argument("--name", help="pod name prefix (default mm-<last part of the store URL>)")
    ap.add_argument("--image", default=STOCK_IMAGE)
    ap.add_argument("--custom-image", action="store_true", help="--image has the code baked in (Dockerfile); no boot script")
    ap.add_argument("--max-boot-failures", type=int, default=2)
    ap.add_argument("--gpu", default="NVIDIA GeForce RTX 3090,NVIDIA GeForce RTX 4090,NVIDIA GeForce RTX 5090",
                    help="comma-separated GPU type ids in order of preference (cheapest per token first); RunPod takes the "
                         "first one it has. Every result records the GPU it ran on. 24 GB or more: observe peaks at 20.5 GB")
    ap.add_argument("--pods", type=int, default=1); ap.add_argument("--max-pods", type=int, default=1)
    ap.add_argument("--max-spend", type=float, default=1.0, help="USD; the launcher stops everything before passing it")
    ap.add_argument("--max-launches", type=int, default=20, help="pods started in total, replacements included")
    ap.add_argument("--interval", type=int, default=120)
    ap.add_argument("--rate-guess", type=float, default=0.5, help="USD/h assumed for a pod not started yet (cap check)")
    ap.add_argument("--disk-gb", type=int, default=40)
    ap.add_argument("--interruptible", action="store_true", help="spot pods (RunPod said they are no longer offered, 2026-10-17)")
    ap.add_argument("--cuda-versions", default="12.8,12.9,13.0", help="host CUDA versions allowed (driver must run the image)")
    ap.add_argument("--max-wait-min", type=float, default=20, help="give up if no machine is available for this long")
    ap.add_argument("--min-balance", type=float, help="USD: stop everything if the RunPod account balance falls below this")
    ap.add_argument("--bad-hosts", help="comma-separated RunPod machine ids to release immediately (also learned from GPU failures)")
    ap.add_argument("--stale-min", type=float, default=10, help="terminate a pod whose worker heartbeat is older than this")
    ap.add_argument("--boot-timeout-min", type=float, default=15,
                    help="terminate a pod whose worker has not started (image pulled, packages installed) by then")
    ap.add_argument("--registry-auth", help="RunPod containerRegistryAuthId for a private image")
    ap.add_argument("--ledger")
    ap.add_argument("--drain-to", help="local folder: finished results are copied here and deleted from the bucket")
    a = ap.parse_args()
    store = open_store(a.store); name = a.name or run_name(a.store)
    if a.cmd == "run":
        cmd_run(a, store, name)
    elif a.cmd == "status":
        b = rp.balance(); print(f"account balance ${b['clientBalance']:.2f}, spending ${b['currentSpendPerHr']:.3f}/h")
        pending, ready, claims = state(store, store.get_json("manifest.json")["tasks"])
        for p in rp.list_pods(name):
            print(p["id"], p.get("desiredStatus"), p.get("costPerHr"), p.get("lastStartedAt"))
        print(f"pending {len(pending)}, ready {len(ready)}, claimed {sorted(claims)}")
    elif a.cmd == "terminate":
        terminate_all(name, "terminate command")
    elif a.cmd == "billing":
        led = store.get_json("logs/launcher.json") or {"pods": {}}
        total = 0.0
        for pid in led["pods"]:
            rows = rp._req("GET", f"/billing/pods?podId={pid}&bucketSize=day") or []
            amt = sum(r.get("amount", 0) for r in rows); total += amt
            print(pid, f"${amt:.4f}")
        print(f"billed total ${total:.4f} (estimate in ledger: see logs/launcher.json)")


if __name__ == "__main__":
    main()

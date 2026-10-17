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
STOCK_IMAGE = "pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime"    # public; same torch / CUDA as the local .venv
CODE_PATHS = ["requirements-dist.txt", "src/config.py", "src/neurons.py", "src/dist"]
REPO = Path(__file__).resolve().parents[2]

# Pod start command for the stock image: no custom image or registry. Fetch the committed code from the bucket,
# install the pinned packages, run the worker. If any step fails, the pod terminates itself (REST, pod-scoped key)
# so a broken boot never sits there billing.
BOOT = r"""
set -e
export T_CONTAINER_START=$(date +%s.%N)
fail() { python -c "import os,urllib.request as u; u.urlopen(u.Request('https://rest.runpod.io/v1/pods/'+os.environ['RUNPOD_POD_ID'], method='DELETE', headers={'Authorization':'Bearer '+os.environ['RUNPOD_API_KEY']}))"; }
trap fail ERR
pip install -q --no-cache-dir boto3==1.43.100
export T_BOTO_DONE=$(date +%s.%N)
python - <<'PY'
import io, os, tarfile, boto3
bucket, _, prefix = os.environ["DIST_STORE"][5:].partition("/")
key = (prefix.strip("/") + "/" if prefix.strip("/") else "") + os.environ["DIST_CODE"]
s3 = boto3.client("s3", endpoint_url=os.environ.get("S3_ENDPOINT_URL"), region_name=os.environ.get("S3_REGION", "auto"),
                  aws_access_key_id=os.environ["S3_ACCESS_KEY_ID"], aws_secret_access_key=os.environ["S3_SECRET_ACCESS_KEY"])
tarfile.open(fileobj=io.BytesIO(s3.get_object(Bucket=bucket, Key=key)["Body"].read())).extractall("/app")
PY
export T_CODE_DONE=$(date +%s.%N)
cd /app
pip install -q --no-cache-dir -r requirements-dist.txt
export T_PIP_DONE=$(date +%s.%N)
python src/dist/worker.py
"""


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
        start_cmd = ["bash", "-c", BOOT]
        print(f"stock image {a.image}; code {pod_env['DIST_CODE']}", flush=True)
    boot_failed = set()
    no_cap_since, gave_up = None, False
    me = f"launcher-{os.getpid()}-{int(time.time())}"
    led = Ledger(a.ledger or f"launcher_{name}.json")
    tasks = store.get_json("manifest.json")["tasks"]
    launched = 0
    try:
        while True:
            now = time.time()
            pods = rp.list_pods(name)
            led.see(pods, now)
            pending, ready, claims = state(store, tasks)
            for pid, e in led.d["pods"].items():   # gone or stopped before the worker ever got going = a failed boot
                if (e["gone"] or e.get("stopped")) and not store.exists(f"logs/{pid}/boot.json") and pid not in boot_failed:
                    boot_failed.add(pid); print(f"pod {pid} ended before its worker booted ({len(boot_failed)} so far)", flush=True)
            if len(boot_failed) >= a.max_boot_failures:
                terminate_all(name, f"{len(boot_failed)} pods failed to boot: stopping (see the pod logs in the RunPod console)")
                break
            for p in pods:
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
            spent = led.spend(now)
            rate = sum(p.get("costPerHr") or 0 for p in live)
            print(f"[{time.strftime('%H:%M:%S')}] live {len(live)}, pending {len(pending)} (ready {len(ready)}, claimed {len(claims)}), "
                  f"spend ~${spent:.3f}, ${rate:.3f}/h, bucket {q['gb_held']} GB, {q['class_a_month']:,} A / {q['class_b_month']:,} B this month", flush=True)
            led.save(store)
            if spent + rate * a.interval / 3600 >= a.max_spend:
                terminate_all(name, f"spend cap: ~${spent:.2f} + next interval reaches ${a.max_spend}")
                break
            if not pending:
                print("all tasks done", flush=True)
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
                try:
                    p = rp.create_pod(f"{name}-{launched:03d}", a.image, a.gpu, pod_env, a.disk_gb, a.registry_auth, start_cmd,
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
                                          "gone": None, "gpu_type": a.gpu, "name": p.get("name")}
                print(f"started {p['id']} on {a.gpu} at ${p.get('costPerHr')}/h", flush=True)
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
    ap.add_argument("--gpu", default="NVIDIA GeForce RTX 4090")
    ap.add_argument("--pods", type=int, default=1); ap.add_argument("--max-pods", type=int, default=1)
    ap.add_argument("--max-spend", type=float, default=1.0, help="USD; the launcher stops everything before passing it")
    ap.add_argument("--max-launches", type=int, default=20, help="pods started in total, replacements included")
    ap.add_argument("--interval", type=int, default=120)
    ap.add_argument("--rate-guess", type=float, default=0.5, help="USD/h assumed for a pod not started yet (cap check)")
    ap.add_argument("--disk-gb", type=int, default=40)
    ap.add_argument("--interruptible", action="store_true", help="spot pods (RunPod said they are no longer offered, 2026-10-17)")
    ap.add_argument("--cuda-versions", default="12.8,12.9,13.0", help="host CUDA versions allowed (driver must run the image)")
    ap.add_argument("--max-wait-min", type=float, default=20, help="give up if no machine is available for this long")
    ap.add_argument("--registry-auth", help="RunPod containerRegistryAuthId for a private image")
    ap.add_argument("--ledger")
    ap.add_argument("--drain-to", help="local folder: finished results are copied here and deleted from the bucket")
    a = ap.parse_args()
    store = open_store(a.store); name = a.name or run_name(a.store)
    if a.cmd == "run":
        cmd_run(a, store, name)
    elif a.cmd == "status":
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

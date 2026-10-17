"""Launcher (runs on James's desktop): keep N interruptible Community pods working until every task is done.

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
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import env  # noqa: E402
from dist import runpod_api as rp  # noqa: E402
from dist.storage import open_store  # noqa: E402
from dist.manifest import state  # noqa: E402

POD_ENV_KEYS = ["S3_ENDPOINT_URL", "S3_ACCESS_KEY_ID", "S3_SECRET_ACCESS_KEY", "S3_REGION", "HF_TOKEN"]


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


def terminate_all(name, why):
    pods = rp.list_pods(name)
    for p in pods:
        print(f"terminating {p['id']} ({p.get('desiredStatus')}): {why}", flush=True)
        rp.terminate(p["id"])
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


def cmd_run(a, store, name):
    if a.pods > a.max_pods:
        raise SystemExit(f"--pods {a.pods} > --max-pods {a.max_pods}")
    pod_env = {k: env(k) for k in POD_ENV_KEYS if env(k)}
    pod_env["DIST_STORE"] = a.store
    led = Ledger(a.ledger or f"launcher_{name}.json")
    tasks = store.get_json("manifest.json")["tasks"]
    launched = 0
    try:
        while True:
            now = time.time()
            pods = rp.list_pods(name)
            led.see(pods, now)
            pending, ready, claims = state(store, tasks)
            for p in pods:
                finished = store.exists(f"logs/{p['id']}/exit.json")
                if p.get("desiredStatus") == "EXITED" or (finished and p.get("desiredStatus") == "RUNNING"):
                    print(f"terminating {p['id']} ({p.get('desiredStatus')}, worker exited: {finished})", flush=True)
                    rp.terminate(p["id"])
            live = [p for p in pods if p.get("desiredStatus") == "RUNNING" and not store.exists(f"logs/{p['id']}/exit.json")]
            spent = led.spend(now)
            rate = sum(p.get("costPerHr") or 0 for p in live)
            print(f"[{time.strftime('%H:%M:%S')}] live {len(live)}, pending {len(pending)} (ready {len(ready)}, claimed {len(claims)}), "
                  f"spend ~${spent:.3f}, ${rate:.3f}/h", flush=True)
            led.save(store)
            if spent + rate * a.interval / 3600 >= a.max_spend:
                terminate_all(name, f"spend cap: ~${spent:.2f} + next interval reaches ${a.max_spend}")
                break
            if not pending:
                print("all tasks done", flush=True)
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
                p = rp.create_pod(f"{name}-{launched:03d}", a.image, a.gpu, pod_env, a.disk_gb, a.registry_auth)
                launched += 1
                print(f"started {p['id']} on {a.gpu} at ${p.get('costPerHr')}/h", flush=True)
                led.see(rp.list_pods(name), time.time())
            time.sleep(a.interval)
    except KeyboardInterrupt:
        print("interrupted", flush=True)
    finally:
        led.see(rp.list_pods(name), time.time()); led.save(store)
        n = terminate_all(name, "launcher finished")
        led.see(rp.list_pods(name), time.time()); led.save(store)
        print(f"estimated spend ~${led.spend(time.time()):.3f}; pods left: {n}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "status", "terminate", "billing"])
    ap.add_argument("--store", default=env("DIST_STORE"), required=env("DIST_STORE") is None)
    ap.add_argument("--name", help="pod name prefix (default mm-<last part of the store URL>)")
    ap.add_argument("--image"); ap.add_argument("--gpu", default="NVIDIA GeForce RTX 4090")
    ap.add_argument("--pods", type=int, default=1); ap.add_argument("--max-pods", type=int, default=1)
    ap.add_argument("--max-spend", type=float, default=1.0, help="USD; the launcher stops everything before passing it")
    ap.add_argument("--max-launches", type=int, default=20, help="pods started in total, replacements included")
    ap.add_argument("--interval", type=int, default=120)
    ap.add_argument("--rate-guess", type=float, default=0.5, help="USD/h assumed for a pod not started yet (cap check)")
    ap.add_argument("--disk-gb", type=int, default=40)
    ap.add_argument("--registry-auth", help="RunPod containerRegistryAuthId for a private image")
    ap.add_argument("--ledger")
    a = ap.parse_args()
    store = open_store(a.store); name = a.name or run_name(a.store)
    if a.cmd == "run":
        if not a.image:
            raise SystemExit("--image is required")
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

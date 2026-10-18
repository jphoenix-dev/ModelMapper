"""Stay inside the R2 free tier automatically (checked 2026-10-17: https://developers.cloudflare.com/r2/pricing/).

Free each month: 10 GB-month of storage, 1M Class A requests (PutObject, ListObjects, ...), 10M Class B requests
(GetObject, HeadObject). DeleteObject is free. Two different kinds of limit:
  - storage is an average over the month, so what matters is how much sits in the bucket. We cap the bytes held at
    any moment (DIST_MAX_GB, default 9, i.e. below 10 even if it stayed that full all month), and the launcher's
    --drain-to copies finished results to a local folder and deletes them from the bucket as the run goes;
  - requests add up over the calendar month and deleting refunds nothing. Every process counts its own requests
    (storage.Store.ops) and saves them to usage/<YYYY-MM>/<process>.json; the caps apply to the month's total
    (DIST_MAX_CLASS_A default 900k, DIST_MAX_CLASS_B default 9M).
Assumes this bucket is the account's only R2 use (the free tier is per account).
Workers check before each task (stop and terminate their pod if it would pass a cap); the launcher checks every
interval (terminate everything and stop).
"""
import time

from config import env

MAX_GB = float(env("DIST_MAX_GB", "9"))
MAX_A = int(env("DIST_MAX_CLASS_A", "900000"))
MAX_B = int(env("DIST_MAX_CLASS_B", "9000000"))


class QuotaExceeded(RuntimeError):
    pass


def month():
    return time.strftime("%Y-%m", time.gmtime())


def save_ops(store, who):
    """Record this process's request counts for the month (one small PutObject), at the bucket root."""
    ops = dict(store.ops or {"A": 0, "B": 0})
    ops["A"] += 1                                    # this write itself
    tag = (getattr(store, "prefix", "") or "").strip("/").replace("/", "_")
    store.bucket_root().put_json(f"usage/{month()}/{tag + '__' if tag else ''}{who}.json", {"t": time.time(), **ops})


def usage(store):
    """(bytes held in the whole bucket, Class A this month, Class B this month): every run in the bucket counts."""
    root = store.bucket_root()
    sizes = root.list_sizes("")
    a = b = 0
    for k in sizes:
        if k.startswith(f"usage/{month()}/"):
            u = root.get_json(k) or {}
            a += u.get("A", 0); b += u.get("B", 0)
    if root is not store:                            # the root view counted its own requests into a copy: fold them back
        store.ops = root.ops
    return sum(sizes.values()), a, b


def check(store, extra_bytes=0, extra_a=0, extra_b=0):
    """Raise QuotaExceeded if the bucket would pass a cap after adding extra_*. Returns the usage."""
    held, a, b = usage(store)
    if held + extra_bytes > MAX_GB * 1e9:
        raise QuotaExceeded(f"storage: {held/1e9:.2f} GB held + {extra_bytes/1e9:.2f} GB > cap {MAX_GB} GB "
                            "(drain finished results with launcher --drain-to)")
    if a + extra_a > MAX_A:
        raise QuotaExceeded(f"Class A requests this month: {a:,} + {extra_a:,} > cap {MAX_A:,}")
    if b + extra_b > MAX_B:
        raise QuotaExceeded(f"Class B requests this month: {b:,} + {extra_b:,} > cap {MAX_B:,}")
    return {"gb_held": round(held / 1e9, 3), "class_a_month": a, "class_b_month": b,
            "caps": {"gb": MAX_GB, "class_a": MAX_A, "class_b": MAX_B}}


def task_output_bytes(cfg, task_id, n_layers=36, width=12288, n_heads=32, hidden=4096, promptset=None):
    """Rough upper bound on what one task writes (for the pre-task storage check)."""
    kind = task_id.split(":")[0]
    if kind == "links":
        return int(cfg.get("links", {}).get("batch_size", 20) * 1e6)     # ~0.2 MB per concept (arrays + links)
    if kind == "map":
        return int(cfg.get("map", {}).get("batch_size", 40) * 10e6)     # up to ~8 MB per concept (records + arrays)
    K = cfg["observe"]["k"]
    if kind in ("observe", "merge"):
        return n_layers * width * K * 2 * 11 + n_layers * width * 40 + n_layers * n_heads * K * 10
    if kind == "ablate" and promptset:
        a, b = (int(x) for x in task_id.split(":")[2].split("-"))
        n = b - a + 1
        rows = n * (promptset["n_pos"] + promptset["n_neg"] + promptset["n_ctrl"]) * len(promptset.get("modes", [0]))
        return rows * (hidden * 2 + 400) + n * 400
    return 1_000_000

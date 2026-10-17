"""Task ids, result keys and pending / ready state. No torch import, so the launcher stays light (and off the GPU).

A task is pending while results/<task>/_DONE does not exist. It is ready when every manifest task matching each of
its "after" patterns (fnmatch, e.g. "observe:*") is done. Claims (claims/<task>) are soft: they steer workers apart
and expire CLAIM_TTL after their last refresh (the worker's heartbeat refreshes its claim every 2 minutes), so a
dead worker's task is picked up again within minutes.
"""
import fnmatch
import time

CLAIM_TTL = 6 * 60


def task_key(task_id: str) -> str:
    return "results/" + task_id.replace(":", "/")


def done_ids(store) -> set[str]:
    return {k[len("results/"):-len("/_DONE")].replace("/", ":") for k in store.list("results/") if k.endswith("/_DONE")}


def state(store, tasks):
    """(pending ids, ready ids, fresh claims {task: claim}) from one listing of results/ and claims/."""
    done = done_ids(store)
    ids = [t["id"] for t in tasks]
    pending = [t for t in tasks if t["id"] not in done]

    def dep_ok(pat):
        hits = [i for i in ids if fnmatch.fnmatch(i, pat)]
        return bool(hits) and all(i in done for i in hits)
    ready = [t["id"] for t in pending if all(dep_ok(p) for p in t.get("after", []))]
    claims = {}
    for k in store.list("claims/"):
        c = store.get_json(k)
        if c and time.time() - c["t"] < CLAIM_TTL:
            claims[k[len("claims/"):].replace("/", ":")] = c
    return [t["id"] for t in pending], ready, claims

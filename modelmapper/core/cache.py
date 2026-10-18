"""Append-only, thread-safe result cache on disk, so long API runs can be resumed.

Every finished result is appended as one JSON line ({"k": key, "v": value}) the moment it arrives.
Re-running the same command reloads the file and skips anything already done; anything that failed
(never written) is simply attempted again. Keys are plain strings.

    c = Cache("results/census.cache.jsonl")
    if "luna|L3_N7" not in c: c.put("luna|L3_N7", {...})
"""
import json
import threading
from pathlib import Path


class Cache:
    def __init__(self, path):
        self.path = Path(path)
        self.lock = threading.Lock()
        self.data = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                try:
                    rec = json.loads(line)
                    self.data[rec["k"]] = rec["v"]
                except (json.JSONDecodeError, KeyError):
                    pass  # a line half-written during a crash: ignore it, the call will be redone
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "a", encoding="utf-8")

    def __contains__(self, k):
        return k in self.data

    def get(self, k, default=None):
        return self.data.get(k, default)

    def put(self, k, v):
        with self.lock:
            self.data[k] = v
            self._f.write(json.dumps({"k": k, "v": v}, ensure_ascii=False) + "\n")
            self._f.flush()

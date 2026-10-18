"""Links between a concept's key neurons (method and pass rule: LINKS.md), for the neuron map's graph view.

Reads the links run on run 1's top 30 when it exists (D:/already-clean-data/loop_v2/links_run1/), and otherwise the
earlier indirect-path study (results/indirect_path.jsonl in the main checkout: 555 concepts, top 5), computing the
same per-pair numbers from its activation records. Also usable as a reference implementation of the pass rule:

    python -I -X utf8 explainer/gen/map/links.py     # prints a summary of the indirect-path links
"""
from __future__ import annotations

import json
import os
from pathlib import Path

INDIRECT = Path(os.environ.get("MAP_INDIRECT", "C:/dev/already-clean/results/indirect_path.jsonl"))
RUN1 = Path(os.environ.get("MAP_LINKS", "D:/already-clean-data/loop_v2/links_run1"))
MIN_DROP, NULL_X, AGREE = 0.05, 3.0, 2 / 3


def layer(u):
    return int(u[1:].split(":")[0])


def pair_stats(rows, i, j, n_units):
    """rows: per prompt, unit_acts[R][n_units] with R = 1 + n_units (each unit zeroed) + n_units (its null zeroed)."""
    d, nl = [], []
    for r in rows:
        a = r["unit_acts"]; base = a[0][j]
        if abs(base) < 1e-3:
            continue
        s = 1.0 if base > 0 else -1.0
        d.append((base - a[1 + i][j]) * s / abs(base))
        nl.append((base - a[1 + n_units + i][j]) * s / abs(base))
    if not d:
        return None
    md, mn = sum(d) / len(d), sum(nl) / len(nl)
    agree = sum((x > 0) == (md > 0) for x in d) / len(d)
    passes = abs(md) >= MIN_DROP and abs(md) >= NULL_X * abs(mn) and agree >= AGREE and len(d) >= 3
    return {"drop": round(md, 4), "null": round(mn, 4), "n": len(d), "agree": round(agree, 3), "passes": passes}


def from_records(path):
    """Concept -> list of links from per-concept activation records (indirect_path.jsonl or links_run1 records)."""
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            units = rec["units"]; k = len(units); links = []
            for i in range(k):
                for j in range(k):
                    if layer(units[i]) < layer(units[j]):
                        st = pair_stats(rec["prompts"], i, j, k)
                        if st:
                            links.append({"from": units[i], "to": units[j], **st})
            out[rec["concept"]] = links
    return out


def load():
    """Prefer the run-1 links file (already computed by the run, per LINKS.md); fall back to the indirect-path records."""
    f = RUN1 / "links.jsonl"
    if f.exists():
        out = {}
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                out.setdefault(r["concept"], []).append(r)
        return out, "links_run1"
    if INDIRECT.exists():
        return from_records(INDIRECT), "indirect_path"
    return {}, None


if __name__ == "__main__":
    L, src = load()
    allp = [x for v in L.values() for x in v]
    ok = [x for x in allp if x["passes"]]
    print(f"source {src}: {len(L)} concepts, {len(allp)} ordered pairs, {len(ok)} pass "
          f"({sum(x['drop'] > 0 for x in ok)} support, {sum(x['drop'] < 0 for x in ok)} suppress)")

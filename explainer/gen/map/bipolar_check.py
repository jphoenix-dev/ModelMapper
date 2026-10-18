"""Exploratory check: do some neurons do one job when they fire positively and a different one when negatively?

    python -I -X utf8 explainer/gen/map/bipolar_check.py

Uses run 1's concept maps (each key neuron's sign at the answer position, and the drop when it is switched off) and
the both-ends weight readout (data/push). A neuron is 'used both ways' if it matters (drop >= 0.5 log-odds) for at
least one concept while firing positively and at least one while firing negatively. For those, we check whether
each end's pushed words name its own concepts: a concept 'matches' an end if a word of its name or answer (4+
letters) appears among that end's top-10 tokens.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

DATA = Path(__file__).resolve().parents[1].parent / "site" / "map" / "data"
MIN_DROP = 0.5


def words(s):
    return {w for w in re.findall(r"[a-z]{4,}", (s or "").lower())}


def main():
    cj = json.loads((DATA / "concepts.json").read_text(encoding="utf-8"))
    I = {k: i for i, k in enumerate(cj["cols"])}
    uses = defaultdict(lambda: {1: [], -1: []})
    shard_cache = {}
    for r in cj["rows"]:
        if r[I["status"]] != "mapped" or "0" not in r[I["sources"]]:
            continue
        if "shard" in I:
            k = r[I["shard"]]; shard_cache.setdefault(k, json.loads((DATA / f"cs/{k}.json").read_text(encoding="utf-8")))
            d = shard_cache[k][r[I["slug"]]]
        else:
            d = json.loads((DATA / f"c/{r[I['slug']]}.json").read_text(encoding="utf-8"))
        m = next((m for m in d["maps"] if m["source"] == "run1" and m.get("top")), None)
        if m is None:
            continue
        for t in m["top"]:
            n, drop, act = t[0], t[1] or 0, t[3]
            if act is None or drop < MIN_DROP:
                continue
            uses[n][1 if act >= 0 else -1].append((r[I["name"]], d.get("answer"), drop))
    both = {n: u for n, u in uses.items() if u[1] and u[-1]}
    one = {n: u for n, u in uses.items() if bool(u[1]) != bool(u[-1])}
    push = {}
    for L in range(36):
        p = DATA / "push" / f"L{L}.json"
        if p.exists():
            for i, v in json.loads(p.read_text(encoding="utf-8")).items():
                push[f"L{L}:{i}"] = v
    print(f"neurons with a strong use: {len(uses)}; used one way only: {len(one)}; used both ways: {len(both)} "
          f"({100 * len(both) / max(1, len(uses)):.1f}%)")
    pos_share = sum(1 for u in one.values() if u[1]) / max(1, len(one))
    print(f"one-way neurons that fire positively for their concepts: {100 * pos_share:.0f}%")

    def match(cs, end):
        ws = set()
        for t in end:
            ws |= words(t)
        return [c for c in cs if (words(c[0]) | words(c[1])) & ws]
    two_clear = []
    for n, u in both.items():
        if n not in push:
            continue
        mp, mn = match(u[1], push[n][0]), match(u[-1], push[n][1])
        if mp and mn:
            two_clear.append((n, mp, mn, push[n]))
    print(f"used both ways AND each end's words name its own concepts: {len(two_clear)}")
    for n, mp, mn, pw in sorted(two_clear, key=lambda x: -max(c[2] for c in x[1] + x[2]))[:15]:
        print(f"\n{n}\n  + fires positively for {[c[0] for c in mp][:4]}; pushes {pw[0][:8]}\n"
              f"  - fires negatively for {[c[0] for c in mn][:4]}; pushes {pw[1][:8]}")
    # same check for one-way neurons, as a baseline for how often an end names its concepts at all
    hit = sum(1 for n, u in one.items() if n in push and match(u[1] or u[-1], push[n][0 if u[1] else 1]))
    print(f"\nbaseline: one-way neurons whose used end names its concepts: {hit} of {len(one)}")


if __name__ == "__main__":
    main()

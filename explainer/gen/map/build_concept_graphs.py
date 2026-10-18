"""Figure 2 of the neuron map: a graph of each concept's 30 key neurons, grouped into labelled boxes in rows, with the
measured links between the groups. Built only from what the map already holds; nothing new is measured here.

    python -I -X utf8 explainer/gen/map/build_concept_graphs.py        # after build_neuronmap.py

Rows (from the layer and the weights, not from an AI):
  answer  layers 20-35, the neuron's push raises the answer's word for this concept (run 1 roles 'say'/'attribute')
  late    layers 20-35, any other job
  early   layers 0-19
Groups inside a row, by what we know about each neuron, strongest evidence first: its checked convenience name
(JOB_NAMES.md; neurons with the same name share a box), a tested century job, a tested two-ended scale, shared answer
machinery; otherwise 'pushes the answer' (answer row), 'already points to the answer' (an early neuron with the answer
among the 10 words its weights push most) or 'job not yet clear' (split into two layer bands).
Links between groups: the passing links (LINKS.md) from a neuron in one group to a neuron in another, summed.
Writes data/cgraph.json: {"rows": [...], "c": {slug: [groups, links]}} with
  group = [row, kind, label, [positions in the concept's 'top' list], sum of single-knockout drops x100]
  link  = [from group, to group, sum of relative drops x100, number of passing links]
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import links as links_mod  # noqa: E402

DATA = HERE.parents[1] / "site" / "map" / "data"
ROWS = ["answer", "late", "early"]
BANDS = {"late": [(20, 29), (30, 35)], "early": [(0, 9), (10, 19)]}
MAX_NAMED = 4           # named boxes per row; weaker named neurons fold into one "other named neurons" box


def main():
    nm = json.loads((DATA / "neuronmap.json").read_text(encoding="utf-8"))
    N, names = nm["neurons"], nm["names"]
    codes = [j["code"] for j in nm["job_types"]]
    SHARED = 1 << codes.index("shared")
    era_of = {e["unit"]: e for e in nm["eras"]}
    unit = lambda k: f"L{N['L'][k]}:{N['i'][k]}"
    LK, _ = links_mod.load()
    lk_by = defaultdict(list)
    for c, v in LK.items():
        lk_by[c.strip().casefold().removeprefix("the ")].extend(v)

    out, kinds = {}, defaultdict(int)
    for c in nm["concepts"]:
        slug, name, answer, top = c[0], c[1], c[2] or c[1], c[4]
        p = DATA / "c" / f"{slug}.json"
        roles, arank = {}, {}
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            m = next((m for m in d["maps"] if m.get("status") == "mapped"), None)
            if m and m.get("top_cols") and "role" in m["top_cols"]:
                ri = m["top_cols"].index("role")
                roles = {t[0]: t[ri] for t in m["top"]}
                if "answer_rank" in m["top_cols"]:
                    ai = m["top_cols"].index("answer_rank")
                    arank = {t[0]: t[ai] for t in m["top"]}
        # each neuron's row and box
        keyed = []
        for j, t in enumerate(top):
            k, drop = t[0], t[1]
            L, u = N["L"][k], unit(k)
            row = "early" if L < 20 else "answer" if roles.get(u) in ("say", "attribute") else "late"
            if N["nm"][k] >= 0:
                key = ("name", names[N["nm"][k]])
            elif u in era_of and era_of[u].get("tested"):
                key = ("era", f"{era_of[u]['c']} things")
            elif str(k) in nm["scales"] and nm["scales"][str(k)]["status"] in ("confirmed", "likely"):
                s = nm["scales"][str(k)]
                key = ("scale", f"{s['pos'].split(',')[0]} or {s['neg'].split(',')[0]}")
            elif N["jt"][k] & SHARED:
                key = ("shared", "Shared answer machinery")
            elif row == "answer":
                key = ("say", f"Pushes “{answer.strip()}”")
            elif row == "early" and (arank.get(u) or 99) <= 10:      # an early neuron whose weights already point there
                key = ("say_early", f"Already points to “{answer.strip()}”")
            else:
                lo, hi = next(b for b in BANDS[row] if b[0] <= L <= b[1])
                key = ("unclear", f"Not yet clear, layers {lo}–{hi}")
            keyed.append((j, row, key, max(0, drop)))
        # fold the weakest named boxes of a crowded row into one
        for row in ROWS:
            named = defaultdict(int)
            for j, r, key, drop in keyed:
                if r == row and key[0] == "name":
                    named[key] += drop
            keep = set(sorted(named, key=lambda x: -named[x])[:MAX_NAMED])
            if len(named) > MAX_NAMED:
                keyed = [(j, r, key if r != row or key[0] != "name" or key in keep else ("names", "Other named neurons"), drop)
                         for j, r, key, drop in keyed]
        groups, gi = [], {}
        for j, row, key, drop in keyed:
            g = gi.get((row, key))
            if g is None:
                g = gi[(row, key)] = len(groups)
                groups.append([ROWS.index(row), key[0], key[1], [], 0])
            groups[g][3].append(j)
            groups[g][4] += drop
        for g in groups:
            kinds[g[1]] += 1
        # links between groups: passing links among the 30, summed
        where = {unit(t[0]): gi_ for gi_, g in enumerate(groups) for jj in g[3] for t in [top[jj]]}
        acc = defaultdict(lambda: [0.0, 0])
        for x in lk_by.get(name.strip().casefold().removeprefix("the "), []):
            a, b = where.get(x["from"]), where.get(x["to"])
            if x["passes"] and a is not None and b is not None and a != b:
                acc[(a, b)][0] += x["drop"]; acc[(a, b)][1] += 1
        lks = [[a, b, round(v * 100), n] for (a, b), (v, n) in acc.items() if round(v * 100)]
        out[slug] = [groups, lks]
    p = DATA / "cgraph.json"
    p.write_text(json.dumps({"rows": ROWS, "c": out}, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    nb = [len(v[0]) for v in out.values()]
    print(f"{p}: {p.stat().st_size / 1e6:.1f} MB; {len(out)} concepts; boxes per concept: median "
          f"{sorted(nb)[len(nb) // 2]}, max {max(nb)}; boxes by kind: {dict(kinds)}; "
          f"{sum(1 for v in out.values() if v[1])} concepts with links")


if __name__ == "__main__":
    main()

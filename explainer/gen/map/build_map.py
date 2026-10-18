"""Build the static data behind the Qwen3-8B map (explainer/site/map/data/).

    python -I -X utf8 explainer/gen/map/build_map.py [--only run1,earlier,...]

Reads every source in sources/ (in SOURCES order), merges concept maps measured by different runs, aggregates the
per-neuron, per-head and per-layer views, attaches findings to their units, and writes small JSON files that the
viewer (and anyone else) can fetch directly. The arrays pass (arrays_pass.py) is separate and cached, because it
reads ~30 GB; its output is merged here when present. Nothing here calls a model or an API.
"""
from __future__ import annotations

import argparse
import importlib
import json
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CACHE, N_HEADS, N_LAYERS, N_NEURONS, OUT, REPO, REPO_URL, clean_text, hlayer, nlayer,  # noqa: E402
                    read_json, read_jsonl, slug, write_json)

SOURCES = ["loop_v2_run1", "earlier_maps", "labels", "web_firings", "facts", "ledger", "curated"]
SRC_CODE = {"run1": 0, "earlier": 1}          # compact source codes inside neuron 'uses'
HUB_SHARE = 0.01                               # a neuron in the top-10 of > 1% of mapped concepts is a 'hub'
STRONG_HEAD = 0.5                              # head knockout drop (log-odds) counted as 'strong' in head summaries


class Ctx:
    def __init__(self):
        self.maps, self.findings, self.warnings, self.extra_sources = [], [], [], []

    def add_source(self, s):
        """A module may register further sources (e.g. one per ledger experiment); findings cite them by id."""
        self.extra_sources.append(dict(s))

    def add_map(self, m):
        self.maps.append(m)

    def add_finding(self, f):
        self.findings.append(f)

    def warn(self, msg):
        self.warnings.append(msg)
        print("WARN", msg)


def ckey(name):
    k = name.strip().casefold()
    return k[4:] if k.startswith("the ") else k


def git_head():
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        return None


def r3(x):
    return None if x is None else round(float(x), 3)


def map_noise(m):
    """Per-concept noise floor: the larger |mean drop| of the two random same-layer neurons (run 1 only)."""
    return max((abs(v) for _, v in m.get("null") or []), default=None)


def explainer_links():
    p = REPO / "explainer" / "site" / "data" / "site.json"
    if not p.exists():
        return {}
    site = read_json(p)
    out = {}
    for c in site.get("concepts", []):
        if isinstance(c, dict) and c.get("name"):
            out[ckey(c["name"])] = c.get("id") or c.get("key")
        elif isinstance(c, str):
            out[ckey(c)] = c
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="comma-separated source modules to load (default: all)")
    ap.add_argument("--shards", type=int, default=0,
                    help="write concepts as N shard files cs/<k>.json instead of one file each (for hosts that cap "
                         "the number of files, e.g. a claude.ai artifact)")
    ap.add_argument("--out", default=None, help="output folder (default: explainer/site/map/data or $MAP_OUT)")
    args = ap.parse_args()
    global OUT
    if args.out:
        OUT = Path(args.out)
    t0 = time.time()
    ctx = Ctx()
    sources = []
    for mod_name in SOURCES if not args.only else args.only.split(","):
        mod = importlib.import_module(f"sources.{mod_name}")
        n_maps, n_f = len(ctx.maps), len(ctx.findings)
        t = time.time()
        mod.load(ctx)
        s = dict(mod.SOURCE)
        s["n_maps"], s["n_findings"] = len(ctx.maps) - n_maps, len(ctx.findings) - n_f
        sources.append(s)
        print(f"{mod_name:14s} maps +{s['n_maps']:5d}  findings +{s['n_findings']:5d}  ({time.time() - t:.1f}s)")
    per_src = Counter(f.get("source") for f in ctx.findings)
    for s in ctx.extra_sources:
        s.setdefault("n_maps", 0)
        s["n_findings"] = per_src.get(s["id"], 0)
        sources.append(s)
        print(f"  {s['id']:24s} findings {s['n_findings']:6d}")

    # ---- arrays pass (lights-up strips, logit lens), cached separately
    arrays = {}
    ap_cache = CACHE / "run1_arrays.jsonl"
    if ap_cache.exists():
        for r in read_jsonl(ap_cache):
            arrays[r["concept"]] = r
    print(f"arrays cache: {len(arrays)} concepts")

    # ---- concepts: merge maps of the same concept from different runs
    by_key = defaultdict(list)
    for m in ctx.maps:
        by_key[ckey(m["concept"])].append(m)
    expl = explainer_links()
    concepts, used_slugs = [], set()
    for key, ms in by_key.items():
        ms.sort(key=lambda m: (SRC_CODE.get(m["source"], 9), m["status"] != "mapped"))
        primary = next((m for m in ms if m["status"] == "mapped"), ms[0])
        s = slug(primary["concept"]); base = s; i = 2
        while s in used_slugs:
            s = f"{base}-{i}"; i += 1
        used_slugs.add(s)
        for m in ms:
            a = arrays.get(m["concept"]) if m["source"] == "run1" else None
            if a:
                m["light"], m["lens"] = a.get("light"), a.get("lens")
        concepts.append({"slug": s, "name": primary["concept"], "answer": primary.get("answer"), "maps": ms,
                         "primary": primary, "meta": next((m["meta"] for m in ms if m.get("meta")), {}),
                         "explainer": expl.get(key)})
    concepts.sort(key=lambda c: c["name"].casefold())
    cindex = {c["slug"]: i for i, c in enumerate(concepts)}
    mapped = [c for c in concepts if c["primary"]["status"] == "mapped"]
    print(f"concepts: {len(concepts)} ({len(mapped)} mapped, {sum(len(c['maps']) > 1 for c in concepts)} measured twice)")

    # ---- neurons: every neuron in any map's top 30
    neurons = defaultdict(lambda: {"uses": [], "roles": Counter(), "jobs": None, "jobs_drop": -1e9, "az": None,
                                   "findings": []})
    top10_count = Counter()
    for c in concepts:
        for m in c["maps"]:
            if m["status"] != "mapped":
                continue
            for r, t in enumerate(m["top"]):
                n = neurons[t["n"]]
                n["uses"].append([cindex[c["slug"]], SRC_CODE.get(m["source"], 9), r + 1, r3(t["drop"])])
                if t.get("role"):
                    n["roles"][t["role"]] += 1
                if t.get("jobs") and (t["drop"] or 0) > n["jobs_drop"]:
                    n["jobs"], n["jobs_drop"] = t["jobs"], t["drop"] or 0
                if t.get("answer_z") is not None:
                    n["az"] = max(n["az"] or -1e9, t["answer_z"])
                if r < 10 and m is c["primary"]:
                    top10_count[t["n"]] += 1
    # maps made before loop v2 have no weights-based roles or pushed words: borrow the neuron's majority role and its
    # words from the maps that measured them, flagged as inferred (the sign at the answer position can differ)
    n_inferred = 0
    for c in concepts:
        for m in c["maps"]:
            for t in m.get("top") or []:
                v = neurons.get(t["n"])
                if v and not t.get("role") and v["roles"]:
                    t["role"], t["inferred"] = v["roles"].most_common(1)[0][0], True; n_inferred += 1
                if v and not t.get("jobs") and v["jobs"]:
                    t["jobs"], t["inferred"] = v["jobs"], True
    print(f"roles inferred for {n_inferred} earlier-map slots")
    hub_cut = max(5, int(HUB_SHARE * len(mapped)))
    hubs = {n for n, k in top10_count.items() if k > hub_cut}
    print(f"neurons in maps: {len(neurons)}; hubs (top-10 of > {hub_cut} concepts): {len(hubs)}")

    # ---- findings: attach to units
    model_f, layer_f, head_f, concept_f, unit_counts = [], [], [], defaultdict(list), Counter()
    for i, f in enumerate(ctx.findings):
        f.setdefault("id", f"f{i}")
        u = f["unit"]
        if u == "model":
            model_f.append(f)
        elif u.startswith("layers:") or u.startswith("layer:"):
            layer_f.append(f)
        elif "H" in u and u.startswith("L") and ":" not in u:
            head_f.append(f)
        elif u.startswith("concept:"):
            concept_f[u.split(":", 1)[1]].append(f)
        else:
            neurons[u]["findings"].append(f)
        if f.get("concept"):
            concept_f[f["concept"]].append(f)
        unit_counts[f["kind"]] += 1

    # ---- related concepts: shared non-hub top-10 neurons (primary maps)
    holders = defaultdict(list)
    for c in mapped:
        for t in c["primary"]["top"][:10]:
            if t["n"] not in hubs:
                holders[t["n"]].append(c["slug"])
    related = {}
    for c in mapped:
        cnt, via = Counter(), defaultdict(list)
        for t in c["primary"]["top"][:10]:
            for o in holders.get(t["n"], []):
                if o != c["slug"]:
                    cnt[o] += 1; via[o].append(t["n"])
        related[c["slug"]] = [[o, k, via[o]] for o, k in cnt.most_common(10)]

    # ---- heads
    heads = defaultdict(lambda: {"drops": [], "top": []})
    for c in concepts:
        for m in c["maps"]:
            if m is not c["primary"]:
                continue
            for h, d in m.get("heads") or []:
                heads[h]["drops"].append(d); heads[h]["top"].append([cindex[c["slug"]], r3(d)])
    heads_out = {}
    for h, v in heads.items():
        v["top"].sort(key=lambda x: -x[1])
        heads_out[h] = {"n": len(v["drops"]), "strong": sum(d >= STRONG_HEAD for d in v["drops"]),
                        "median": r3(statistics.median(v["drops"])), "max": r3(max(v["drops"])),
                        "top": v["top"][:25], "findings": [f for f in head_f if f["unit"] == h]}
    for f in head_f:
        heads_out.setdefault(f["unit"], {"n": 0, "strong": 0, "median": None, "max": None, "top": [], "findings": [f]})

    # ---- layers
    layers = []
    top5_roles = [Counter() for _ in range(N_LAYERS)]
    top5_slots = [0] * N_LAYERS
    for c in mapped:
        for t in c["primary"]["top"][:5]:
            L = nlayer(t["n"]); top5_slots[L] += 1
            if t.get("role"):
                top5_roles[L][t["role"]] += 1
    in_layer = Counter(nlayer(n) for n, v in neurons.items() if v["uses"])
    share_sum = [0.0] * N_LAYERS; share_n = 0
    for c in mapped:
        ls = (c["primary"].get("light") or {}).get("layer_share")
        if ls:
            share_n += 1
            for L in range(N_LAYERS):
                share_sum[L] += ls[L]
    for L in range(N_LAYERS):
        hs = {h: v for h, v in heads_out.items() if hlayer(h) == L}
        bands = [f["id"] for f in layer_f if _in_range(f["unit"], L)]
        layers.append({"layer": L, "neurons_in_maps": in_layer[L], "top5_slots": top5_slots[L],
                       "top5_roles": dict(top5_roles[L]), "hubs": sum(1 for n in hubs if nlayer(n) == L),
                       "attr_share": r3(share_sum[L] / share_n) if share_n else None,
                       "heads": {h.split("H")[1]: [v["n"], v["strong"], v["median"]] for h, v in hs.items()},
                       "bands": bands})

    # ---- write
    OUT.mkdir(parents=True, exist_ok=True)
    sizes = Counter()
    src_ids = [s["id"] for s in sources]
    per_shard = -(-len(concepts) // args.shards) if args.shards else 0
    rows = []
    for ci_, c in enumerate(concepts):
        p = c["primary"]
        rows.append([c["slug"], c["name"], c["answer"], c["meta"].get("section") or c["meta"].get("category"),
                     c["meta"].get("tier"), c["meta"].get("kind"), c["meta"].get("type"), p["status"],
                     p.get("k_mean"), p.get("k_flip"), p["top"][0]["n"] if p.get("top") else None,
                     "".join(sorted({str(SRC_CODE.get(m["source"], 9)) for m in c["maps"]})),
                     r3(map_noise(p)), r3(p.get("baseline"))] + ([ci_ // per_shard] if per_shard else []))
    sizes["concepts.json"] += write_json(OUT / "concepts.json", {
        "cols": ["slug", "name", "answer", "section", "tier", "kind", "type", "status", "k_mean", "k_flip", "top1",
                 "sources", "noise", "baseline"] + (["shard"] if per_shard else []), "rows": rows})
    shard_docs = defaultdict(dict)
    for ci_, c in enumerate(concepts):
        out = {"slug": c["slug"], "name": c["name"], "answer": c["answer"], "meta": c["meta"],
               "explainer": c["explainer"], "related": related.get(c["slug"], []),
               "findings": concept_f.get(c["slug"], []), "maps": []}
        for m in c["maps"]:
            mm = {k: m.get(k) for k in ("source", "batch", "status", "recall", "baseline", "k_mean", "k_median",
                                        "k_flip", "ks", "drops", "random_drops", "prompts", "prov", "light", "lens",
                                        "answer", "file") if m.get(k) is not None}
            if m.get("top"):
                mm["top"] = [[t["n"], r3(t["drop"]), t.get("attr_rank"), r3(t.get("act")), t.get("role"),
                              t.get("answer_rank"), r3(t.get("answer_z")), (t.get("jobs") or [])[:8],
                              1 if t["n"] in hubs else 0, 1 if t.get("inferred") else 0] for t in m["top"]]
                mm["top_cols"] = ["n", "drop", "attr_rank", "act", "role", "answer_rank", "answer_z", "jobs", "hub",
                                  "inferred"]
            if m.get("null"):
                mm["null"] = m["null"]; mm["noise"] = r3(map_noise(m))
            if m.get("heads"):
                mm["heads"] = [[h, r3(d)] for h, d in m["heads"]]
            out["maps"].append(mm)
        if len([m for m in c["maps"] if m["status"] == "mapped"]) > 1:
            a, b = [m for m in c["maps"] if m["status"] == "mapped"][:2]
            ta, tb = [t["n"] for t in a["top"][:10]], [t["n"] for t in b["top"][:10]]
            out["replication"] = {"maps": [a["source"], b["source"]], "top10_overlap": len(set(ta) & set(tb)),
                                  "top1_same": ta[0] == tb[0], "shared": sorted(set(ta) & set(tb)),
                                  "top5_in_other_top30": sum(n in {t["n"] for t in b["top"]} for n in ta[:5])}
        if per_shard:
            shard_docs[ci_ // per_shard][c["slug"]] = out
        else:
            sizes["c/"] += write_json(OUT / "c" / f"{c['slug']}.json", out)
    for k, docs in shard_docs.items():
        sizes["cs/"] += write_json(OUT / "cs" / f"{k}.json", docs)

    by_layer = defaultdict(dict)
    for n, v in neurons.items():
        L = nlayer(n)
        roles = v["roles"]
        by_layer[L][n.split(":")[1]] = {
            "uses": sorted(v["uses"], key=lambda u: (u[1], -(u[3] or 0)))[:400], "n_uses": len(v["uses"]),
            "role": roles.most_common(1)[0][0] if roles else None, "roles": dict(roles),
            "jobs": v["jobs"], "az": r3(v["az"]), "hub": n in hubs, "top10": top10_count.get(n, 0),
            "findings": v["findings"]}
    for L in range(N_LAYERS):
        sizes["n/"] += write_json(OUT / "n" / f"L{L}.json", by_layer.get(L, {}))

    # job-token index: say/attribute neurons' top pushed tokens -> neurons (for search)
    tok = defaultdict(set)
    for n, v in neurons.items():
        if v["jobs"] and (v["roles"].get("say") or v["roles"].get("attribute")):
            for w in v["jobs"][:6]:
                k = w.strip().casefold()
                if len(k) >= 3 and any(ch.isalpha() for ch in k):
                    tok[k].add(n)
    sizes["tokens.json"] += write_json(OUT / "tokens.json", {k: sorted(v) for k, v in tok.items() if len(v) <= 60})
    sizes["heads.json"] += write_json(OUT / "heads.json", heads_out)
    sizes["layers.json"] += write_json(OUT / "layers.json", {"layers": layers, "bands": layer_f})
    sizes["findings.json"] += write_json(OUT / "findings.json", {"model": model_f, "layers": layer_f, "heads": head_f,
                                                                "counts": dict(unit_counts)})
    with open(OUT / "findings.jsonl", "w", encoding="utf-8") as fh:
        for f in ctx.findings:
            fh.write(clean_text(json.dumps(f, ensure_ascii=False)) + "\n")
    sizes["findings.jsonl"] += (OUT / "findings.jsonl").stat().st_size
    meta = {"model": "Qwen/Qwen3-8B", "revision": "b968826d9c46dd6066d109eabc6255188de91218",
            "layers": N_LAYERS, "neurons_per_layer": N_NEURONS, "heads_per_layer": N_HEADS,
            "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "code_commit": git_head(),
            "repo": REPO_URL, "sources": sources, "source_codes": SRC_CODE, "hub_cut": hub_cut,
            "strong_head": STRONG_HEAD,
            "counts": {"concepts": len(concepts), "mapped": len(mapped),
                       "neurons_in_maps": sum(1 for v in neurons.values() if v["uses"]),
                       "neurons_with_findings": sum(1 for v in neurons.values() if v["findings"]),
                       "heads": len(heads_out), "findings": len(ctx.findings),
                       "measured_twice": sum(len([m for m in c["maps"] if m["status"] == "mapped"]) > 1 for c in concepts)},
            "warnings": ctx.warnings, "src_order": src_ids}
    sizes["meta.json"] += write_json(OUT / "meta.json", meta)
    print("sizes (MB):", {k: round(v / 1e6, 2) for k, v in sizes.items()}, f"total {sum(sizes.values()) / 1e6:.1f} MB")
    print(f"done in {time.time() - t0:.0f}s -> {OUT}")


def _in_range(unit, L):
    spec = unit.split(":", 1)[1]
    a, _, b = spec.partition("-")
    return int(a) <= L <= int(b or a)


if __name__ == "__main__":
    main()

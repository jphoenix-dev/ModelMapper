"""What is already known about a neuron or a concept, without running a model.

    lookup("L31:8930")          # the published Qwen3-8B map: the concepts that depend on this neuron, by drop
    lookup("Eiffel Tower")      # the published map: the concept's neurons
    lookup("L23:2392", records="maps/Qwen3-0.6B")   # your own records (map-concept / map-batch output), any model

The published map is read from MODEL_MAPPER_MAP_URL (FORMAT.md, "Published map summary": concepts.json and one
n/L<layer>.json per layer), downloaded once into the cache and re-read from there (refresh=True downloads again).
A neuron lookup needs one layer's file (~1.4 MB); the first concept lookup downloads all of them (~17 MB).
"""
from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path

from modelmapper.settings import CACHE_DIR, MAP_URL

SOURCES = {0: "run 1", 1: "earlier maps"}
CROSS_RUN = ("drops from different runs (and possibly different hardware): each call (in / uncertain / out) holds within "
             "its own run; comparing drops across runs is not covered by the hardware band (METHOD.md)")      # the 'source' code in a published neuron's uses (FORMAT.md)
_UNIT = re.compile(r"^\s*L?(\d+)[:_]N?(\d+)\s*$")


def _get(name, refresh=False):
    path = CACHE_DIR / "map" / name
    if refresh or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"{MAP_URL}/{name}"
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                data = r.read()
        except Exception as e:  # noqa: BLE001
            raise ConnectionError(f"could not download {url} ({type(e).__name__}: {e}); set MODEL_MAPPER_MAP_URL, "
                                  "or look in your own records with records=<folder>") from e
        tmp = path.with_suffix(".tmp"); tmp.write_bytes(data); tmp.replace(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _concepts(refresh=False):
    c = _get("concepts.json", refresh)
    cols = c["cols"]
    return [dict(zip(cols, r)) for r in c["rows"]]


def _layer(l, refresh=False):
    return _get(f"n/L{l}.json", refresh)


def _find_concept(rows, name):
    key = name.strip().casefold()
    hit = [i for i, r in enumerate(rows) if key in (str(r.get("name", "")).casefold(), str(r.get("slug", "")).casefold(),
                                                     str(r.get("answer", "") or "").strip().casefold())]
    if not hit:
        close = [r["name"] for r in rows if key in str(r.get("name", "")).casefold()][:8]
        raise KeyError(f"{name!r} is not in the map" + (f"; did you mean: {', '.join(close)}?" if close else ""))
    return hit[0]


def lookup_published(query, refresh=False) -> dict:
    rows = _concepts(refresh)
    m = _UNIT.match(query)
    if m:
        l, n = int(m.group(1)), int(m.group(2))
        e = _layer(l, refresh).get(str(n))
        if e is None:
            return {"query": query, "kind": "neuron", "source": MAP_URL, "found": False,
                    "note": "this neuron is in no mapped concept's top 30 in the published map"}
        uses = [{"concept": rows[ci]["name"] if ci < len(rows) else ci, "rank": rk, "drop": d, "from": SOURCES.get(src, src)}
                for ci, src, rk, d in e.get("uses", [])]
        return {"query": f"L{l}:{n}", "kind": "neuron", "source": MAP_URL, "found": True, "n_uses": e.get("n_uses", len(uses)),
                "role": e.get("role"), "roles": e.get("roles"), "pushes": e.get("jobs"), "hub": e.get("hub"),
                "uses": sorted(uses, key=lambda u: -u["drop"]), "findings": e.get("findings", [])}
    ci = _find_concept(rows, query)
    r = rows[ci]
    neurons = []
    for l in range(36):
        for n, e in _layer(l, refresh).items():
            for c2, src, rk, d in e.get("uses", []):
                if c2 == ci:
                    neurons.append({"neuron": f"L{l}:{n}", "rank": rk, "drop": d, "from": SOURCES.get(src, src),
                                    "role": e.get("role")})
    return {"query": query, "kind": "concept", "source": MAP_URL, "found": True, "concept": r,
            "neurons": sorted(neurons, key=lambda x: x["rank"])}


def lookup_records(query, folder) -> dict:
    """The same questions, answered from a folder of records (FORMAT.md), e.g. map-batch output."""
    from modelmapper.records import MapResult
    recs = [MapResult.load(p) for p in sorted(Path(folder).rglob("*.json")) if p.parent.name != "spec_cache"]
    recs = [r for r in recs if r.record.get("concept")]
    m = _UNIT.match(query)
    if m:
        u = f"L{int(m.group(1))}:{int(m.group(2))}"
        uses = [{"concept": r.concept, "rank": i + 1, "drop": t.drop, "model": r.model, "role": t.role, "call": t.call,
                 "gpu": r.provenance.get("gpu"), "dtype": r.provenance.get("dtype")}
                for r in recs for i, t in enumerate(r.top) if t.neuron == u]
        return {"query": u, "kind": "neuron", "source": str(folder), "found": bool(uses), "n_uses": len(uses),
                "uses": sorted(uses, key=lambda x: -x["drop"]),
                "note": CROSS_RUN if uses else "no record in this folder has it in its top 30"}
    hit = [r for r in recs if r.concept.casefold() == query.strip().casefold()]
    if not hit:
        raise KeyError(f"{query!r} has no record in {folder}")
    r = hit[0]
    return {"query": query, "kind": "concept", "source": r.path, "found": True, "status": r.status, "model": r.model,
            "neurons": [{"neuron": t.neuron, "rank": i + 1, "drop": t.drop, "role": t.role} for i, t in enumerate(r.top)]}


def lookup(query, records=None, refresh=False) -> dict:
    return lookup_records(query, records) if records else lookup_published(query, refresh)


def show(res, n=15) -> str:
    """The result as text for the terminal."""
    lines = []
    if res["kind"] == "neuron":
        if not res["found"]:
            return f"{res['query']}: not found ({res.get('note', 'no record uses it')}). Source: {res['source']}"
        lines.append(f"{res['query']}: in {res['n_uses']} concept map(s)" + (f"; role {res['role']}" if res.get("role") else "")
                     + (" (a hub: in very many maps)" if res.get("hub") else "") + f". Source: {res['source']}")
        if res.get("pushes"):
            lines.append(f"  pushes the model towards: {' '.join(str(w).strip() for w in res['pushes'][:12])}")
        lines.append(f"  {'concept':<40} {'rank':>4} {'drop':>7}")
        for u in res["uses"][:n]:
            lines.append(f"  {str(u['concept'])[:40]:<40} {u['rank']:>4} {u['drop']:>7.2f}")
        if len(res["uses"]) > n:
            lines.append(f"  ... and {len(res['uses']) - n} more (--json for all)")
        if res.get("note") and len(res["uses"]) > 1:            # each record is its own run
            lines.append(f"  note: {res['note']}")
    else:
        c = res.get("concept") or {}
        lines.append(f"{c.get('name', res['query'])}: {len(res['neurons'])} neurons in its map. Source: {res['source']}")
        lines.append(f"  {'neuron':<12} {'rank':>4} {'drop':>7}  role")
        for x in res["neurons"][:n]:
            lines.append(f"  {x['neuron']:<12} {x['rank']:>4} {x['drop']:>7.2f}  {x.get('role') or ''}")
        if len(res["neurons"]) > n:
            lines.append(f"  ... and {len(res['neurons']) - n} more (--json for all)")
    return "\n".join(lines)

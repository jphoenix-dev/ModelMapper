"""What is already known about a neuron or a concept, without running a model.

    lookup("L31:8930")          # the published Qwen3-8B map: the concepts that depend on this neuron, by drop,
                                # and the published v3 labels of its gate, + side and - side (if it is one of the 11,000)
    lookup("Eiffel Tower")      # the published map: the concept's neurons; and the labelled neurons whose labels
                                # contain the word(s)
    lookup("L23:2392", records="maps/Qwen3-0.6B")   # your own records (map-concept / map-batch output, and
                                                    # `label` output: <prefix>_results.json), any model

The published map is read from MODEL_MAPPER_MAP_URL (FORMAT.md, "Published map summary": concepts.json and one
n/L<layer>.json per layer), downloaded once into the cache and re-read from there (refresh=True downloads again).
A neuron lookup needs one layer's file (~1.4 MB); the first concept lookup downloads all of them (~17 MB).
The published labels ship in the package (data/labels/Qwen3-8B_v3.json.gz, built by scripts/build_label_data.py) and
need no download; they are shown even when the map can't be downloaded.
"""
from __future__ import annotations

import gzip
import json
import re
import textwrap
import urllib.request
from pathlib import Path

from modelmapper.settings import CACHE_DIR, MAP_URL

SOURCES = {0: "run 1", 1: "earlier maps"}
CROSS_RUN = ("drops from different runs (and possibly different hardware): each call (in / uncertain / out) holds within "
             "its own run; comparing drops across runs is not covered by the hardware band (METHOD.md)")      # the 'source' code in a published neuron's uses (FORMAT.md)
_UNIT = re.compile(r"^\s*L?(\d+)[:_]N?(\d+)\s*$")
LABELS_FILE = Path(__file__).parent / "data" / "labels" / "Qwen3-8B_v3.json.gz"
PARTS = {"gate": "Gate", "apos": "Positive", "aneg": "Negative"}
MAX_HITS = 10
_published_labels: dict | None = None


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


# ---------------------------------------------------------------- labels (published v3, or your own `label` output)
def short_label(text: str, n: int = 56) -> str:
    """The label's first phrase, for one-line lists (a summary of the full label, never a replacement for it)."""
    t = (text or "").strip()
    cuts = [i for i in (t.find(c, 12) for c in (":", ";", ". ", " (", " — ", ", plus", ", with")) if i >= 12]
    t = t[:min(cuts + [len(t)])].rstrip(" .,;:")
    return t if len(t) <= n else t[:n].rsplit(" ", 1)[0].rstrip(" .,;:") + "…"


def published_labels() -> dict:
    """The packaged v3 labels: {"source": {...}, "neurons": {neuron: record}} (records in the scorecard's format)."""
    global _published_labels
    if _published_labels is None:
        d = json.loads(gzip.decompress(LABELS_FILE.read_bytes()).decode("utf-8"))
        meta = {k: v for k, v in d["_meta"].items() if k not in ("scorecard", "fields")}
        _published_labels = {"source": {"file": f"modelmapper/data/labels/{LABELS_FILE.name}", **meta}, "neurons": d["neurons"]}
    return _published_labels


def own_labels(folder) -> list[dict]:
    """Every `<prefix>_results.json` (the `label` command's scorecard) under folder, in the same shape."""
    out = []
    for p in sorted(Path(folder).rglob("*_results.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(d, dict) and isinstance(d.get("neurons"), dict):
            out.append({"source": {"file": str(p), "own": True}, "neurons": d["neurons"]})
    return out


def _entry(u, rec, src) -> dict:
    e = {"neuron": u, "source": src, **rec}
    for b in PARTS:
        if e.get(b) and not e[b].get("short"):
            e[b] = {**e[b], "short": short_label(e[b].get("label") or "")}
    return e


def labels_for(u, sets) -> list[dict]:
    return [_entry(u, s["neurons"][u], s["source"]) for s in sets if u in s["neurons"]]


def search_labels(word, sets) -> list[dict]:
    """The labelled neurons whose labels contain the word(s) (whole words, a plural allowed): one hit per neuron and
    source, its best-matching part first (verified, then the word in the short label, then the exam AUC)."""
    w = word.strip()
    if len(w) < 2:
        return []
    rx = re.compile(r"(?<!\w)" + re.escape(w) + r"(?:s|es)?(?!\w)", re.IGNORECASE)
    hits = []
    for s in sets:
        for u, rec in s["neurons"].items():
            best = None
            for b in PARTS:
                p = rec.get(b) or {}
                m = rx.search(p.get("label") or "")
                if not m:
                    continue
                short = p.get("short") or short_label(p["label"])
                key = (bool(p.get("verified")), bool(rx.search(short)), p.get("auc") or 0)
                if best is None or key > best[0]:
                    lab = p["label"]
                    snip = None if key[1] else "…" + lab[max(0, m.start() - 25):m.end() + 25].strip() + "…"
                    best = (key, {"neuron": u, "part": b, "short": short, "verified": key[0], "auc": p.get("auc"),
                                  "gen_pass": p.get("gen_pass"), "match": snip, "source": s["source"].get("file")})
            if best:
                hits.append(best)
    hits.sort(key=lambda h: (not h[0][0], not h[0][1], -h[0][2]))
    return [h[1] for h in hits]


# ---------------------------------------------------------------- lookup
def _neuron_published(l, n, refresh):
    rows = _concepts(refresh)
    e = _layer(l, refresh).get(str(n))
    if e is None:
        return {"query": f"L{l}:{n}", "kind": "neuron", "source": MAP_URL, "found": False,
                "note": "this neuron is in no mapped concept's top 30 in the published map"}
    uses = [{"concept": rows[ci]["name"] if ci < len(rows) else ci, "rank": rk, "drop": d, "from": SOURCES.get(src, src)}
            for ci, src, rk, d in e.get("uses", [])]
    return {"query": f"L{l}:{n}", "kind": "neuron", "source": MAP_URL, "found": True, "n_uses": e.get("n_uses", len(uses)),
            "role": e.get("role"), "roles": e.get("roles"), "pushes": e.get("jobs"), "hub": e.get("hub"),
            "uses": sorted(uses, key=lambda u: -u["drop"]), "findings": e.get("findings", [])}


def _concept_published(query, refresh):
    rows = _concepts(refresh)
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


def _with_labels(query, source, concepts, label_sets):
    """Run the concept-map lookup, add the labels; a map that is missing or has no answer is fine when labels have one."""
    m = _UNIT.match(query)
    u = f"L{int(m.group(1))}:{int(m.group(2))}" if m else None
    extra = {"labels": labels_for(u, label_sets)} if u else {"label_hits": search_labels(query, label_sets)}
    have = bool(next(iter(extra.values())))
    try:
        res = concepts()
    except (KeyError, ConnectionError) as e:
        if not have:
            raise
        res = {"query": u or query, "kind": "neuron" if u else "concept", "source": source, "found": False,
               "note": e.args[0] if e.args else str(e), **({} if u else {"neurons": []})}
    return {**res, **extra}


def lookup_published(query, refresh=False) -> dict:
    m = _UNIT.match(query)
    sets = [published_labels()]
    if m:
        return _with_labels(query, MAP_URL, lambda: _neuron_published(int(m.group(1)), int(m.group(2)), refresh), sets)
    return _with_labels(query, MAP_URL, lambda: _concept_published(query, refresh), sets)


def _records(folder):
    from modelmapper.records import MapResult
    out = []
    for p in sorted(Path(folder).rglob("*.json")):
        if p.parent.name == "spec_cache":
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(d, dict) and d.get("concept"):         # a concept record; label files and the like are skipped
            out.append(MapResult.from_record(d, p))
    return out


def lookup_records(query, folder) -> dict:
    """The same questions, answered from a folder of records (FORMAT.md), e.g. map-batch output, and the labels in any
    `<prefix>_results.json` under it (the `label` command's output)."""
    recs = _records(folder)
    m = _UNIT.match(query)

    def neuron():
        u = f"L{int(m.group(1))}:{int(m.group(2))}"
        uses = [{"concept": r.concept, "rank": i + 1, "drop": t.drop, "model": r.model, "role": t.role, "call": t.call,
                 "gpu": r.provenance.get("gpu"), "dtype": r.provenance.get("dtype")}
                for r in recs for i, t in enumerate(r.top) if t.neuron == u]
        return {"query": u, "kind": "neuron", "source": str(folder), "found": bool(uses), "n_uses": len(uses),
                "uses": sorted(uses, key=lambda x: -x["drop"]),
                "note": CROSS_RUN if uses else "no record in this folder has it in its top 30"}

    def concept():
        hit = [r for r in recs if r.concept.casefold() == query.strip().casefold()]
        if not hit:
            raise KeyError(f"{query!r} has no record in {folder}")
        r = hit[0]
        return {"query": query, "kind": "concept", "source": r.path, "found": True, "status": r.status, "model": r.model,
                "neurons": [{"neuron": t.neuron, "rank": i + 1, "drop": t.drop, "role": t.role} for i, t in enumerate(r.top)]}

    return _with_labels(query, str(folder), neuron if m else concept, own_labels(folder))


def lookup(query, records=None, refresh=False) -> dict:
    return lookup_records(query, records) if records else lookup_published(query, refresh)


# ---------------------------------------------------------------- text
def _f(x):
    return "?" if x is None else f"{x:.2f}".replace("-0.00", "0.00")


def _part_text(name, p, width) -> str:
    if not p or not p.get("label"):
        return f"  {name}: not labelled"
    scores = f"exam {_f(p.get('auc'))} vs {_f(p.get('auc_control'))}"
    pad = " " * (len(name) + 4)
    if p.get("verified"):
        g = p.get("gen_pass")
        if p.get("at_edge"):
            scores += "; at the edge: a control also scored above 0.65 (lower confidence)"
        gen = ("generative ✓" if g else f"generative ✗ (AUC {_f(p.get('gen_auc'))}, needs 0.75)" if g is False
               else "generative check not run")
        text, verdict = f"{name}: {p['label'].strip()}", f"✓ {scores}, {gen}"
    else:
        text, verdict = f"{name}: (unchecked guess) {p['label'].strip()}", f"NOT VERIFIED: {p.get('bin')}; {scores}"
    return textwrap.fill(text, width, initial_indent="  ", subsequent_indent=pad) + "\n" + pad + verdict


def _knockout_text(k) -> str:
    prof, n, mx = k.get("profile"), k.get("n_concepts"), k.get("max_drop")
    tested = f"{n} concept{'s' if n != 1 else ''} tested in all" if n else None
    if prof in (None, "untested"):
        return "Knockout: not tested"
    cs = ", ".join(f"{c} ({_f(d)})" for c, d in k.get("concepts") or [])
    if prof == "no effect":
        return ("Knockout: no effect" + (f" (largest drop {_f(mx)}" if mx is not None else " (")
                + (f"; {tested})" if tested else ")")).replace(" ()", "")
    if prof == "effect":
        return "Knockout: breaks " + (cs or f"a concept (largest drop {_f(mx)})") + (f" ({tested})" if tested else "")
    return (f"Knockout: weak (largest drop {_f(mx)}, below the effect bar" + (f"; {tested}" if tested else "") + ")"
            + (f"; drops of 0.3 or more: {cs}" if cs else ""))


def _source_text(src) -> str:
    if src.get("own"):
        return f"Source: your labels, {src['file']}"
    r = src.get("rules") or {}
    return (f"Source: {src.get('source')}; labeller {src.get('labeller')}, judge {src.get('judge')}; method {src.get('method')}. "
            f"Verified: exam AUC ≥ {r.get('exam_auc_min')} and ≥ {r.get('exam_margin_min')} above control; generative ✓: "
            f"AUC ≥ {r.get('generative_auc_min')}; job type cut {r.get('job_cut')}.")


def show_labels(e, width=110) -> str:
    """One neuron's labels as a block, like the explainer's side pane."""
    t = e.get("triage") or {}
    src = e["source"]
    head = "Your labels:" if src.get("own") else f"Published labels ({str(src.get('model', '')).split('/')[-1]}, v3 and v3x runs):"
    bits = [f"triage {t['read']}" + (f", {t['with_knockout']}" if t.get("with_knockout") else "")] if t.get("read") else []
    if e.get("set"):
        bits.append(f"set {e['set']}")
    lines = [head + (f" {'; '.join(bits)}" if bits else "")]
    lines += [_part_text(name, e.get(b), width) for b, name in PARTS.items()]
    j = e.get("jobtype")
    if j:
        lines.append(f"  Job: positive {j['apos']} (cos {_f(j.get('c_pos'))}), negative {j['aneg']} (cos {_f(j.get('c_neg'))})"
                     " (from the weights)")
    pu = e.get("pushes") or {}
    if pu.get("apos") or pu.get("aneg"):
        w = lambda xs: ", ".join(str(x).strip() or repr(x) for x in xs[:8]) or "?"
        lines.append(f"  Pushes: + {w(pu.get('apos') or [])} / − {w(pu.get('aneg') or [])}")
    if e.get("knockout"):
        lines.append(textwrap.fill(_knockout_text(e["knockout"]), width, initial_indent="  ", subsequent_indent="    "))
    lines.append(textwrap.fill(_source_text(e["source"]), width, initial_indent="  ", subsequent_indent="    ",
                               break_long_words=False, break_on_hyphens=False))      # a path stays whole
    return "\n".join(lines)


def show_hits(hits, query, n=MAX_HITS) -> str:
    if not hits:
        return ""
    lines = [f"Labelled neurons whose labels mention {query.strip()!r}: {len(hits)}"]
    for h in hits[:n]:
        tag = "✓" if h["verified"] else "(not verified)"
        line = f"  {h['neuron']:<10} {PARTS[h['part']]:<8} {tag} {h['short']}" + (f"  [{h['match']}]" if h.get("match") else "")
        lines.append(line if len(line) <= 140 else line[:139] + "…")
    if len(hits) > n:
        lines.append(f"  … {len(hits) - n} more (--json for all)")
    return "\n".join(lines)


def show(res, n=15) -> str:
    """The result as text for the terminal."""
    lines = []
    if res["kind"] == "neuron":
        if not res["found"]:
            lines.append(f"{res['query']}: not found ({res.get('note', 'no record uses it')}). Source: {res['source']}")
        else:
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
        for e in res.get("labels") or []:
            lines += ["", show_labels(e)]
    else:
        if not res["found"]:
            lines.append(f"{res['query']}: no concept ({res.get('note')}). Source: {res['source']}")
        else:
            c = res.get("concept") or {}
            lines.append(f"{c.get('name', res['query'])}: {len(res['neurons'])} neurons in its map. Source: {res['source']}")
            lines.append(f"  {'neuron':<12} {'rank':>4} {'drop':>7}  role")
            for x in res["neurons"][:n]:
                lines.append(f"  {x['neuron']:<12} {x['rank']:>4} {x['drop']:>7.2f}  {x.get('role') or ''}")
            if len(res["neurons"]) > n:
                lines.append(f"  ... and {len(res['neurons']) - n} more (--json for all)")
        if res.get("label_hits"):
            lines += ["", show_hits(res["label_hits"], res["query"])]
    return "\n".join(lines)

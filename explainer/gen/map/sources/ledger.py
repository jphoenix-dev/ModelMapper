"""The research ledger's export (D:/already-clean-data/ledger/export; spec LEDGER.md on research/attention).

Each ledger experiment becomes its own source on the map (method, status, reproduce, caveats come from
experiments.jsonl). The export holds ~1.7M evidence rows, so this adapter is selective:
  - summary evidence types (a neuron's job class, readout, entity specificity, web-firing share) -> one finding each;
  - any other evidence that passes its experiment's own threshold (random-neuron controls excluded) -> one finding
    each when there are few, else one finding per unit summarising how many of its tests pass and the strongest.
Experiments the map already reads directly (run 1) and ones marked invalid are skipped.
"""
from __future__ import annotations

import gzip
import json
import os
from collections import defaultdict
from pathlib import Path

from common import RDATA, finding, read_json, read_jsonl

EXPORT = Path(os.environ.get("MAP_LEDGER", RDATA / "ledger" / "export"))

SOURCE = {
    "id": "ledger",
    "title": "Research ledger export",
    "kind": "evidence ledger",
    "status": "per experiment (see each)",
    "date": "2026-10-18",
    "method": "Evidence rows from the project's ledger: one row per measured fact about a unit, each with its "
              "experiment, noise floor and threshold. Each ledger experiment is listed as its own source.",
    "reproduce": "python src/ledger/export.py   (research/attention branch)",
    "docs": ["LEDGER.md"],
    "caveats": ["Selective: summary evidence and threshold-passing evidence only; the full export has every row."],
}

SKIP = {"loop_v2_run1"}                 # read directly by sources/loop_v2_run1.py (same data); switch over later
# the ledger's ports of this map's own adapters (earlier maps, labels, web firings, curated findings): read directly
# until the map switches to the ledger wholesale, so the same evidence isn't shown twice
SKIP_PREFIX = ("earlier_", "labels_", "web_firings", "curated")
ALWAYS = {"job_class", "readout_tokens", "specificity", "web_firing_share"}
PER_ROW_CAP = 500                       # more passing rows than this for one (type, measure): summarise per unit
KIND = {"job_class": "job class", "readout_tokens": "pushes (readout)", "specificity": "entity specificity",
        "web_firing_share": "web firings", "activation_auc": "selectivity", "knockout_drop": "knockout",
        "dose_response": "dose response", "sharing": "shared with other entities", "position_split": "position"}


def available():
    return (EXPORT / "manifest.json").exists()


def _rows(exp_id):
    for p in sorted((EXPORT / "evidence" / exp_id).glob("*.jsonl.gz")):
        with gzip.open(p, "rt", encoding="utf-8") as fh:
            for line in fh:
                yield json.loads(line)


def _slug(r):
    return r.get("slug") if str(r.get("concept_id") or "").startswith("thing:") else None


def _metric(r):
    m = {k: r.get(k) for k in ("measure", "value", "value_text", "noise_floor", "threshold", "strength", "n", "context")
         if r.get(k) not in (None, "")}
    if isinstance(m.get("value"), float):
        m["value"] = round(m["value"], 3)
    if isinstance(m.get("strength"), float):
        m["strength"] = round(m["strength"], 1)
    return m


def load(ctx):
    if not available():
        ctx.warn(f"ledger: no export at {EXPORT}")
        return
    manifest = read_json(EXPORT / "manifest.json")
    for e in read_jsonl(EXPORT / "experiments.jsonl"):
        eid = e["experiment_id"]
        if (eid in SKIP or eid.startswith(SKIP_PREFIX) or eid in manifest.get("superseded", [])
                or (e.get("status_class") or ("invalid" if "invalid" in (e.get("status") or "").lower() else "")) == "invalid"):
            continue
        prereg = (e.get("status_class") or "") == "pre-registered" or "pre-registered" in (e.get("status") or "")
        ctx.add_source({"id": eid, "title": e.get("title") or eid, "kind": "ledger experiment",
                        "status": e.get("status") or "exploratory", "date": e.get("date"),
                        "method": e.get("method", ""), "reproduce": e.get("reproduce"),
                        "docs": e.get("docs", []), "caveats": e.get("caveats", []),
                        "code_commit": e.get("code_commit"), "pack": e.get("pack_id")})
        status = "pre-registered" if prereg else "exploratory"
        buckets = defaultdict(list)
        for r in _rows(eid):
            if not r.get("unit") or "control=random" in (r.get("context") or ""):
                continue
            t = r["evidence_type"]
            if t in ALWAYS:
                ctx.add_finding(finding(r["unit"], KIND.get(t, t), r.get("claim") or t, eid, status=status,
                                        concept=_slug(r), metric=_metric(r)))
            elif r.get("passes_threshold") is True:
                buckets[(t, r.get("measure"))].append(r)
        for (t, measure), rows in buckets.items():
            if len(rows) <= PER_ROW_CAP:
                for r in rows:
                    ctx.add_finding(finding(r["unit"], KIND.get(t, t), r.get("claim") or t, eid, status=status,
                                            concept=_slug(r), metric=_metric(r)))
                continue
            by_unit = defaultdict(list)
            for r in rows:
                by_unit[r["unit"]].append(r)
            for unit, rs in by_unit.items():
                rs.sort(key=lambda r: -(r.get("value") or 0))
                names = []
                for r in rs:
                    if r.get("concept") and r["concept"] not in names:
                        names.append(r["concept"])
                best = rs[0]
                ctx.add_finding(finding(
                    unit, KIND.get(t, t),
                    f"Passes in {len(rs)} {measure.replace('_', ' ')} test{'s' if len(rs) > 1 else ''} across "
                    f"{len(names)} concept{'s' if len(names) != 1 else ''}; strongest: {best.get('claim')}", eid,
                    status=status, concept=_slug(best),
                    detail="Also: " + ", ".join(names[1:8]) + ("…" if len(names) > 8 else "") if len(names) > 1 else None,
                    metric={"measure": measure, "n_passing": len(rs), "value": best.get("value"),
                            "noise_floor": best.get("noise_floor"), "context": best.get("context")}))

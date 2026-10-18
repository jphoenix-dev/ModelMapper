"""Hand-written findings from the written-up syntheses (findings/*.jsonl): layer bands, head roles, pre-registered tests.

This is also the low-effort way to add a finding: append one JSON line to a file in findings/ (see MAP.md for the
fields), cite the document or results file that backs it, and rebuild.
"""
from __future__ import annotations

from common import HERE, read_jsonl

SOURCE = {
    "id": "syntheses",
    "title": "Written-up syntheses (LAYERS_AND_JOBS, PREREG_MAP, attention study)",
    "kind": "curated findings",
    "status": "as marked on each finding",
    "date": "2026-10-18",
    "method": (
        "Claims transcribed from the project's own write-ups, each with the strength the write-up gives it: 'strong' "
        "needs several independent methods agreeing, 'moderate' fewer or narrower evidence; pre-registered tests "
        "were frozen before their data existed. Each finding links the section it comes from."),
    "reproduce": "See the linked document section; each one names the script and results file behind it.",
    "docs": ["LAYERS_AND_JOBS.md", "PREREG_MAP.md", "RESEARCH_STATE.md"],
    "caveats": ["Layer bands are bins over a gradient, not hard boundaries."],
}

REQUIRED = {"unit", "kind", "title"}


def load(ctx):
    for f in sorted((HERE / "findings").glob("*.jsonl")):
        for i, r in enumerate(read_jsonl(f)):
            missing = REQUIRED - set(r)
            if missing:
                ctx.warn(f"{f.name}:{i + 1}: missing {sorted(missing)}; skipped")
                continue
            r.setdefault("source", "syntheses")
            r.setdefault("status", "exploratory")
            ctx.add_finding(r)

"""Facts grid (landmarks x relations) and direct entity neurons (86 entities, 333 facts)."""
from __future__ import annotations

from common import RDATA, finding, read_json, slug

SOURCE = {
    "id": "facts",
    "title": "Facts grid and direct entity neurons",
    "kind": "neuron jobs from causal tests",
    "status": "exploratory",
    "date": "2026-10-18",
    "method": (
        "Facts grid: for landmarks x relations (city, country, designer, century, material...), neurons are ranked for "
        "each fact and zeroed; the drop on the fact itself is compared with the drop on the same landmark's other "
        "facts and on other landmarks' same relation. Class: 'entity' (all of one landmark's facts drop), 'relation' "
        "(the same relation across landmarks), 'fact-specific', 'generic'. Entity neurons: for 86 entities "
        "(landmarks, people, works) neurons are ranked by attribution summed only over the entity's NAME tokens, then "
        "zeroed only there; 'own' is the mean drop on the entity's own facts, compared with other entities."),
    "reproduce": "facts.py / facts2.py (research/attention branch; outputs in D:/already-clean-data/facts*)",
    "docs": ["LAYERS_AND_JOBS.md#s4", "LAYERS_AND_JOBS.md#m2", "RESULTS.md"],
    "caveats": ["One question format; 5 relations; landmarks dominate the grid.",
                "Early entity neurons are broad on the input side (median 5% of web firings about their entity)."],
}

CLASS_TITLE = {"entity": "entity neuron", "relation": "relation neuron", "entity+relation": "entity + relation neuron",
               "fact-specific": "fact-specific neuron", "generic": "generic answer machinery"}


def load(ctx):
    from sources import ledger
    if ledger.available():
        ctx.warn("facts: superseded by the ledger export (facts_grid, facts2_entity); not loaded directly")
        return
    p = RDATA / "facts" / "grid_classes.json"
    if p.exists():
        for r in read_json(p):
            ent, rel = r["fact"].split("|", 1)
            ctx.add_finding(finding(
                r["unit"], "fact job", f"{CLASS_TITLE.get(r['class'], r['class'])}: {rel} of {ent}", "facts",
                concept=slug(ent),
                metric={"class": r["class"], "own": r["own"], "same_entity": r.get("same_entity"),
                        "same_relation": r.get("same_relation"), "neither": r.get("neither")},
                detail="drop in log-odds when zeroed: on this fact (own), on the same landmark's other facts, on other "
                       "landmarks' same relation"))
    p = RDATA / "facts2" / "entity_sharing.json"
    if p.exists():
        for r in read_json(p):
            shared = ", ".join(f"{e} ({d:.2f})" for e, d in r.get("shared_with", [])[:4])
            ctx.add_finding(finding(
                r["unit"], "entity neuron", f"entity neuron for {r['entity']} ({r['kind']}), acting at its name",
                "facts", concept=slug(r["entity"]), strength="strong" if r["own"] >= 0.5 else None,
                metric={"own": r["own"], "input_about_entity": r.get("input_about_entity"),
                        "shared_with": r.get("shared_with", [])[:6]},
                detail=(f"also matters for: {shared}" if shared else "specific to this entity among those tested")))

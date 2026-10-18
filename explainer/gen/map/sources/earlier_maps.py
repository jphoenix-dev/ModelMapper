"""Concept maps from before loop v2 (15-17 Oct 2026): CR-2, CR-3, coverage waves 2-4 and the fame-stratified samples."""
from __future__ import annotations

from common import EARLIER, question, read_json

SOURCE = {
    "id": "earlier",
    "title": "Earlier concept maps (CR-2, CR-3, waves 2-4, fame samples, coverage-4)",
    "kind": "concept maps",
    "status": "mixed: CR-3 pre-registered, the rest exploratory",
    "date": "2026-10-17",
    "method": (
        "The same core measurement as loop v2 (recall questions -> attribution top 200 -> each candidate zeroed alone "
        "-> ranked by the drop in the answer's log-odds; top 30 kept), run as rank_save / map_loop before the v2 "
        "additions. No per-concept random-neuron null, no weights-based roles, and head knockouts only for the "
        "later batches. Where an '_exact' file exists its log-odds are exact float64 (ISSUES #35)."),
    "reproduce": "RECALL_VARIANT=V3 CONCEPT_OUT_DIR=<dir> python src/map_loop.py <wave json> <tag>   (or src/rank_save.py)",
    "docs": ["RESULTS.md", "PREREG.md#cr-3", "ISSUES.md"],
    "caveats": ["Files without '_exact' predate the log-odds cap fix (ISSUES #35): drops near saturation may be "
                "understated.",
                "Prompts were written independently of run 1's, so a concept mapped in both is a genuine re-measurement "
                "on new questions."],
}

# (relative rank_save file, map_loop rows file or None, batch label); later entries win for a repeated concept
FILES = [
    ("concepts_v3/rank_save_cr2_exact.json", None, "CR-2"),
    ("concepts_w2/rank_save_w2_exact.json", None, "wave 2"),
    ("concepts_w3/rank_save_w3_exact.json", None, "wave 3"),
    ("concepts_w4/rank_save_w4_exact.json", None, "wave 4"),
    ("concepts_cr3/rank_save_cr3_exact.json", None, "CR-3 (pre-registered)"),
    ("concepts_fame/rank_save_fame_exact.json", None, "fame sample 1"),
    ("concepts_fame2/rank_save_fame2_exact.json", None, "fame sample 2"),
    ("concepts_fame3/rank_save_fame3_exact.json", None, "fame sample 3"),
    ("concepts_fame3/rank_save_fame3m.json", "concepts_fame3/map_loop_fame3m.json", "coverage-3 (map loop)"),
    ("concepts_fame5/rank_save_fame5.json", "concepts_fame5/map_loop_fame5.json", "coverage-4 (map loop)"),
]


def load(ctx):
    maps = {}
    for rel, rows_rel, label in FILES:
        p = EARLIER / rel
        if not p.exists():
            ctx.warn(f"earlier maps: missing {p}")
            continue
        ranks = read_json(p)
        rows = read_json(EARLIER / rows_rel) if rows_rel and (EARLIER / rows_rel).exists() else {}
        for concept, rank in ranks.items():
            if not rank or not rank.get("top"):
                continue
            row = rows.get(concept, {})
            m = {"source": "earlier", "batch": label, "file": rel, "concept": concept,
                 "answer": row.get("answer_form") or row.get("answer") or concept,   # rank_save keeps only the first token
                 "status": "mapped", "recall": rank.get("recall"),
                 "meta": {"category": row.get("category")} if row.get("category") else {},
                 "prompts": [question(q) for q in (row.get("recall_prompts") or rank.get("prompts") or [])[:6]],
                 "baseline": rank.get("baseline_mean_logodds"), "k_mean": rank.get("k_mean"),
                 "k_median": row.get("k_median"), "k_flip": rank.get("k_flip"),
                 "ks": [1, 2, 3, 5, 10, 20, 50] if row.get("drops") else None,
                 "drops": row.get("drops"), "random_drops": row.get("random_drops"),
                 "top": [{"n": t["neuron"], "drop": t.get("single_drop"), "attr_rank": t.get("attr_rank"),
                          "act": t.get("act_answer_pos")} for t in rank["top"]],
                 "heads": sorted(([h, d] for h, d in (rank.get("head_knockout_mean") or {}).items()),
                                 key=lambda x: -x[1])}
            maps[concept] = m
    for m in maps.values():
        ctx.add_map(m)

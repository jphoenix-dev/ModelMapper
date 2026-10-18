"""Loop v2, run 1: 5,000 Vital Articles concepts mapped on 8 rented RTX 3090s + the local 5090 (17-18 Oct 2026)."""
from __future__ import annotations

import json

from common import RUN1, FRAME, question, read_json, read_jsonl

SOURCE = {
    "id": "run1",
    "title": "Loop v2, run 1 (5,000 Vital Articles concepts)",
    "kind": "concept maps",
    "status": "exploratory",
    "date": "2026-10-18",
    "method": (
        "For each concept, GPT-6 Luna writes ~10 questions whose answer is the concept's name; questions Qwen3-8B "
        "recalls (P >= 0.2, leak-filtered) are kept. Every one of the 442,368 MLP neurons is ranked by attribution "
        "(activation x gradient of the answer's log-odds); the top 200 are each zeroed alone at every position and "
        "re-ranked by the measured drop in the answer's exact log-odds. The top 30 are the map. Controls in the same "
        "run: 2 random same-layer neurons zeroed alone (a per-concept noise floor) and a random-neuron k curve. The 50 "
        "most-attributed attention heads are zeroed one at a time. Roles come from the weights: the logit-lens push of "
        "each neuron's output vector ('say' = answer in its top-10 tokens; 'attribute' = answer z >= 2; 'early' = "
        "layer < 20)."),
    "reproduce": "python src/dist/make_run.py --kind map ... ; python src/dist/worker.py --store <store>  "
                 "(research/distributed branch; gate: src/loop_v2/gate_repro.py reproduces v1 bitwise)",
    "docs": ["LOOP_V2.md", "PREREG_MAP.md", "RESEARCH_STATE.md"],
    "caveats": [
        "Task is one behaviour: answering 'What is ...? Answer with just the name.' Neurons can matter here and do "
        "other jobs elsewhere.",
        "Ranking by effect on the answer favours late neurons (LAYERS_AND_JOBS, cautions).",
        "bf16: a single-neuron effect near the per-concept null is noise; read drops against the null.",
        "A concept's exact top-10 depends on the questions (about 5/10 overlap between prompt sets); shared parts are "
        "stable (P1, P3).",
    ],
}

KS = [1, 2, 3, 5, 10, 20, 50]


def _frame():
    meta = {}
    try:
        fr = read_json(FRAME / "frame.json")
        types = read_json(FRAME / "types.json")
        p31, labels = types.get("p31", {}), types.get("labels", {})
        try:
            wd = read_json(FRAME / "vital.json").get("wd", {})
        except FileNotFoundError:
            wd = {}
        for d in fr["draw"]:
            q = d.get("qid")
            t = [labels.get(x) for x in (p31.get(q) or (wd.get(q) or {}).get("p31") or [])]
            meta[d["concept"]] = {"section": d.get("section"), "tier": d.get("tier"), "kind": d.get("kind"),
                                  "qid": d.get("qid"), "links": d.get("links"),
                                  "type": next((x for x in t if x), None)}
    except FileNotFoundError:
        pass
    return meta


def load(ctx):
    meta = _frame()
    batches = sorted((RUN1 / "results" / "results" / "map").glob("b*/records.jsonl"))
    for bf in batches:
        for rec in read_jsonl(bf):
            row, rank, v2 = rec["row"], rec.get("rank"), rec.get("v2") or {}
            m = {"source": "run1", "concept": rec["concept"], "answer": row.get("answer_form") or row.get("answer"),
                 "status": "mapped" if rank else ("not recalled" if row.get("status") != "ok" else row.get("status")),
                 "recall": row.get("recall"), "meta": meta.get(rec["concept"], {}),
                 "prompts": [question(p) for p in row.get("recall_prompts", [])[:6]],
                 "prov": v2.get("provenance"), "arrays": str(bf.parent / "arrays" / v2["arrays"]) if v2.get("arrays") else None}
            if rank:
                roles = {r["neuron"]: r for r in v2.get("roles_top30", [])}
                m.update({
                    "baseline": rank.get("baseline_mean_logodds"), "k_mean": rank.get("k_mean"),
                    "k_median": row.get("k_median"), "k_flip": rank.get("k_flip"),
                    "ks": KS, "drops": row.get("drops"), "random_drops": row.get("random_drops"),
                    "answer_ids": rank.get("answer_ids"),
                    "top": [{"n": t["neuron"], "drop": t["single_drop"], "attr_rank": t.get("attr_rank"),
                             "act": t.get("act_answer_pos"), "role": roles.get(t["neuron"], {}).get("role"),
                             "answer_rank": roles.get(t["neuron"], {}).get("answer_rank"),
                             "answer_z": roles.get(t["neuron"], {}).get("answer_z"),
                             "jobs": roles.get(t["neuron"], {}).get("job_tokens", [])[:12]} for t in rank["top"]],
                    "null": [[n, round(sum(v) / len(v), 4)] for n, v in (v2.get("null_single_drops") or {}).items() if v],
                    "heads": sorted(([h, d] for h, d in (rank.get("head_knockout_mean") or {}).items()),
                                    key=lambda x: -x[1])})
            ctx.add_map(m)
    # run-level checks recorded next to the run
    p1 = RUN1 / "p1_check.txt"
    if p1.exists():
        ctx.add_finding({"unit": "model", "kind": "generalisation check", "source": "run1", "status": "exploratory",
                         "title": "Run 1's top-5 neurons break recall on independent Wikipedia cloze prompts",
                         "detail": p1.read_text(encoding="utf-8").strip()[:1200],
                         "docs": ["PREREG_MAP.md#p1---generalisation-to-independent-prompts"]})

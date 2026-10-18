"""Search-then-verify pilot (search_verify.py) -> ledger pack.

One 'search_verify' row per neuron (350): unit = the neuron, concept = the concept Luna proposed from its web firings +
pushed tokens (None when Luna named none), value = mean exact log-odds drop when zeroed everywhere on the recalled
questions (None if untested), value_text = status (tested / not_recalled / no_concept / bad_reply), context = group.
noise_floor = pooled 99th percentile of the random same-layer null drops (recomputed from verify.jsonl), threshold =
max(1.0, that), passes = drop >= threshold (the script's 'verified' rule; None if untested). details = Luna's reply and
prompt, answer form, per-question probabilities, base log-odds and drops, the null neurons, and the neuron's
concept-search score / percentile (scores.npz). Each null neuron gets a 'null_knockout_drop' row.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ledger.core import DATA_ROOT, PackWriter, concept_id, read_jsonl  # noqa: E402

SV = DATA_ROOT / "search_verify"
VER, LUNA, SUMT = SV / "verify.jsonl", SV / "luna.json", SV / "summary.txt"
SCORES = DATA_ROOT / "concept_search" / "scores.npz"
EXP = "search_verify_pilot"
DOC = "RESULTS.md#17 Oct 2026 (late evening): search-then-verify pilot - verification works, proposing from web text doesn't (EXPLORATORY; research branch)"
GROUPS = {"top": "100 random from the top 1% by concept-search score, in no concept's top-30",
          "random": "100 layer-matched neurons in no top-30", "size": "100 size-matched (same layer, closest corpus peak) in no top-30",
          "known": "50 positive controls: new coverage-4 top-5 neurons (in no earlier top-30)"}


def main(force=False):
    with PackWriter("search_verify", inputs=[VER, LUNA, SUMT, SCORES], code=[__file__], force=force) as pw:
        if pw.skip:
            print("search_verify: unchanged, skipped"); return
        R = list(read_jsonl(VER)); L = json.loads(LUNA.read_text(encoding="utf-8"))
        sc = np.load(SCORES); score, pct = sc["score"], sc["pct"]
        nulls = [v for r in R if r["status"] == "tested" for v in r["null_drop_means"]]
        p99 = float(np.percentile(nulls, 99)); thr = max(1.0, p99)
        pw.experiment(EXP, title="Search-then-verify pilot: Luna proposes a concept from web firings, Qwen verifies by knockout",
                      method=("350 neurons, shuffled and unlabelled: 100 top-ranked by the concept-search score but in no map, 100 "
                              "layer-matched random, 100 size-matched, 50 known concept neurons. Luna reads each neuron's 15 strongest "
                              "positive and negative web firings and its 20 most-pushed tokens each way and names one specific concept "
                              "(with 8 quiz questions) or none. With >= 3 questions recalled (P >= MIN_P, leak filter), the neuron is "
                              "zeroed at every position and the mean drop in exact log-odds compared with 2 random same-layer neurons."),
                      status="pre-registered", date="2026-10-17", script="src/search_verify.py", reproduce="python src/search_verify.py",
                      data_files=["data:search_verify/verify.jsonl", "data:search_verify/luna.json", "data:search_verify/summary.txt"],
                      caveats=["Criterion NOT MET (top 0 vs random 0 verified), so the pilot cannot judge the ranking.",
                               "The proposal step fails: Luna found the concept for only 4 of 50 known concept neurons.",
                               "Pooled null from only 30 random-neuron drops."],
                      docs=[DOC],
                      params={"groups": GROUPS, "min_recalled": 3, "verified_rule": "drop >= 1.0 and above the pooled null's 99th percentile",
                              "noise_floor": {"rule": "pooled 99th pct of null drop means", "value": p99, "n": len(nulls), "max": max(nulls)},
                              "threshold": thr, "criterion": "verified rate top >= 3x random", "criterion_outcome": "NOT MET (0 vs 0)",
                              "seeds": {"select": 51, "shuffle": 52, "nulls": 53}, "summary_text": SUMT.read_text(encoding="utf-8")})
        for i, r in enumerate(R):
            u = r["neuron"]; lu = L.get(u, {}); rep = lu.get("reply") or {}
            Lr, n_ = (int(x) for x in u[1:].split(":")); x = Lr * 12288 + n_
            cname = r.get("concept")
            cid = pw.concept(concept_id("thing", cname), cname, "thing", aliases=[rep.get("answer")]) if cname else None
            tested = r["status"] == "tested"
            det = {k: v for k, v in r.items() if k not in ("neuron", "group", "concept", "status", "drop_mean")}
            det.update({"luna_reply": rep, "luna_prompt": lu.get("prompt"), "concept_search_score": float(score[x]),
                        "concept_search_layer_pct": float(pct[x])})
            src = f"search_verify/verify.jsonl#{i}"
            pw.evidence(experiment=EXP, unit=u, concept=cid, evidence_type="search_verify", metric="mean_logodds_drop",
                        value=r.get("drop_mean"), value_text=r["status"], uom="logodds", n=r.get("recalled") if tested else None,
                        noise_floor=p99, threshold=thr, passes=(r["drop_mean"] >= thr) if tested else None,
                        context=f"group={r['group']};proposal=luna_web_firings;prompts=luna_quiz;positions=all", details=det, source=src)
            for j, (nu, nd) in enumerate(zip(r.get("null_neurons") or [], r.get("null_drop_means") or [])):
                pw.evidence(experiment=EXP, unit=nu, concept=cid, evidence_type="null_knockout_drop", metric="mean_logodds_drop", value=nd,
                            uom="logodds", n=r.get("recalled"), noise_floor=p99,
                            context=f"group={r['group']};control=random_same_layer;for={u}", source=f"{src}.null[{j}]")


if __name__ == "__main__":
    main(force="--force" in sys.argv)

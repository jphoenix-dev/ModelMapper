"""H9: answer-format neurons - are the widely shared neurons the format-bound ones? (MECHANISMS.md H9) Ledger only (8B).

Data: loop v3 enrich smoke test, stage E2 (50 concepts): activation at the answer position of each concept's 200 candidates
in three instructions - F1 ' Answer with just the name.', F2 ' Answer in as few words as possible.', F3 ' Answer in one
short sentence.'.
Per (unit, concept) where the unit is active (max |activation| over F1/F3 >= 1): format shift
  fb = (a_F1 - a_F3) / (|a_F1| + |a_F3|)        in [-1, 1]; 0 = same in both formats
Noise floor: the same with F2 in place of F3 (F1 and F2 are near-identical short-answer instructions).
Per unit seen in >= 5 concepts: median fb and sign consistency. Format-bound = |median fb| >= 0.5 and >= 80% of its
concepts on the same side. Sharing = n_users from loop_v2_shared_neurons (run 1: # concepts whose map uses it).
Prediction (MECHANISMS.md, before running): shared units (n_users >= 50) are format-bound more often than less-shared
units (n_users < 10 or absent): rate ratio >= 2; and |fb| for F1 vs F3 exceeds the F1 vs F2 floor overall.
Output: D:/already-clean-data/mechanisms/format_bound.json
"""
import json
import statistics as stx
import sys
from collections import defaultdict
from pathlib import Path

from modelmapper.experiments.mechanisms.common import save  # noqa: E402
from modelmapper.ledger.query import connect  # noqa: E402


def main():
    con = connect()
    rows = con.execute("""SELECT unit_key, concept_id, context, value FROM evidence
                          WHERE experiment_id='loop_v3_enrich' AND evidence_type='activation' AND unit_kind='neuron'""").fetchall()
    act = defaultdict(dict)
    for u, c, ctx, v in rows:
        act[(u, c)][ctx.split("format=")[1]] = v
    users = dict(con.execute("SELECT unit_key, value FROM evidence WHERE experiment_id='loop_v2_shared_neurons' AND metric='n_users'").fetchall())
    per_unit = defaultdict(lambda: {"f3": [], "f2": []})
    pairs_f3, pairs_f2 = [], []
    for (u, c), a in act.items():
        if not all(k in a for k in ("F1", "F2", "F3")):
            continue
        if max(abs(a["F1"]), abs(a["F3"]), abs(a["F2"])) < 1:
            continue
        f3 = (a["F1"] - a["F3"]) / (abs(a["F1"]) + abs(a["F3"]) + 1e-9)
        f2 = (a["F1"] - a["F2"]) / (abs(a["F1"]) + abs(a["F2"]) + 1e-9)
        per_unit[u]["f3"].append(f3); per_unit[u]["f2"].append(f2); pairs_f3.append(abs(f3)); pairs_f2.append(abs(f2))
    out, groups = [], {"shared_ge50": [], "mid_10_49": [], "low_lt10_or_none": []}
    for u, d in per_unit.items():
        if len(d["f3"]) < 5:
            continue
        m = stx.median(d["f3"]); same = sum(1 for x in d["f3"] if (x > 0) == (m > 0)) / len(d["f3"])
        m2 = stx.median(d["f2"]); same2 = sum(1 for x in d["f2"] if (x > 0) == (m2 > 0)) / len(d["f2"])
        fbnd = abs(m) >= 0.5 and same >= 0.8
        fbnd2 = abs(m2) >= 0.5 and same2 >= 0.8
        nu = users.get(u, 0)
        g = "shared_ge50" if nu >= 50 else "mid_10_49" if nu >= 10 else "low_lt10_or_none"
        rec = {"unit": u, "n_concepts": len(d["f3"]), "n_users_run1": nu, "median_fb_F1_F3": m, "same_side": same,
               "format_bound": fbnd, "median_fb_F1_F2": m2, "format_bound_F2_control": fbnd2}
        out.append(rec); groups[g].append(rec)
    S = {"n_pairs": len(pairs_f3), "median_abs_fb_F1_F3": stx.median(pairs_f3), "median_abs_fb_F1_F2_floor": stx.median(pairs_f2),
         "n_units_ge5_concepts": len(out)}
    for g, rs in groups.items():
        S[g] = {"n_units": len(rs), "frac_format_bound": (sum(r["format_bound"] for r in rs) / len(rs)) if rs else None,
                "frac_format_bound_F2_control": (sum(r["format_bound_F2_control"] for r in rs) / len(rs)) if rs else None,
                "frac_fires_more_in_short_answer": (sum(1 for r in rs if r["format_bound"] and r["median_fb_F1_F3"] > 0) / len(rs)) if rs else None}
    lo = S["low_lt10_or_none"]["frac_format_bound"] or 0
    S["ratio_shared_vs_low"] = (S["shared_ge50"]["frac_format_bound"] / lo) if lo and S["shared_ge50"]["frac_format_bound"] is not None else None
    S["prediction_pass"] = bool(S["ratio_shared_vs_low"] and S["ratio_shared_vs_low"] >= 2 and S["median_abs_fb_F1_F3"] > S["median_abs_fb_F1_F2_floor"])
    S["top_format_bound_shared"] = sorted([r for r in groups["shared_ge50"] if r["format_bound"]], key=lambda r: -r["n_users_run1"])[:20]
    save("format_bound.json", {"summary": S, "units": out})
    print(json.dumps({k: v for k, v in S.items() if k != "top_format_bound_shared"}, indent=1))
    for r in S["top_format_bound_shared"][:10]:
        print(r)


if __name__ == "__main__":
    main()

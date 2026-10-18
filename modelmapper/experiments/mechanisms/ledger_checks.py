"""Free checks on Qwen3-8B data already in the ledger (read-only), for MECHANISMS.md H1 and H5.

H1-8B  Among run 1's knockouts (top 200 by POSITIVE attribution per concept; zeroed alone, all positions, mean exact
       log-odds drop over recall prompts), how many RAISE the answer beyond the null? Null = run 1's own random-neuron
       knockouts (2 per concept, same procedure). Inhibitor-like = drop < the null's 0.1st percentile (and < -0.5 as a
       stricter cut). Per concept, per layer, per role; units that do it for many concepts.
       Prediction (stated before running): such 'sign-flip' neurons are rare among positive-attribution candidates
       (< 1% of knockouts beyond the strict cut) but present in >= 20% of concepts, concentrated in layers 28-35.
H5-8B  Massive activations: the largest |activation| recorded per unit in era_formats and loop_v3_enrich (all prompts are
       chat-formatted questions), vs the median unit. Prediction: a handful of units (<= 10) exceed 20x the median unit's
       max, and they are among the shared 'always-on' units.
Output: D:/already-clean-data/mechanisms/ledger_checks.json
"""
import sys
from pathlib import Path

from modelmapper.experiments.mechanisms.common import save  # noqa: E402
from modelmapper.ledger.query import connect  # noqa: E402


def main():
    con = connect(); q = lambda s: con.execute(s).fetchall()
    S = {}
    null = q("""SELECT quantile_cont(value, [0.001, 0.01, 0.5, 0.99, 0.999]), min(value), count(*) FROM evidence
                WHERE experiment_id='loop_v2_run1' AND evidence_type='null_knockout_drop'""")[0]
    p001 = null[0][0]
    S["null"] = {"quantiles_0.001_0.01_0.5_0.99_0.999": null[0], "min": null[1], "n": null[2]}
    base = "experiment_id='loop_v2_run1' AND evidence_type='knockout_drop'"
    S["n_knockouts"] = q(f"SELECT count(*) FROM evidence WHERE {base}")[0][0]
    for name, cut in (("beyond_null_p0.1", p001), ("strict_lt_-0.5", -0.5)):
        n, nc = q(f"SELECT count(*), count(DISTINCT concept_id) FROM evidence WHERE {base} AND value < {cut}")[0]
        S[name] = {"cut": cut, "n_knockouts": n, "frac_knockouts": n / S["n_knockouts"], "n_concepts": nc,
                   "frac_concepts": nc / q(f"SELECT count(DISTINCT concept_id) FROM evidence WHERE {base}")[0][0],
                   "by_layer": q(f"SELECT layer, count(*) FROM evidence WHERE {base} AND value < {cut} GROUP BY 1 ORDER BY 1"),
                   "attr_rank_quartiles": q(f"""SELECT quantile_cont(CAST(json_extract(details, '$.attr_rank') AS INT), [0.25, 0.5, 0.75])
                                                FROM evidence WHERE {base} AND value < {cut}""")[0][0],
                   "by_role": q(f"""SELECT r.value_text, count(*) FROM evidence e JOIN evidence r
                                    ON r.experiment_id='loop_v2_run1' AND r.evidence_type='role' AND r.unit_id=e.unit_id AND r.concept_id=e.concept_id
                                    WHERE e.{base.replace(' AND ', ' AND e.')} AND e.value < {cut} GROUP BY 1"""),
                   "shared_units": q(f"""SELECT unit_key, count(DISTINCT concept_id) n, round(median(value), 3)
                                         FROM evidence WHERE {base} AND value < {cut} GROUP BY 1 ORDER BY 2 DESC LIMIT 15"""),
                   "biggest": q(f"""SELECT e.unit_key, c.label, round(e.value, 3) FROM evidence e JOIN concepts c ON c.concept_id=e.concept_id
                                    WHERE e.{base.replace(' AND ', ' AND e.')} AND e.value < {cut} ORDER BY e.value LIMIT 15""")}
    # for comparison: how often a candidate helps beyond the null's 99.9th percentile
    S["helpers_beyond_null_p99.9"] = q(f"SELECT count(*) FROM evidence WHERE {base} AND value > {null[0][4]}")[0][0]
    for exp, met in (("era_formats", "activation"), ("loop_v3_enrich", "activation_answer_position")):
        rows = q(f"""SELECT unit_key, max(abs(value)) m FROM evidence WHERE experiment_id='{exp}' AND metric='{met}'
                     GROUP BY 1 ORDER BY 2 DESC""")
        med = sorted(r[1] for r in rows)[len(rows) // 2]
        S[f"massive_{exp}"] = {"n_units": len(rows), "median_unit_max_abs": med, "top15": rows[:15],
                               "n_over_20x_median": sum(1 for r in rows if r[1] > 20 * med)}
    save("ledger_checks.json", S)
    for k, v in S.items():
        print(k, v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk not in ("by_layer",)})


if __name__ == "__main__":
    main()

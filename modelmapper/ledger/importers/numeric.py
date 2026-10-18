"""Numeric facts (numeric.py) + magnitude dose series (numeric_mag.py) -> ledger pack.

Facts: concept = the entity, context 'relation=<year|height|...>;answer=<number>', concept2 = the century for years.
Includes the two invalid first runs (log-prob saturation; rounded dose log-probs) as separate experiments, flagged.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ledger.core import DATA_ROOT, PackWriter, era_century, read_jsonl, thing  # noqa: E402

D = DATA_ROOT / "numeric"
DOC_N = "RESULTS.md#18 Oct 2026 - Numeric facts: era neurons, few digit neurons, no magnitude coding shown (numeric.py)"
DOC_M = "RESULTS.md#18 Oct 2026 - Magnitude: no number-line neurons; the number line is in the output geometry (numeric_mag.py)"
THR = 1.0


def century_of(ans):
    return (int(ans) - 1) // 100 + 1 if ans.isdigit() and len(ans) == 4 else None


def main(force=False):
    names = ["entities.json", "facts.json", "maps.jsonl", "maps_logprob.jsonl", "share.jsonl", "share_units.json", "share_nulls.json",
             "magnitude.jsonl", "position.jsonl", "summary_rows.json", "mag_uncertain.json", "mag_rows.json", "mag_dose.jsonl", "mag_dose_rounded.jsonl"]
    files = [D / f for f in names]
    with PackWriter("numeric_facts", inputs=files, code=[__file__], force=force) as pw:
        if pw.skip:
            print("numeric: unchanged, skipped"); return
        S = list(read_jsonl(D / "share.jsonl")); nulls = json.loads((D / "share_nulls.json").read_text())
        nv = sorted(abs(sum(s["per_pos"][u])) for s in S for u in nulls if u in s["per_pos"]); q99 = nv[int(0.99 * len(nv))]
        common = dict(date="2026-10-18", docs=[DOC_N, "LAYERS_AND_JOBS.md#moderately-supported"])
        pw.experiment("numeric_facts", title="Numeric facts: years and heights, digit by digit",
                      method=("86 numeric facts (landmark years and heights, birth years, works' years) that Qwen recalls. Each digit is "
                              "scored teacher-forced by exact float64 log-odds, summed over the number. Attribution of that sum over all "
                              "positions -> top 100 each zeroed alone -> top 30 with per-digit drops. Each fact's top-3 zeroed on every fact "
                              "(sharing by digit / prefix), zeroed vs doubled (expected-digit shift), and zeroed only at the name tokens vs "
                              "only at the answer positions. Noise floor: 24 random neurons zeroed on every fact."),
                      status="pre-registered predictions (P1 held; P2 mostly failed - era neurons the exception; P3 not shown)",
                      script="src/loop_v2/numeric.py", reproduce="python src/loop_v2/numeric.py entities|build|map|share|magnitude|position|summary",
                      data_files=[f"data:numeric/{n}" for n in names[:10]],
                      caveats=["Qwen is > 0.9999 sure of 87% of digits: log-prob drops saturate, hence log-odds.", "Numbers from Luna; recall-filtered."],
                      params={"threshold": THR, "noise_floor": {"rule": "99th pct |summed drop| of 24 random neurons on every fact", "value": q99}}, **common)
        pw.experiment("numeric_facts_logprob_invalid", title="Numeric facts, first map (INVALID: log-prob saturation)",
                      method="The first numeric map measured log-probability drops, which cannot move when Qwen is near-certain; kept for the record only.",
                      status="invalid (superseded by numeric_facts)", script="src/loop_v2/numeric.py (before the log-odds fix)",
                      reproduce="n/a", data_files=["data:numeric/maps_logprob.jsonl"], caveats=["Do not use: drops of 0.000 are a measurement artefact."], **common)
        pw.experiment("numeric_magnitude", title="Magnitude: dose series and less-certain numbers",
                      method=("Each numeric fact's top-3 neurons (and 3 random layer-matched controls) scaled x -4, -2, -1, 0, 0.5, 1, 2, 4 at every "
                              "position, on the 86 facts and on 43 numbers Qwen is unsure of (its own greedy number as target); the full float64 "
                              "distribution over the 10 digit tokens recorded. Where lost probability goes (neighbouring digits?), whether the "
                              "expected digit moves gradedly, Qwen's own uncertainty, and the digit tokens' unembedding geometry."),
                      status="pre-registered predictions (A1 failed as a neuron property; A2 held but is push-pull for the neuron's own digit; A3 held modestly)",
                      date="2026-10-18", script="src/loop_v2/numeric_mag.py", reproduce="python src/loop_v2/numeric_mag.py uncertain|dose|summary",
                      data_files=["data:numeric/mag_uncertain.json", "data:numeric/mag_dose.jsonl", "data:numeric/mag_rows.json"],
                      caveats=["The first dose run rounded log-probs to 6 decimals (zeroing near-certain digits); kept as mag_dose_rounded.jsonl, invalid."],
                      docs=[DOC_M, "LAYERS_AND_JOBS.md#moderately-supported"], params={"factors": [-4, -2, -1, 0, 0.5, 1, 2, 4], "graded_rule": "|spearman| >= 0.9 and range >= 0.3 digits"})
        F = json.loads((D / "facts.json").read_text(encoding="utf-8")); ans = {(f["entity"], f["relation"]): f["answer"] for f in F}
        ent = {}

        def fact(e, r, kind=None):
            if e not in ent:
                ent[e] = thing(pw, e, entity_kind=kind)
            a = ans.get((e, r)); c = century_of(a) if a else None
            return ent[e], (era_century(pw, c) if c and r == "year" else None), f"relation={r}" + (f";answer={a}" if a else "")
        for f in F:
            fact(f["entity"], f["relation"], f["kind"])
        for exp, fn, ok in (("numeric_facts", "maps.jsonl", True), ("numeric_facts_logprob_invalid", "maps_logprob.jsonl", False)):
            for i, m in enumerate(read_jsonl(D / fn)):
                c, c2, ctx = fact(m["entity"], m["relation"])
                for k, x in enumerate(m["top"]):
                    pw.evidence(experiment=exp, unit=x["neuron"], concept=c, concept2=c2, context=ctx + ";test=own_map", evidence_type="knockout_drop",
                                metric="summed_digit_logodds_drop" if ok else "summed_digit_logprob_drop_INVALID", value=x["full"], uom="logodds" if ok else "logprob",
                                noise_floor=q99 if ok else None, threshold=THR if ok else None, passes=(x["full"] > THR) if ok else None, rank=k + 1,
                                details={"per_pos": x["per_pos"]}, source=f"numeric/{fn}#{i}")
        for i, s in enumerate(S):
            c, c2, ctx = fact(s["entity"], s["relation"])
            for u, pp in s["per_pos"].items():
                v = sum(pp)
                pw.evidence(experiment="numeric_facts", unit=u, concept=c, concept2=c2, context=ctx + ";test=share" + (";control=random" if u in nulls else ""),
                            evidence_type="knockout_drop", metric="summed_digit_logodds_drop", value=v, uom="logodds", noise_floor=q99, threshold=THR,
                            passes=v > THR, details={"per_pos": pp}, source=f"numeric/share.jsonl#{i}")
        for i, p in enumerate(read_jsonl(D / "position.jsonl")):
            c, _, ctx = fact(p["entity"], p["relation"])
            for pos, v in (("all", p["drop_all"]), ("name_tokens", p["drop_subject_only"]), ("answer_positions", p["drop_answer_only"])):
                pw.evidence(experiment="numeric_facts", unit=p["neuron"], concept=c, context=ctx + f";positions={pos}", evidence_type="position_split",
                            metric="summed_digit_logodds_drop", value=v, uom="logodds", noise_floor=q99, source=f"numeric/position.jsonl#{i}")
        for i, g in enumerate(read_jsonl(D / "magnitude.jsonl")):
            c, _, ctx = fact(g["entity"], g["relation"])
            for met in ("exp_shift_zero", "exp_shift_double", "corr_drop_zero", "corr_drop_double"):
                pw.evidence(experiment="numeric_facts", unit=g["neuron"], concept=c, context=ctx + ";test=zero_vs_double", evidence_type="dose_response",
                            metric=met, value=max(g[met], key=abs), uom="digits" if met.startswith("exp") else "logodds", details={"per_pos": g[met]},
                            source=f"numeric/magnitude.jsonl#{i}")
        for i, r in enumerate(json.loads((D / "summary_rows.json").read_text(encoding="utf-8"))):
            if r["n"] < 3:
                pw.evidence(experiment="numeric_facts", unit=r["u"], evidence_type="job_class", metric="facts_hurt", value=r["n"], value_text="specific",
                            details=r, source=f"numeric/summary_rows.json#{i}")
                continue
            digit = r["posdigit"] >= 0.8 and r["posdigit"] > r["posdigit_null"] + 0.3
            prefix = r.get("prefix3", 0) >= 0.8 and r.get("prefix3", 0) > r.get("prefix3_null", 1) + 0.3
            cls = "digit" if digit else "prefix" if prefix else "broad" if r["n"] >= 20 else "shared"
            n_, q, d = r["top_digit"]
            dc = pw.concept(f"digit:{d}", str(d), "digit") if digit else None
            pw.evidence(experiment="numeric_facts", unit=r["u"], concept=dc, context=f"digit_position={q};number_length={n_}" if digit else None,
                        evidence_type="job_class", metric="same_digit_same_position_purity", value=r["posdigit"], value_text=cls, n=r["n"],
                        noise_floor=r["posdigit_null"], threshold=0.8, passes=digit, details=r, source=f"numeric/summary_rows.json#{i}")
        for i, u in enumerate(json.loads((D / "mag_uncertain.json").read_text(encoding="utf-8"))):
            c = ent.get(u["entity"]) or thing(pw, u["entity"], entity_kind=u["kind"]); ent[u["entity"]] = c
            pw.evidence(experiment="numeric_magnitude", concept=c, context=f"relation={u['relation']};set=uncertain", evidence_type="concept_stat",
                        metric="min_p_top_digit", value=min(min(x) for x in u["p_top_digit"]), n=len(u["prompts"]),
                        details={"greedy": u["greedy"], "p_top_digit": u["p_top_digit"]}, source=f"numeric/mag_uncertain.json#{i}")
        for i, r in enumerate(json.loads((D / "mag_rows.json").read_text(encoding="utf-8"))):
            c = ent.get(r["entity"]) or thing(pw, r["entity"]); ent[r["entity"]] = c
            ctx = f"relation={r['relation']};set={r['group']};digit_position={r['q']}" + (";control=random" if r["type"] == "random" else "")
            k = f"{i}"
            pw.evidence(experiment="numeric_magnitude", unit=r["neuron"], concept=c, context=ctx, evidence_type="dose_response", metric="spearman_dose_vs_expected_digit",
                        value=r["rho"], threshold=0.9, passes=r["graded"], details={"E": r["E"], "E_range": r["E_range"], "drop0": r["drop0"], "c_mean": r.get("c_mean")},
                        key=f"dose|{k}", source=f"numeric/mag_rows.json#{i}")
            for fac in ("0.0", "-2.0", "-4.0"):
                if f"nb_share_{fac}" in r:
                    pw.evidence(experiment="numeric_magnitude", unit=r["neuron"], concept=c, context=ctx + f";factor={fac}", evidence_type="digit_shift",
                                metric="neighbour_share_of_lost_probability", value=r[f"nb_share_{fac}"], noise_floor=r[f"nb_uniform_{fac}"],
                                details={"mass_weighted_distance": r[f"dist_{fac}"], "top_gainer_distance": r[f"topgain_dist_{fac}"]},
                                key=f"shift|{k}|{fac}", source=f"numeric/mag_rows.json#{i}")
        pw.artifact(D / "mag_dose.jsonl", experiment="numeric_magnitude", fmt="jsonl", description="full float64 log-probs of the 10 digit tokens per factor x prompt x position")
        pw.artifact(D / "mag_dose_rounded.jsonl", experiment="numeric_magnitude", fmt="jsonl", description="INVALID first dose run (log-probs rounded to 6 decimals)")


if __name__ == "__main__":
    main(force="--force" in sys.argv)

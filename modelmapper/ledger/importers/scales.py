"""Two-ended neurons (map session's SCALES.md; steps 1-3 Luna / Jev, step 4 causal steering here) -> ledger pack `scales`."""
import json
import os
import statistics as stx
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ledger.core import DATA_ROOT, PackWriter, read_jsonl, thing  # noqa: E402

SC = DATA_ROOT / "scales"
SPEC = Path(os.environ.get("MAP_WORKTREE", "C:/dev/already-clean-map")) / "explainer/gen/map/SCALES.md"


def main(force=False):
    files = [SC / f for f in ("scales.cache.jsonl", "verdicts.json", "summary.json", "steer_candidates.json", "steer.jsonl",
                              "steer_summary.txt", "steer_allpos_invalid.jsonl")] + [SPEC]
    with PackWriter("scales", inputs=[f for f in files if f.exists()], code=[__file__], force=force) as pw:
        if pw.skip:
            print("scales: unchanged, skipped"); return
        summ = json.loads((SC / "summary.json").read_text(encoding="utf-8"))
        base = dict(date="2026-10-18", docs=["SCALES.md (map worktree)", "RESULTS.md (18 Oct, two-ended neurons)"])
        pw.experiment("scales", title="Two-ended neurons: one scale (north <-> south) or two jobs? Steps 1-3 (Luna / Jev)",
                      method=SPEC.read_text(encoding="utf-8") if SPEC.exists() else "See SCALES.md.", status="exploratory (pass rule fixed before the run)",
                      status_class="exploratory", script="explainer/gen/map/scales.py (map worktree)", reproduce="see SCALES.md",
                      data_files=["data:scales/scales.cache.jsonl", "data:scales/verdicts.json", "data:scales/summary.json"],
                      caveats=["Descriptions are written from weight-readout words only; a label for reading, not the neuron's true meaning."],
                      params={"summary": summ}, **base)
        pw.experiment("scales_steer", title="Two-ended neurons, step 4: push the neuron towards each end at the answer position",
                      method=("For 24 candidates (4 strict scales, 10 near-scales, 10 'unrelated'): one concept per end (the usage-check concepts), "
                              "each end's sign from run 1; neuron activation + s x k x peak (k 1, 2, 4) at the answer position only; change in each "
                              "concept's exact answer log-odds; a random same-layer neuron at its own peak as the control. Pass: towards A raises A "
                              "and lowers B by >= 0.5 and the reverse, control quiet (< 0.5) at that k."),
                      status="pre-registered predictions P1-P4 (P1-P3 held; P4 failed: controls move at k >= 2)", status_class="pre-registered",
                      script="src/loop_v2/scales_steer.py", reproduce="python src/loop_v2/scales_steer.py",
                      data_files=["data:scales/steer.jsonl", "data:scales/steer_summary.txt"],
                      caveats=["Only k = 1 is usable: a random neuron pushed at 2x or 4x its peak moves answers too (controls quiet 17/24 at k1, 4/24 at k2, 1/24 at k4).",
                               "Lowering effects are larger than raising effects."],
                      params={"summary": (SC / "steer_summary.txt").read_text(encoding="utf-8")}, **base)
        pw.experiment("scales_steer_allpos_invalid", title="Two-ended neurons, step 4 first run (INVALID: push at every position)",
                      method="As scales_steer but the push was added at every position; the random control moved the answers in 65/72 cases.",
                      status="invalid (superseded by scales_steer)", status_class="invalid", script="src/loop_v2/scales_steer.py (first version)",
                      reproduce="n/a", data_files=["data:scales/steer_allpos_invalid.jsonl"], caveats=["Do not use."], **base)
        cache = {}
        for r in read_jsonl(SC / "scales.cache.jsonl"):
            cache[r["k"]] = r["v"]
        for v in json.loads((SC / "verdicts.json").read_text(encoding="utf-8")):
            u = v["n"]; lu = cache.get(f"luna|{u}") or {}; ua = cache.get(f"useA|{u}") or {}; ub = cache.get(f"useB|{u}") or {}
            pw.evidence(experiment="scales", unit=u, evidence_type="job_class", metric="two_ended_relation", value_text=v.get("verdict") or v.get("luna"),
                        passes=v.get("verdict") is not None, context=f"luna={v.get('luna')};conf={v.get('conf')};jev={v.get('jev')}",
                        details={"end_descriptions": [lu.get("a"), lu.get("b")], "luna": lu, "jev_relation": cache.get(f"rel|{u}"), "useA": ua, "useB": ub,
                                 "use_ok": v.get("use_ok"), "note": "labels for reading, not the neuron's true meaning"},
                        source=f"scales/verdicts.json#{u}")
        for k, v in cache.items():
            if k.startswith("ctrl|"):
                i = k.split("|")[1]
                pw.evidence(experiment="scales", evidence_type="job_class", metric="two_ended_relation_shuffled_control", value_text=v.get("relation"),
                            context=f"control=shuffled_pair;conf={v.get('confidence')}", details={"luna": v, "jev_relation": cache.get(f"ctrlrel|{i}")},
                            key=f"ctrl|{i}", source=f"scales/scales.cache.jsonl#{k}")
        for exp, fn in (("scales_steer", "steer.jsonl"), ("scales_steer_allpos_invalid", "steer_allpos_invalid.jsonl")):
            if not (SC / fn).exists():
                continue
            for i, r in enumerate(read_jsonl(SC / fn)):
                if "class" not in r:
                    continue
                cA, cB = thing(pw, r["ends"]["A"]), thing(pw, r["ends"]["B"])
                pw.evidence(experiment=exp, unit=r["neuron"], concept=cA, concept2=cB, evidence_type="job_class", metric="steer_pattern",
                            value_text=r["class"], passes=r["class"].startswith("scale") if exp == "scales_steer" else None, context=f"group={r['group']}",
                            details={k: r[k] for k in ("ends", "end_sign", "picked", "peak", "control", "control_peak", "by_k", "control_ok", "descriptions")},
                            key=f"{fn}|{i}|class", source=f"scales/{fn}#{i}")
                for key, d in r["d"].items():
                    unit, end, kk, to = key.split("|")
                    pw.evidence(experiment=exp, unit=r["neuron"] if unit == "neuron" else r["control"], concept=cA if end == "A" else cB,
                                evidence_type="dose_response", metric="logodds_change", value=stx.mean(d), uom="logodds", n=len(d),
                                context=f"push_towards={to};k={kk[1:]};position={'answer' if exp == 'scales_steer' else 'all'}" + (";control=random_same_layer" if unit == "control" else ""),
                                details={"per_prompt": d, "end": end}, key=f"{fn}|{i}|{key}", source=f"scales/{fn}#{i}")


if __name__ == "__main__":
    main(force="--force" in sys.argv)

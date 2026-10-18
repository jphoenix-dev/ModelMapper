"""Loop v3 pilots + enrichment -> ledger pack (`loop_v3`).

Experiments: loop_v3_ranking_check (log-prob vs log-odds candidate search; fair 'disputed' knockouts), loop_v3_subject_pilot
(12 gate concepts), loop_v3_enrich (enrichment records: E0 missed candidates, E2 formats, E3 nulls, E4 subject pass, E7
inhibitors), loop_v3_e7_pilot (the E7-only pilot, loop_v3/e7_pilot/records.jsonl, when it exists).
E7 rows: knockout_drop with context 'stage=E7;selection=bottom_attr;...'; value < 0 = the answer went UP when the neuron was
zeroed; passes = raise beyond run 1's floor (value < -FLOOR). Its random nulls are null_knockout_drop rows.
"""
import json
import statistics as stx
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ledger.core import DATA_ROOT, PackWriter, read_jsonl, thing  # noqa: E402

V3 = DATA_ROOT / "loop_v3"
FLOOR = 0.100        # run 1's pooled random-neuron floor (99th pct |drop|), used for v3 knockouts on run 1's prompts


def subject_rows(pw, exp, c, out, src):
    pw.evidence(experiment=exp, concept=c, evidence_type="recall", metric="n_recalled_subject_facts", value=out.get("n_recalled"), n=len(out.get("facts", [])),
                threshold=3, passes=(out.get("n_recalled") or 0) >= 3, details={"name": out.get("name"), "facts": out.get("facts")}, key=f"{src}|recall", source=src)
    nulls = [abs(x["mean_drop"]) for x in out.get("nulls", [])]
    nf = max(nulls) if nulls else None
    for k, x in enumerate(out.get("top", [])):
        pw.evidence(experiment=exp, unit=x["neuron"], concept=c, evidence_type="knockout_drop", metric="mean_logodds_drop_over_facts",
                    value=x["mean_drop"], uom="logodds", n=len(x["per_fact"]), noise_floor=nf, threshold=nf, passes=(nf is not None and x["mean_drop"] > nf),
                    rank=k + 1, context="positions=name_tokens;test=subject_pass", details={"per_fact": x["per_fact"], "attr_rank": x.get("attr_rank")},
                    key=f"{src}|top|{k}", source=src)
    for k, x in enumerate(out.get("nulls", [])):
        pw.evidence(experiment=exp, unit=x["neuron"], concept=c, evidence_type="null_knockout_drop", metric="mean_logodds_drop_over_facts",
                    value=x["mean_drop"], uom="logodds", n=len(x["per_fact"]), context="positions=name_tokens;test=subject_pass;control=random_layer_matched",
                    details={"per_fact": x["per_fact"]}, key=f"{src}|null|{k}", source=src)


def main(force=False):
    files = [V3 / "ranking_check.jsonl", V3 / "ranking_disputed.jsonl", V3 / "subject" / "subject_maps.jsonl", V3 / "enrich_smoke" / "records.jsonl",
             V3 / "e7_pilot" / "records.jsonl"]
    files = [f for f in files if f.exists()]
    if not files:
        print("loop_v3: no data"); return
    with PackWriter("loop_v3", inputs=files + [V3 / "ranking_check.txt", V3 / "ranking_disputed.txt"], code=[__file__], force=force) as pw:
        if pw.skip:
            print("loop_v3: unchanged, skipped"); return
        base = dict(date="2026-10-18", docs=["LOOP_V3.md", "RESULTS.md#18 Oct 2026 - Loop v3 pilots: the candidate search misses strong neurons; the subject pass finds a new layer of neurons"])
        pw.experiment("loop_v3_ranking_check", title="Candidate search: log-prob vs log-odds attribution (50 run-1 concepts)",
                      method=("Run 1 searches candidates by the log-prob gradient but ranks them by exact log-odds knockout drop. Stage 1: overlap of the two "
                              "top-200 lists. Stage 2 (fair): every log-odds-only candidate zeroed alone with map_core's procedure; would it enter run 1's top 30?"),
                      status="pre-registered predictions (both failed: the log-prob search misses strong neurons)", status_class="pre-registered",
                      script="src/loop_v2/ranking_check.py", reproduce="python src/loop_v2/ranking_check.py check; python src/loop_v2/ranking_check.py disputed",
                      data_files=["data:loop_v3/ranking_check.jsonl", "data:loop_v3/ranking_disputed.jsonl"],
                      caveats=["A large log-odds drop on a near-certain prompt can be a small probability change.", "Some entrants may be generic machinery (e.g. L35:9346)."],
                      params={"stage1": (V3 / "ranking_check.txt").read_text(encoding="utf-8"), "stage2": (V3 / "ranking_disputed.txt").read_text(encoding="utf-8"), "noise_floor": FLOOR},
                      **base)
        for i, r in enumerate(read_jsonl(V3 / "ranking_disputed.jsonl")):
            c = thing(pw, r["concept"])
            pw.evidence(experiment="loop_v3_ranking_check", concept=c, evidence_type="concept_stat", metric="logodds_only_candidates_entering_top30",
                        value=r["n_enter_top30"], n=r["n_new"], details={k: r[k] for k in ("thr30", "new_best", "n_enter_top10")}, key=f"disp|{i}", source=f"loop_v3/ranking_disputed.jsonl#{i}")
            for k, e in enumerate(r["entering"]):
                pw.evidence(experiment="loop_v3_ranking_check", unit=e["neuron"], concept=c, evidence_type="knockout_drop", metric="mean_logodds_drop",
                            value=e["drop"], uom="logodds", noise_floor=FLOOR, threshold=r["thr30"], passes=True, rank=k + 1,
                            context="positions=all;prompts=recall;search=logodds_only", details={"logodds_rank": e["logodds_rank"], "run1_thr30": r["thr30"]},
                            key=f"disp|{i}|{k}", source=f"loop_v3/ranking_disputed.jsonl#{i}")
        sp = V3 / "subject" / "subject_maps.jsonl"
        if sp.exists():
            pw.experiment("loop_v3_subject_pilot", title="Subject pass pilot: neurons acting at a concept's name (12 gate concepts)",
                          method=("Luna writes 6 questions ABOUT the concept (name verbatim); recall-filtered; attribution of exact log-odds at the name tokens "
                                  "only, all layers; top 100 zeroed only at the name; 6 layer-matched random nulls."),
                          status="pre-registered predictions S1-S3 (all held)", status_class="pre-registered", script="src/loop_v2/subject_pass.py",
                          reproduce="python src/loop_v2/subject_pass.py questions | map | summary", data_files=["data:loop_v3/subject/"],
                          caveats=["One phrasing per fact (facts2 used three)."], **base)
            for i, out in enumerate(read_jsonl(sp)):
                subject_rows(pw, "loop_v3_subject_pilot", thing(pw, out["concept"]), out, f"loop_v3/subject/subject_maps.jsonl#{i}")
        ep = V3 / "enrich_smoke" / "records.jsonl"
        if ep.exists():
            pw.experiment("loop_v3_enrich", title="Loop v3 enrichment (smoke test, 50 run-1 concepts): missed candidates, formats, nulls, subject pass",
                          method=("Per concept only the stages it lacks (skip what we have). E0: log-odds-only candidates zeroed alone (map_core procedure). "
                                  "E2: recall prompts with two other instructions (as few words / one short sentence), Qwen's own answer as target; "
                                  "activations of the 200 candidates at the answer position per format; the top 30 zeroed per format. E3: 6 layer-matched "
                                  "random nulls. E4: the subject pass."),
                          status="pre-registered predictions C1-C2 (see enrich.py)", status_class="pre-registered", script="src/loop_v2/enrich.py",
                          reproduce="python src/loop_v2/enrich.py sample | questions | run | summary", data_files=["data:loop_v3/enrich_smoke/"],
                          caveats=["F3 (one sentence) targets Qwen's whole sentence: its knockout log-odds are not like-for-like with F1; compare activations.",
                                   "Run 1 concepts only; E1 (re-running v2 for earlier-only concepts) not in the smoke test."],
                          params={"noise_floor_E0": FLOOR}, **base)
            enrich_rows(pw, "loop_v3_enrich", ep, "loop_v3/enrich_smoke/records.jsonl")
        e7p = V3 / "e7_pilot" / "records.jsonl"
        if e7p.exists():
            pw.experiment("loop_v3_e7_pilot", title="Loop v3 stage E7 pilot: inhibitor neurons (knockout raises the answer)",
                          method=("For run-1 concepts (the first N of the full run's fixed order): run 1's saved attribution array; the 30 most NEGATIVE-"
                                  "attribution neurons (never knocked out by run 1) each zeroed alone with map_core's procedure (all positions, exact log-odds, "
                                  "mean over recall prompts); 4 layer-matched random nulls; change in the strongest rival token's log-prob."),
                          status="pre-registered predictions I1-I5 (see enrich.py)", status_class="pre-registered", script="src/loop_v2/enrich.py",
                          reproduce="ENRICH_NAME=e7_pilot ENRICH_ALL=1 ENRICH_SAMPLE_N=500 python src/loop_v2/enrich.py sample; ENRICH_NAME=e7_pilot ENRICH_STAGES=E7 python src/loop_v2/enrich.py run",
                          data_files=["data:loop_v3/e7_pilot/"], caveats=["Single-neuron knockouts; a raise on a near-certain prompt can be a small probability change."],
                          params={"noise_floor": FLOOR}, date="2026-10-18", docs=["MECHANISMS.md", "LOOP_V3.md"])
            enrich_rows(pw, "loop_v3_e7_pilot", e7p, "loop_v3/e7_pilot/records.jsonl")


def enrich_rows(pw, exp, path, rel):
    """One enrich records file -> evidence rows (E0, E2, E3, E4, E7), keyed by '<rel>#<line>' as before."""
    for i, r in enumerate(read_jsonl(path)):
        c = thing(pw, r["concept"]); src = f"{rel}#{i}"
        pw.evidence(experiment=exp, concept=c, evidence_type="concept_stat", metric="enrich_timing_s", value=r["t"].get("total"),
                    details={"t": r["t"], "gpu_busy": r["gpu_busy"], "peak_gb": r["peak_gb"], "skipped": r["skipped"], "stages_done": r["stages_done"]},
                    key=f"{src}|timing", source=src)
        e0 = r.get("E0")
        if e0:
            ent = set(e0["entering_top30"])
            for x in e0["new"]:
                pw.evidence(experiment=exp, unit=x["neuron"], concept=c, evidence_type="knockout_drop", metric="mean_logodds_drop",
                            value=x["mean_drop"], uom="logodds", n=len(x["per_prompt"]), noise_floor=FLOOR, threshold=FLOOR, passes=x["mean_drop"] > FLOOR,
                            context="stage=E0;positions=all;prompts=recall;search=logodds_only",
                            details={"per_prompt": x["per_prompt"], "logodds_rank": x["logodds_rank"], "enters_run1_top30": x["neuron"] in ent, "run1_thr30": e0["thr30"]},
                            key=f"{src}|E0|{x['neuron']}", source=src)
        e2 = r.get("E2")
        if e2:
            pw.evidence(experiment=exp, concept=c, evidence_type="concept_stat", metric="own_answers_by_format", value=None,
                        details={"own_answers": e2["own_answers"], "suffix_found": e2["suffix_found"]}, key=f"{src}|E2|own", source=src)
            for fm, vals in e2["acts"].items():
                for u, v in vals.items():
                    pw.evidence(experiment=exp, unit=u, concept=c, evidence_type="activation", metric="activation_answer_position",
                                value=stx.mean(v), n=len(v), context=f"stage=E2;format={fm}", details={"per_prompt": v}, key=f"{src}|E2|act|{fm}|{u}", source=src)
            for u, ks in e2["knockouts_top30"].items():
                for fm, v in ks.items():
                    if fm == "F1" or not v:
                        continue
                    pw.evidence(experiment=exp, unit=u, concept=c, evidence_type="knockout_drop", metric="logodds_drop_own_answer",
                                value=stx.mean(v), uom="logodds", n=len(v), noise_floor=FLOOR, context=f"stage=E2;format={fm};target=qwen_own_answer",
                                details={"per_prompt": v}, key=f"{src}|E2|ko|{fm}|{u}", source=src)
        e3 = r.get("E3")
        if e3:
            for u, v in e3["nulls"].items():
                pw.evidence(experiment=exp, unit=u, concept=c, evidence_type="null_knockout_drop", metric="mean_logodds_drop", value=stx.mean(v),
                            uom="logodds", n=len(v), context="stage=E3;positions=all;prompts=recall;control=random_layer_matched",
                            details={"per_prompt": v}, key=f"{src}|E3|{u}", source=src)
        if r.get("E4"):
            subject_rows(pw, exp, c, r["E4"], f"{src}|E4")
        e7 = r.get("E7")
        if e7:
            for k, x in enumerate(e7["bottom"]):
                pw.evidence(experiment=exp, unit=x["neuron"], concept=c, evidence_type="knockout_drop", metric="mean_logodds_drop",
                            value=x["mean_drop"], uom="logodds", n=len(x["per_prompt"]), noise_floor=FLOOR, threshold=-FLOOR,
                            passes=x["mean_drop"] < -FLOOR, rank=k + 1, context="stage=E7;selection=bottom_attr;positions=all;prompts=recall",
                            details={"attr": x["attr"], "per_prompt": x["per_prompt"], "d_rival": x["d_rival"], "d_rival_per_prompt": x["d_rival_per_prompt"],
                                     "rival_tokens": e7["rival_tokens"], "attr_source": e7["attr_source"]},
                            key=f"{src}|E7|{x['neuron']}", source=src)
            for u, v in e7["nulls"].items():
                pw.evidence(experiment=exp, unit=u, concept=c, evidence_type="null_knockout_drop", metric="mean_logodds_drop", value=stx.mean(v["per_prompt"]),
                            uom="logodds", n=len(v["per_prompt"]), context="stage=E7;positions=all;prompts=recall;control=random_layer_matched",
                            details={"per_prompt": v["per_prompt"], "d_rival": v["d_rival"]}, key=f"{src}|E7null|{u}", source=src)


if __name__ == "__main__":
    main(force="--force" in sys.argv)

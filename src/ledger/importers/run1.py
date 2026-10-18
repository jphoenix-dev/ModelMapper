"""Loop v2 run 1 (5,000 Vital Articles concepts; 4,194 mapped) -> ledger pack.

Per concept: the concept (frame attributes), recall + map statistics, every one of the 200 attributed candidates' single
knockout (per-prompt drops, attribution rank, activation in details), the 2 random-neuron controls, top-50 head knockouts,
weights-based roles + job tokens for the top 30, and the per-concept .npz arrays as artifacts. Noise floor for knockouts =
the run-wide 99th percentile of |mean drop| over all random-neuron controls (pooled; computed in a first pass).
"""
import glob
import json
import statistics as stx
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ledger.core import DATA_ROOT, PackWriter, concept_id, read_jsonl  # noqa: E402

RUN = DATA_ROOT / "loop_v2" / "run1"
FRAME = DATA_ROOT / "loop_v2" / "frame" / "frame.json"
KS = [1, 2, 3, 5, 10, 20, 50]
EXP = "loop_v2_run1"
NPZ_KEYS = {"attr": "[36,12288] activation x gradient per neuron (summed over positions, prompts)",
            "last_act": "[36,12288] activation at the answer position", "last_act_attr_pass": "[36,12288] same, attribution pass",
            "pmax": "[36,12288] max activation over prompt positions", "pmin": "[36,12288] min activation over prompt positions",
            "top200": "[200,2] (layer, neuron) of the attributed candidates", "head_attr": "[36,32] head attribution",
            "head_out_answer_pos": "[36,4096] head outputs at the answer position", "resid_answer_pos": "[prompts,37,4096] residual stream at the answer position",
            "att<i>": "[36,32,T] attention pattern from the answer position, prompt i"}


def records():
    for f in sorted(glob.glob(str(RUN / "results" / "**" / "records.jsonl"), recursive=True)):
        for i, r in enumerate(read_jsonl(f)):
            yield f, i, r


def main(force=False):
    files = sorted(glob.glob(str(RUN / "results" / "**" / "records.jsonl"), recursive=True))
    frame = {d["concept"]: d for d in json.loads(FRAME.read_text(encoding="utf-8"))["draw"]}
    with PackWriter("loop_v2_run1", inputs=files + [FRAME], code=[__file__], force=force) as pw:
        if pw.skip:
            print("run1: unchanged, skipped"); return
        nulls = [abs(stx.mean(v)) for _, _, r in records() for v in (r.get("v2", {}).get("null_single_drops") or {}).values() if v]
        nulls.sort(); floor = nulls[int(0.99 * len(nulls))]
        print(f"run1: pooled null from {len(nulls)} random-neuron controls: 99th pct |drop| {floor:.3f}")
        pw.experiment(EXP, title="Loop v2 run 1: 5,000 Vital Articles concepts mapped to neurons and heads",
                      method=("For each concept, Luna writes recall questions whose answer is the concept; Qwen must answer >= 5 "
                              "(P >= 0.2, leak-filtered). Every MLP neuron is scored by activation x gradient of the answer's "
                              "log-prob; the top 200 are each zeroed alone at all positions and ranked by the drop in the answer's "
                              "exact float64 log-odds. Also: k curves (top-k zeroed together vs random), 2 random same-layer "
                              "control neurons, top-50 heads by attribution each zeroed alone, and each top-30 neuron's weights-"
                              "based role and the tokens it pushes."),
                      status="maps exploratory; P1-P5 pre-registered (PREREG_MAP.md)", date="2026-10-18",
                      script="src/loop_v2/map_core.py (via src/dist runner)",
                      reproduce="python src/dist/make_run.py --kind map --specs <specs dir> ...   (LOOP_V2.md)",
                      data_files=["data:loop_v2/run1/results/results/map/<batch>/records.jsonl", "data:loop_v2/run1/results/results/map/<batch>/arrays/<slug>.npz"],
                      caveats=["Attribution ranks by log-prob gradient: prompts Qwen is already sure of weigh less.",
                               "Knockout drops are per-neuron and single; redundant neurons can carry a concept without a large single drop.",
                               "Format-specific: every question asks to 'Answer with just the name'."],
                      docs=["LOOP_V2.md", "RESULTS.md", "PREREG_MAP.md"],
                      params={"n_candidates": 200, "ks": KS, "min_recall": 5, "noise_floor": {"rule": "pooled 99th pct |mean drop| of random controls", "value": floor, "n": len(nulls)}},
                      code_commit=None)
        for f, i, r in records():
            c = r["concept"]; row = r.get("row") or {}; rank = r.get("rank") or {}; v2 = r.get("v2") or {}
            fr = frame.get(c, {})
            cid = pw.concept(concept_id("thing", c), c, "thing", aliases=[row.get("answer_form"), row.get("answer")], qid=fr.get("qid"),
                             attributes={"section": fr.get("section"), "fame_tier": fr.get("tier"), "links": fr.get("links"),
                                         "named_or_common": fr.get("kind"), "named_why": fr.get("kind_why"), "frame": "vital_articles_l4"})
            src = f"{Path(f).relative_to(DATA_ROOT).as_posix()}#{i}"
            prov = v2.get("provenance"); n = row.get("recall")
            pw.evidence(experiment=EXP, concept=cid, evidence_type="recall", metric="n_recall_prompts", value=n, n=20,
                        threshold=5, passes=(n or 0) >= 5, value_text=row.get("status"), source=src,
                        details={"answer": row.get("answer"), "answer_form": row.get("answer_form"), "answer_form_scores": row.get("answer_form_scores"),
                                 "prompts": row.get("recall_prompts"), "provenance": prov, "timing_s": v2.get("timing_s")})
            if row.get("status") != "ok":
                continue
            for m_ in ("baseline_mean_logodds", "k_mean", "k_median", "k_flip"):
                if row.get(m_) is not None:
                    pw.evidence(experiment=EXP, concept=cid, evidence_type="concept_stat", metric=m_, value=row[m_], n=n, source=src,
                                details={"base_per_prompt": row.get("base_per_prompt")} if m_ == "baseline_mean_logodds" else None)
            pw.evidence(experiment=EXP, concept=cid, evidence_type="concept_stat", metric="k_curve", value=None, n=n, source=src,
                        details={"ks": KS, "drops": row.get("drops"), "random_drops": row.get("random_drops")})
            ppd = rank.get("per_prompt_drop_all200") or {}
            cands = sorted(rank.get("candidates_all200") or [], key=lambda x: -(x.get("single_drop") or -1e9))
            for k, x in enumerate(cands):
                pw.evidence(experiment=EXP, unit=x["neuron"], concept=cid, evidence_type="knockout_drop", metric="mean_logodds_drop",
                            value=x.get("single_drop"), uom="logodds", n=n, noise_floor=floor, threshold=floor,
                            passes=(x.get("single_drop") or 0) > floor, rank=k + 1, context="positions=all;prompts=recall",
                            details={"attr_rank": x.get("attr_rank"), "act_answer_pos": x.get("act_answer_pos"), "per_prompt": ppd.get(x["neuron"])},
                            source=src)
            for u, pp in (v2.get("null_single_drops") or {}).items():
                pw.evidence(experiment=EXP, unit=u, concept=cid, evidence_type="null_knockout_drop", metric="mean_logodds_drop",
                            value=stx.mean(pp) if pp else None, uom="logodds", n=len(pp or []), noise_floor=floor,
                            context="positions=all;prompts=recall;control=random_same_layer", details={"per_prompt": pp}, source=src)
            hk = rank.get("head_knockout_mean") or {}; hpp = rank.get("head_knockout_per_prompt") or {}
            hattr = {h: j + 1 for j, h in enumerate(rank.get("heads_top50_by_attr") or [])}
            for k, (h, d) in enumerate(sorted(hk.items(), key=lambda kv: -kv[1])):
                pw.evidence(experiment=EXP, unit=h, concept=cid, evidence_type="knockout_drop", metric="mean_logodds_drop", value=d,
                            uom="logodds", n=n, rank=k + 1, context="positions=all;prompts=recall",
                            details={"attr_rank": hattr.get(h), "per_prompt": hpp.get(h)}, source=src)
            for x in v2.get("roles_top30") or []:
                pw.evidence(experiment=EXP, unit=x["neuron"], concept=cid, evidence_type="role", metric="weights_role", value_text=x.get("role"),
                            details={"answer_rank": x.get("answer_rank"), "answer_z": x.get("answer_z")}, source=src)
                if x.get("job_tokens"):
                    pw.evidence(experiment=EXP, unit=x["neuron"], concept=cid, evidence_type="readout_tokens", metric="direct_logit_top30",
                                value_text=" | ".join(t.strip() or repr(t) for t in x["job_tokens"][:10]), details={"tokens": x["job_tokens"]}, source=src)
            if v2.get("arrays"):
                ap = Path(f).parent / "arrays" / v2["arrays"]
                if ap.exists():
                    pw.artifact(ap, experiment=EXP, fmt="npz", keys=NPZ_KEYS, concept=cid, description=f"run 1 dense arrays for {c}")


if __name__ == "__main__":
    main(force="--force" in sys.argv)

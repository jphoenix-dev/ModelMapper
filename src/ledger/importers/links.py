"""Links between a concept's key neurons -> ledger packs.

  indirect_path  results/indirect_path.jsonl (src/indirect_path.py, 17 Oct: 555 concepts, top 5 neurons): per (earlier i,
                 later j) the relative drop of j's activation at the answer position when i is zeroed, vs i's
                 activation-matched null - scored with the LINKS.md rule; plus each neuron's total / direct / indirect
                 effect on the answer score.
  links_run1     D:/already-clean-data/loop_v2/links_run1/ (map session's LINKS.md, top 30, all run-1 concepts; gpu-fan-out
                 runner) - imported when the data exists.
Pass rule (LINKS.md, fixed before data): |mean drop| >= 0.05, >= 3x |mean null|, sign agrees in >= 2/3 of prompts and >= 3
prompts. Positive = i supports j, negative = i suppresses j (value keeps the sign).
"""
import json
import statistics as stx
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ledger.core import DATA_ROOT, REPO_RESULTS, PackWriter, read_jsonl, thing  # noqa: E402

IP = REPO_RESULTS / "indirect_path.jsonl"
LR = DATA_ROOT / "loop_v2" / "links_run1"
LINKS_MD = Path("C:/dev/already-clean-map/explainer/gen/map/LINKS.md")
lay = lambda u: int(u[1:].split(":")[0])


def pair_stats(prompts, i, j, n_units):
    """prompts: list of (acts rows, null rows) - acts[r][k]; row 0 base, row 1+i = i zeroed, row 1+n+i = i's null."""
    d, nl = [], []
    for acts in prompts:
        b = acts[0][j]
        if abs(b) < 1e-3:
            continue
        s = 1.0 if b > 0 else -1.0
        d.append((b - acts[1 + i][j]) * s / abs(b)); nl.append((b - acts[1 + n_units + i][j]) * s / abs(b))
    if not d:
        return None
    md, mn = stx.mean(d), stx.mean(nl)
    agree = sum((x > 0) == (md > 0) for x in d) / len(d)
    passes = abs(md) >= 0.05 and abs(md) >= 3 * abs(mn) and agree >= 2 / 3 and len(d) >= 3
    return md, mn, len(d), agree, passes, d, nl


def link_rule():
    return {"min_abs_drop": 0.05, "min_ratio_to_null": 3, "min_sign_agreement": 2 / 3, "min_prompts": 3}


def import_indirect(force):
    with PackWriter("indirect_path", inputs=[IP, REPO_RESULTS / "indirect_path_summary.txt"], code=[__file__], force=force) as pw:
        if pw.skip:
            print("indirect_path: unchanged, skipped"); return
        pw.experiment("indirect_path", title="Indirect path: do 'say' neurons write the answer, and do others work through them? (top 5 per concept)",
                      method=("Per mapped concept (555, earlier maps) and recall prompt, at the answer position: each of the top-5 neurons zeroed at "
                              "every position, and for each an activation-matched null (same layer, outside the top 30, closest |activation|). "
                              "Per row: answer score, answer log-odds and all top-5 activations. direct = what the neuron wrote, read through the "
                              "final norm into the answer score; total = the score drop; indirect = total - direct. Links between earlier and "
                              "later top-5 neurons scored here with the LINKS.md rule."),
                      status="exploratory (predictions (a) and (b) stated before running; both held)", status_class="exploratory", date="2026-10-17",
                      script="src/indirect_path.py", reproduce="python src/indirect_path.py",
                      data_files=["repo_results:indirect_path.jsonl", "repo_results:indirect_path_summary.txt"],
                      caveats=["Top 5 neurons per concept only; links_run1 extends to the top 30 on run 1."],
                      docs=["RESULTS.md (17 Oct, indirect path)", "LAYERS_AND_JOBS.md#strongly-supported"],
                      params={"link_rule": link_rule(), "summary": (REPO_RESULTS / "indirect_path_summary.txt").read_text(encoding="utf-8")})
        for ci, r in enumerate(read_jsonl(IP)):
            c = thing(pw, r["concept"]); U = r["units"]; n = len(U)
            P = [p["unit_acts"] for p in r["prompts"]]
            for i, u in enumerate(U):
                tot = [p["score"][0] - p["score"][1 + i] for p in r["prompts"]]
                nt = [p["score"][0] - p["score"][1 + n + i] for p in r["prompts"]]
                dr = [p["direct"][i] for p in r["prompts"]]
                lo = [p["logodds"][0] - p["logodds"][1 + i] for p in r["prompts"]]
                pw.evidence(experiment="indirect_path", unit=u, concept=c, evidence_type="knockout_drop", metric="answer_score_drop_total",
                            value=stx.mean(tot), n=len(tot), noise_floor=abs(stx.mean(nt)), value_text=r["types"][i],
                            details={"direct": stx.mean(dr), "indirect": stx.mean(tot) - stx.mean(dr), "logodds_drop": stx.mean(lo),
                                     "null_total": stx.mean(nt), "per_prompt_total": tot, "per_prompt_direct": dr,
                                     "nulls": [p["nulls"][i] for p in r["prompts"]], "role": r["types"][i]},
                            source=f"repo_results:indirect_path.jsonl#{ci}")
            for i in range(n):
                for j in range(n):
                    if lay(U[i]) >= lay(U[j]):
                        continue
                    s = pair_stats(P, i, j, n)
                    if not s:
                        continue
                    md, mn, k, agree, passes, d, nl = s
                    pw.evidence(experiment="indirect_path", unit=U[i], unit2=U[j], concept=c, evidence_type="causal_link",
                                metric="relative_activation_drop", value=md, noise_floor=abs(mn), n=k, passes=passes, threshold=0.05,
                                context="direction=" + ("supports" if md > 0 else "suppresses") + f";roles={r['types'][i]}->{r['types'][j]}",
                                details={"mean_null": mn, "sign_agreement": agree, "per_prompt": d, "per_prompt_null": nl},
                                source=f"repo_results:indirect_path.jsonl#{ci}")


def import_links_run1(force):
    """Format (gpu-fan-out, src/dist/links.py on research/distributed 5d851e8): <LR>/results/links/<batch>/{records.jsonl,
    links.jsonl, arrays.npz}; links.jsonl one row per (i, j) with layer_i < layer_j, passing or not."""
    batches = sorted(p.parent for p in LR.rglob("links.jsonl")) if LR.exists() else []
    if not batches:
        print("links_run1: no data yet at", LR); return
    inputs = [f for b in batches for f in (b / "records.jsonl", b / "links.jsonl") if f.exists()] + ([LINKS_MD] if LINKS_MD.exists() else [])
    with PackWriter("links_run1", inputs=inputs, code=[__file__], force=force) as pw:
        if pw.skip:
            print("links_run1: unchanged, skipped"); return
        pw.experiment("links_run1", title="Links between each run-1 concept's 30 key neurons (zero i, measure later j)",
                      method=LINKS_MD.read_text(encoding="utf-8") if LINKS_MD.exists() else "See LINKS.md (map worktree).",
                      status="exploratory (pass rule fixed before the run)", status_class="exploratory", date="2026-10-18",
                      script="src/dist/links.py (branch research/distributed) via the gpu-fan-out runner",
                      reproduce="python src/dist/make_run.py --kind links ...   (gpu-fan-out session)",
                      data_files=["data:loop_v2/links_run1/results/links/<batch>/{records.jsonl,links.jsonl,arrays.npz}"],
                      caveats=["Answer position only (the last prompt token); first answer token's log-odds.",
                               "Activation-matched nulls (same layer, outside the top 30, closest |activation|)."],
                      docs=["LINKS.md (map worktree)"], params={"link_rule": link_rule()})
        for b in batches:
            rel = b.relative_to(DATA_ROOT).as_posix()
            for i, r in enumerate(read_jsonl(b / "records.jsonl")):
                c = thing(pw, r["concept"])
                pw.evidence(experiment="links_run1", concept=c, evidence_type="concept_stat", metric="links_gate_and_counts", value=r.get("n_links_passing"),
                            n=r.get("n_prompts"), details={k: r.get(k) for k in ("units", "types", "answer_id", "gate_zero_fail", "gate_logodds_fail",
                                                                                  "n_pairs", "n_links_passing", "source", "array_index", "prompts")},
                            key=f"rec|{rel}|{i}", source=f"{rel}/records.jsonl#{i}")
            for i, x in enumerate(read_jsonl(b / "links.jsonl")):
                c = thing(pw, x["concept"]); md = x.get("mean_drop")
                ctx = f"direction={x.get('direction') or ('supports' if (md or 0) > 0 else 'suppresses' if md else 'none')}"
                if x.get("type_i"):
                    ctx += f";roles={x.get('type_i')}->{x.get('type_j')}"
                pw.evidence(experiment="links_run1", unit=x["unit_i"], unit2=x["unit_j"], concept=c, evidence_type="causal_link",
                            metric="relative_activation_drop", value=md, noise_floor=abs(x["mean_null"]) if x.get("mean_null") is not None else None,
                            n=x.get("n_prompts"), threshold=0.05, passes=bool(x.get("passes")), context=ctx,
                            details={k: x.get(k) for k in ("mean_null", "sign_agree", "n_agree", "i", "j", "layer_i", "layer_j")},
                            key=f"link|{rel}|{i}", source=f"{rel}/links.jsonl#{i}")
            if (b / "arrays.npz").exists():
                pw.artifact(b / "arrays.npz", experiment="links_run1", fmt="npz",
                            keys={"<array_index>|acts": "float32 [P, 61, 30]: row 0 base, 1-30 unit k zeroed, 31-60 its null zeroed",
                                  "<array_index>|logodds": "float64 [P, 61] first-token log-odds per row"},
                            description=f"links_run1 dense arrays, batch {b.name}")


def main(force=False):
    import_indirect(force)
    import_links_run1(force)


if __name__ == "__main__":
    main(force="--force" in sys.argv)

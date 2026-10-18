"""Concept atlas (concept_atlas.py: build / propose / verify_fast / verify_gate) -> ledger pack.

- 'recall' rows: every atlas name with >= 1 cloze prompt (index.json): value = recalled cloze prompts (P >= MIN_P),
  n = prompts tried; details = prompts, per-prompt P, best answer form, answer ids, acts.npy rows.
- 'search_verify' rows (verify_fast.jsonl, 300 neurons): unit = neuron, concept = its proposed atlas name (top-1 by z,
  either sign), value = mean exact log-odds drop when zeroed everywhere on Luna's independent quiz questions,
  value_text = status, context = group; noise_floor = pooled 99th pct of the null drops (recomputed), threshold =
  max(1.0, that), passes = the script's 'verified' rule (None if untested). details = z1 / sign, the neuron's top-10
  atlas names and z for both signs (proposals.npz), recall probabilities and indices, base log-odds, per-question drops,
  null neurons, and the questions. Each null neuron gets a 'null_knockout_drop' row.
- Dense / raw files (acts.npy 7 GB, proposals.npz, intros.json) and the trial runs (vf_trial*, gate files) are artifacts.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ledger.core import DATA_ROOT, PackWriter, concept_id, read_jsonl  # noqa: E402

CA = DATA_ROOT / "concept_atlas"
VF, VFS, VQ, IDX, PCHK = CA / "verify_fast.jsonl", CA / "verify_fast_summary.txt", CA / "verify_questions.jsonl", CA / "index.json", CA / "proposals_check.txt"
ACTS, PROP, INTROS = CA / "acts.npy", CA / "proposals.npz", CA / "intros.json"
TRIALS = [CA / f for f in ("vf_trial.jsonl", "vf_trial_summary.txt", "vf_trial_gate.txt", "vf_trial_v1.jsonl", "vf_trial_v1_gate.txt")]
LOGS = [CA / f for f in ("build.log", "propose.log", "verify_fast.log")]
EXP = "concept_atlas"
DOC = "RESULTS.md#17 Oct 2026 (night): the concept atlas - asking Qwen which names its neurons serve finds new concept neurons (EXPLORATORY; research branch)"
MIN_P = 0.2
GROUPS = {"top": "100 unmapped neurons (no concept's top-30) with the highest top-1 z (either sign), one per name",
          "random": "100 unmapped neurons, layer-matched to 'top', each with its own top-1 name",
          "known": "100 known concept neurons (a mapped top-5 neuron whose concept is in the atlas), each with its top-1 atlas name"}


def parse_gate(p):
    """'verify gate (...)' header + per-neuron 'Lx:n name batched +a unbatched +b' lines."""
    lines = p.read_text(encoding="utf-8").splitlines()
    rows = []
    for ln in lines[1:]:
        m = re.match(r"\s*(L\d+:\d+)\s+(.*?)\s+batched ([-+]?[\d.]+|nan) unbatched ([-+]?[\d.]+|nan)\s*$", ln)
        if m:
            rows.append({"neuron": m.group(1), "name": m.group(2), "batched": float(m.group(3)), "unbatched": float(m.group(4))})
    return {"header": lines[0] if lines else None, "rows": rows}


def main(force=False):
    inputs = [VF, VFS, VQ, IDX, PCHK, ACTS, PROP, INTROS] + TRIALS + LOGS
    with PackWriter("concept_atlas", inputs=inputs, code=[__file__], force=force) as pw:
        if pw.skip:
            print("concept_atlas: unchanged, skipped"); return
        R = list(read_jsonl(VF)); index = json.loads(IDX.read_text(encoding="utf-8"))
        Q = {}
        for q in read_jsonl(VQ):
            Q[q["name"]] = q["questions"]
        P = np.load(PROP); topn, topz, pnames = P["topn"], P["topz"], [str(x) for x in P["names"]]
        nulls = [v for r in R if r["status"] == "tested" for v in r["null_drop_means"]]
        p99 = float(np.percentile(nulls, 99)); thr = max(1.0, p99)
        gates = {p.name: parse_gate(p) for p in TRIALS if p.name.endswith("_gate.txt")}
        pw.experiment(EXP, title="Concept atlas: asking Qwen which names each neuron serves, then verifying causally",
                      method=("6,061 English Wikipedia names (>= 16 language editions, plus our Wikipedia-titled mapped concepts); up to "
                              "4 cloze prompts per name from its intro (name blanked, as in P1). For every recalled prompt (P >= 0.2) all "
                              "442,368 neurons are recorded at the answer position. Each neuron's proposals = the names that make it fire "
                              "unusually strongly (z across names, either sign). Verification (verify_fast) on independent Luna-written quiz "
                              "questions for the proposed name: with >= 3 recalled, the neuron is zeroed everywhere, measured against an "
                              "as-is copy and 2 random same-layer null neurons in the same pass."),
                      status="pre-registered", date="2026-10-17", script="src/concept_atlas.py",
                      reproduce="python src/concept_atlas.py fetch && ... build && ... propose && ... verify_fast && VF_TAG=vf_trial ... verify_gate",
                      data_files=["data:concept_atlas/verify_fast.jsonl", "data:concept_atlas/verify_fast_summary.txt", "data:concept_atlas/verify_questions.jsonl",
                                  "data:concept_atlas/index.json", "data:concept_atlas/proposals_check.txt", "data:concept_atlas/acts.npy",
                                  "data:concept_atlas/proposals.npz", "data:concept_atlas/intros.json"] + [f"data:concept_atlas/{p.name}" for p in TRIALS],
                      caveats=["Criterion NOT MET on its absolute bar (top 16% < 20%); met on the ratio (16 vs 0 random).",
                               "Gate rule (Spearman >= 0.95 between batched and unbatched drops, same verdict >= 95%) NOT met as written: Spearman 0.79 "
                               "on identical question sets (verdicts 30/30, max diff 0.15); proceeded on verdict agreement.",
                               "One sample of 100 per group; Luna's questions can drift from the exact name.",
                               "'known' verifies on its TOP atlas name, which is its own concept only ~21% of the time.",
                               "Layer-35 neurons dominate some groups.",
                               "Dense activations (acts.npy) and proposals are artifacts, not evidence rows."],
                      docs=[DOC],
                      params={"groups": GROUPS, "min_recalled": 3, "recall_min_p": MIN_P, "max_cloze_prompts": 4, "min_links": 16,
                              "verified_rule": "drop >= max(1.0, pooled null 99th percentile)",
                              "noise_floor": {"rule": "pooled 99th pct of null drop means (2 random same-layer neurons per tested neuron)",
                                              "value": p99, "n": len(nulls)}, "threshold": thr,
                              "criterion": "verified rate top >= 3x random and top >= 20%", "criterion_outcome": "NOT MET (top 16%, random 0%)",
                              "proposals_check": PCHK.read_text(encoding="utf-8").strip(),
                              "verify_fast_summary": VFS.read_text(encoding="utf-8"),
                              "gate_rule": "Spearman(batched, unbatched drop) >= 0.95 and same verified call (threshold 1.0) for >= 95% of neurons",
                              "gates": gates, "seeds": {"select": 61, "nulls": 62},
                              "n_names_with_prompts": len(index["names"]), "n_recalled_names": sum(bool(e["rows"]) for e in index["names"].values()),
                              "n_recalled_prompts": index.get("n_rows")})
        # ---- recall per atlas name
        for t, e in index["names"].items():
            cid = pw.concept(concept_id("thing", t), t, "thing", aliases=[e.get("name"), e.get("form")])
            nrec = len(e["rows"])
            pw.evidence(experiment=EXP, concept=cid, evidence_type="recall", metric="n_recalled_cloze_prompts", value=nrec, n=len(e["prompts"]),
                        threshold=1, passes=nrec >= 1, context="prompts=atlas_cloze", source=f"concept_atlas/index.json#names/{t}",
                        key=f"recall|{t}",          # two titles can share a slug (JoJo Siwa / Jojo Siwa): one row per title
                        details={k: v for k, v in e.items() if k != "title"})
        # ---- verify_fast
        ppos = {t: i for i, t in enumerate(pnames)}
        for i, r in enumerate(R):
            u = r["neuron"]; nm = r["name"]
            Lr, n_ = (int(x) for x in u[1:].split(":")); x = Lr * 12288 + n_
            e = index["names"].get(nm, {})
            cid = pw.concept(concept_id("thing", nm), nm, "thing", aliases=[e.get("name"), r.get("form")])
            tested = r["status"] == "tested"
            det = {k: v for k, v in r.items() if k not in ("neuron", "group", "name", "status", "drop_mean")}
            det["atlas_top10"] = {sg: [[pnames[int(j)], round(float(z), 3)] for j, z in zip(topn[x, s], topz[x, s])]
                                  for s, sg in enumerate(("+", "-"))}
            det["proposed_name_index_in_atlas"] = ppos.get(nm)
            det["questions"] = Q.get(nm)
            src = f"concept_atlas/verify_fast.jsonl#{i}"
            pw.evidence(experiment=EXP, unit=u, concept=cid, evidence_type="search_verify", metric="mean_logodds_drop", value=r.get("drop_mean"),
                        value_text=r["status"], uom="logodds", n=r.get("recalled"), noise_floor=p99, threshold=thr,
                        passes=(r["drop_mean"] >= thr) if tested else None,
                        context=f"group={r['group']};proposal=atlas_top1_z;prompts=luna_quiz;positions=all", details=det, source=src)
            for j, (nu, nd) in enumerate(zip(r.get("null_neurons") or [], r.get("null_drop_means") or [])):
                pw.evidence(experiment=EXP, unit=nu, concept=cid, evidence_type="null_knockout_drop", metric="mean_logodds_drop", value=nd,
                            uom="logodds", n=r.get("recalled"), noise_floor=p99,
                            context=f"group={r['group']};control=random_same_layer;for={u}", source=f"{src}.null[{j}]")
        # ---- artifacts
        pw.artifact(ACTS, experiment=EXP, fmt="npy", description="activations of all 442,368 neurons at the answer position, per recalled cloze prompt (fp16 memmap; row index in index.json)",
                    keys={"acts": f"{list(np.load(ACTS, mmap_mode='r').shape)} float16; rows 0..{index.get('n_rows')} used (rest unused capacity)"})
        pw.artifact(PROP, experiment=EXP, fmt="npz", sha=True, description="per neuron and sign: top-10 atlas names by z across names",
                    keys={"topn": "[442368, 2 signs (+,-), 10] index into names", "topz": "[442368, 2, 10] z", "names": "[1514] recalled names"})
        pw.artifact(INTROS, experiment=EXP, fmt="json", sha=True, description="Wikipedia intros fetched for the 6,061 atlas names (cloze source)")
        for p in TRIALS:
            pw.artifact(p, experiment=EXP, sha=True, description=f"verify_fast trial / gate run ({p.name}); superseded by verify_fast.jsonl, kept for the gate record")
        for p in LOGS:
            pw.artifact(p, experiment=EXP, sha=True, description=f"run log ({p.name})")


if __name__ == "__main__":
    main(force="--force" in sys.argv)

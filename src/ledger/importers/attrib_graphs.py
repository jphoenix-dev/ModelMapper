"""Attribution graphs on Qwen3's own neurons (ATTRIB_GRAPHS.md + amendments; src/loop_v2/attrib_graph.py) -> pack `attrib_graphs`.

Per question (concept): gates + coverage (concept_stat), every kept node (attribution: total act x grad of the target,
plus its direct contribution to the target), every stored edge (graph_edge: direct contribution src -> dst, through the
frozen model), group ablations when present. Test runs on other models (folders starting with '_') are skipped.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ledger.core import DATA_ROOT, PackWriter, thing  # noqa: E402

D = DATA_ROOT / "attrib_graphs"
SPEC = Path(os.environ.get("MAP_WORKTREE", "C:/dev/already-clean-map")) / "explainer/gen/map/ATTRIB_GRAPHS.md"


def main(force=False):
    gdirs = sorted(p for p in D.iterdir() if p.is_dir() and not p.name.startswith("_") and (p / "graph.json").exists()) if D.exists() else []
    if not gdirs:
        print("attrib_graphs: no graphs yet"); return
    files = [f for d in gdirs for f in (d / "graph.json", d / "groups.json", d / "ablations.json") if f.exists()] + ([SPEC] if SPEC.exists() else [])
    with PackWriter("attrib_graphs", inputs=files, code=[__file__], force=force) as pw:
        if pw.skip:
            print("attrib_graphs: unchanged, skipped"); return
        pw.experiment("attrib_graphs", title="Attribution graphs on Qwen3-8B's own neurons (pilot, 18 questions)",
                      method=SPEC.read_text(encoding="utf-8") if SPEC.exists() else "See ATTRIB_GRAPHS.md.",
                      status="exploratory (checks fixed before data; amendments before any 8B data)", status_class="exploratory", date="2026-10-18",
                      script="src/loop_v2/attrib_graph.py", reproduce="python src/loop_v2/attrib_graph.py graphs; python src/loop_v2/attrib_graph.py ablate",
                      data_files=["data:attrib_graphs/<slug>/{graph.json,groups.json,ablations.json}"],
                      caveats=["First answer token only; attention frozen (heads appear only as routes); first-order (linearised) attributions.",
                               "300 nodes hold only ~10% of total |attribution|; the joint knockout says how much of the answer the drawn graph carries.",
                               "Position 0 (the attention sink) is not a node."],
                      docs=["ATTRIB_GRAPHS.md (map worktree)"])
        for d in gdirs:
            g = json.loads((d / "graph.json").read_text(encoding="utf-8")); c = thing(pw, g["concept"]); src = f"attrib_graphs/{d.name}/graph.json"
            pw.evidence(experiment="attrib_graphs", concept=c, evidence_type="concept_stat", metric="attrib_graph_gates_and_coverage",
                        value=(g["gates"].get("g3_knockouts") or {}).get("spearman_top30"), threshold=0.7, passes=g["gates"].get("pass"),
                        details={"gates": g["gates"], "coverage": g.get("coverage"), "completeness": g["completeness"], "question": g.get("question"),
                                 "tokens": g["tokens"], "target": g["target"], "seconds": g.get("seconds")}, key=f"{d.name}|gates", source=src)
            for n in g["nodes"]:
                if n["type"] != "neuron":
                    continue
                pw.evidence(experiment="attrib_graphs", unit=n["neuron"], concept=c, evidence_type="attribution", metric="act_x_grad_target_linearised",
                            value=n["attr"], context=f"pos={n['pos']};token={n['token']!r}",
                            details={"act": n["act"], "direct_to_target": n.get("direct_to_target"), "node_id": n["id"]}, key=f"{d.name}|node|{n['id']}", source=src)
            nid = {n["id"]: n for n in g["nodes"]}
            for i, e in enumerate(g["edges"]):
                s_, t_ = nid.get(e["src"], {}), nid.get(e["dst"], {})
                pw.evidence(experiment="attrib_graphs", unit=s_.get("neuron"), unit2=t_.get("neuron"), concept=c, evidence_type="graph_edge",
                            metric="direct_contribution", value=e["w"], context=f"src={e['src']};dst={e['dst']}", key=f"{d.name}|edge|{i}", source=src)
            if (d / "ablations.json").exists():
                a = json.loads((d / "ablations.json").read_text(encoding="utf-8"))
                for lab, v in a.get("groups", {}).items():
                    pw.evidence(experiment="attrib_graphs", concept=c, evidence_type="knockout_drop", metric="p_answer_with_group_off", value=v["p"],
                                passes=v.get("beats_all_random"), context=f"group={lab};row={v.get('row')}",
                                details={**v, "p_base": a.get("p_base")}, key=f"{d.name}|group|{lab}", source=f"attrib_graphs/{d.name}/ablations.json")


if __name__ == "__main__":
    main(force="--force" in sys.argv)

"""Turn attribution graphs (ATTRIB_GRAPHS.md) into the viewer's data: explainer/site/map/data/graphs/<slug>.json.

    python -I -X utf8 explainer/gen/map/build_graphs.py

For each D:/already-clean-data/attrib_graphs/<slug>/ with graph.json and groups.json (and ablations.json when the
group checks have run): group-level boxes (label, row, neurons, total attribution), links between groups (the sum of
the edges between their neurons) and from groups to the answer, and each group's removal result against random
neurons. Writes an index (graphs/index.json) for the viewer's pager.
"""
from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "site" / "map" / "data" / "graphs"
ROOT = Path(os.environ.get("MAP_GRAPHS", "D:/already-clean-data/attrib_graphs"))


def question_text(prompt):
    m = re.search(r"<\|im_start\|>user\n(.*?)(?: Answer with just the name\.)?<\|im_end\|>", prompt or "", re.S)
    return (m.group(1) if m else prompt or "").strip()


def r3(x):
    return None if x is None else round(float(x), 4)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    index = []
    for d in sorted(p for p in ROOT.iterdir() if (p / "graph.json").exists() and (p / "groups.json").exists()):
        g = json.loads((d / "graph.json").read_text(encoding="utf-8"))
        grp = json.loads((d / "groups.json").read_text(encoding="utf-8"))
        abl = json.loads((d / "ablations.json").read_text(encoding="utf-8")) if (d / "ablations.json").exists() else {}
        nodes = {n["id"]: n for n in g["nodes"]}
        target = next((n["id"] for n in g["nodes"] if n.get("type") == "target" or n.get("kind") == "target" or n["id"] == "target"), "target")
        owner = {}
        groups = []
        for gi, x in enumerate(grp["groups"]):
            mem = [nodes[i] for i in x["nodes"] if i in nodes]
            for n in mem:
                owner[n["id"]] = gi
            a = (abl.get("groups") or {}).get(x["label"]) or (abl.get("groups") or {}).get(str(gi)) or {}
            groups.append({"id": gi, "label": x["label"], "row": x["row"], "n": len(mem),
                           "attr": r3(sum(n.get("attr", 0) for n in mem)),
                           "nodes": [{"neuron": n.get("neuron"), "pos": n.get("pos"), "token": n.get("token"),
                                      "attr": r3(n.get("attr")), "act": r3(n.get("act"))}
                                     for n in sorted(mem, key=lambda n: -abs(n.get("attr", 0)))],
                           "ablation": {"p": r3(a.get("p")), "p_random": [r3(v) for v in a.get("p_random", [])]} if a else None})
        for i in grp.get("other", []):
            owner.setdefault(i, -2)                                    # -2: the faint "other" box
        # group-to-group links: the sum of the edges between their members; group -> answer (-1): edges into the target
        agg = defaultdict(float)
        for e in g["edges"]:
            s, t = e["src"], e["dst"]
            gs = owner.get(s)
            gt = -1 if t == target else owner.get(t)
            if gs is None or gt is None or gs == gt:
                continue
            agg[(gs, gt)] += e["w"]
        edges = sorted(([a, b, r3(w)] for (a, b), w in agg.items()), key=lambda e: -abs(e[2]))[:40]
        other = [nodes[i] for i in grp.get("other", []) if i in nodes]
        # the research session's names: gates.g3_knockouts.spearman_top30, gates.pass, coverage.{attr_share, joint_knockout}
        gates, cov = g.get("gates") or {}, g.get("coverage") or {}
        g3 = gates.get("g3_knockouts") or {}
        jk = abl.get("joint") or cov.get("joint_knockout")
        if jk and "p_base" not in abl and jk.get("p_base") is not None:
            abl["p_base"] = jk["p_base"]
        abl["joint"] = jk
        out = {"slug": d.name, "concept": g.get("concept", d.name), "question": g.get("question") or question_text(g.get("prompt")),
               "answer": g.get("answer"), "p_answer": r3(g.get("p_answer") or abl.get("p_base")),
               "tokens": g.get("tokens"), "completeness": {"attr_share": r3(cov.get("attr_share"))},
               "gate": {"knockout_rank_corr": r3(g3.get("spearman_top30")), "pass": gates.get("pass")},
               "groups": groups, "other": {"n": len(other), "attr": r3(sum(n.get("attr", 0) for n in other))},
               "edges": edges, "p_base": r3(abl.get("p_base")),
               "joint": {"p": r3((abl.get("joint") or {}).get("p")), "p_random": [r3(v) for v in (abl.get("joint") or {}).get("p_random", [])],
                         "n": sum(1 for n in g["nodes"] if n.get("type", "neuron") == "neuron" and n.get("neuron"))} if abl.get("joint") else None}
        (OUT / f"{d.name}.json").write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        index.append({"slug": d.name, "concept": out["concept"], "question": out["question"], "answer": out["answer"]})
        print(d.name, len(groups), "groups,", len(edges), "links")
    (OUT / "index.json").write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    print(f"{len(index)} graphs -> {OUT}")


if __name__ == "__main__":
    main()

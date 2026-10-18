"""Step 2 of ATTRIB_GRAPHS.md: Luna groups an attribution graph's neurons into labelled boxes, in rows.

    python -I -X utf8 explainer/gen/map/attrib_groups.py [slug ...]      # default: every graph without groups yet

Reads D:/already-clean-data/attrib_graphs/<slug>/graph.json (from the research session's step 1) and writes
groups.json next to it: {"groups": [{"label", "row", "nodes": [node ids]}], "other": [node ids], "luna": {...}}.
Rows: "answer" (writes the answer), "concept" (parts of what is asked about), "clues" (neurons at the question's
words that pick up what it asks). Evidence per neuron: layer, the token it sits on, its attribution, its checked
convenience name / tested jobs, blind labels and the words it pushes in the direction it fired.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "site" / "map" / "data"
ROOT = Path(os.environ.get("MAP_GRAPHS", "D:/already-clean-data/attrib_graphs"))
sys.path.insert(0, os.environ.get("MAP_API_SRC", "C:/dev/already-clean/src"))
import oai  # noqa: E402

SYSTEM = (
    "You organise an attribution graph from a language model into a few labelled groups for a lay reader, like the "
    "groups in published circuit diagrams. Each item is one neuron at one word of the question, with what we know "
    "about it. Group neurons that do the same job for this question (at most 10 groups; a group can be one neuron). "
    "Give each group a short plain label (2-5 words) and a row: 'answer' = writes or selects the answer itself; "
    "'concept' = parts of the thing being asked about; 'clues' = picks up a clue or word in the question. "
    "Put neurons you can't place in 'other'. Labels are convenience labels; don't overclaim. "
    'Reply as JSON: {"groups": [{"label": "...", "row": "answer"|"concept"|"clues", "nodes": ["<id>", ...]}], '
    '"other": ["<id>", ...]}')


def neuron_info():
    nm = json.loads((DATA / "neuronmap.json").read_text(encoding="utf-8"))
    N = nm["neurons"]; info = {}
    for k in range(len(N["L"])):
        key = f"L{N['L'][k]}:{N['i'][k]}"
        bits = [nm["job_types"][j]["label"] for j in range(len(nm["job_types"])) if N["jt"][k] & (1 << j)]
        info[key] = {"name": nm["names"][N["nm"][k]] if N["nm"][k] >= 0 else None, "jobs": bits}
    return info


def push_words(L, n):
    p = DATA / "push" / f"L{L}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8")).get(str(n))


def describe(node, info, labels):
    L, n = node["layer"], int(node["neuron"].split(":")[1])
    pw = push_words(L, n)
    side = 0 if node.get("act", 0) >= 0 else 1
    parts = [f'{node["id"]}: layer {L}, at word "{node.get("token", "?")}", attribution {node.get("attr", 0):+.2f}']
    i = info.get(node["neuron"]) or {}
    if i.get("name"):
        parts.append(f'name "{i["name"]}"')
    if i.get("jobs"):
        parts.append("jobs: " + ", ".join(i["jobs"][:4]))
    if pw:
        parts.append("pushes: " + " | ".join(w.strip() for w in pw[side][:8] if w.strip()))
    lab = labels.get(node["neuron"])
    if lab:
        parts.append("fires on: " + lab[:120])
    return "; ".join(parts)


def labels_by_neuron():
    out = {}
    for L in range(36):
        p = DATA / "n" / f"L{L}.json"
        if not p.exists():
            continue
        for i, v in json.loads(p.read_text(encoding="utf-8")).items():
            lab = next((f["title"] for f in v.get("findings", []) if f.get("kind") == "label" and (f.get("metric") or {}).get("clean")), None)
            if lab:
                out[f"L{L}:{i}"] = lab
    return out


def main():
    slugs = sys.argv[1:] or sorted(p.parent.name for p in ROOT.glob("*/graph.json") if not (p.parent / "groups.json").exists())
    info, labels = neuron_info(), labels_by_neuron()
    for s in slugs:
        g = json.loads((ROOT / s / "graph.json").read_text(encoding="utf-8"))
        nodes = [n for n in g["nodes"] if n.get("neuron")]
        nodes.sort(key=lambda n: -abs(n.get("attr", 0)))
        text = (f'Question: {g.get("question") or g.get("prompt", "")}\nAnswer: {g.get("answer")}\n\nNeurons:\n' +
                "\n".join(describe(n, info, labels) for n in nodes[:150]))
        r = oai.chat_json(text, system=SYSTEM)
        known = {n["id"] for n in nodes}
        groups = [{"label": x.get("label", "?"), "row": x.get("row", "concept"),
                   "nodes": [i for i in x.get("nodes", []) if i in known]} for x in r.get("groups", [])]
        for x in groups:
            x["members"] = x["nodes"]                 # the name step 3 (attrib_graph.py ablate) reads
        groups = [x for x in groups if x["nodes"]]
        placed = {i for x in groups for i in x["nodes"]}
        out = {"groups": groups, "other": sorted(known - placed), "luna": {"usage": dict(oai.usage)}}
        (ROOT / s / "groups.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        print(s, [(x["row"], x["label"], len(x["nodes"])) for x in groups])


if __name__ == "__main__":
    main()

"""Join the blind labels, Jev grades and intruder scores with the answer key; summarise by unit type.

Usage: python src/summarise_gg.py <model>   (reads results/*_<model>* files, writes results/summary_<model>.md)
"""
import json
import sys
from collections import defaultdict
from pathlib import Path
from modelmapper.core.settings import REPO_ROOT

ROOT = REPO_ROOT
name = sys.argv[1].split("/")[-1]
R = ROOT / "results"
key = json.loads((R / f"dossiers_{name}_KEY.json").read_text(encoding="utf-8"))
labels = json.loads((R / f"labels_{name}.json").read_text(encoding="utf-8"))
intr_path = R / f"intruder_{name}.json"
intr = json.loads(intr_path.read_text(encoding="utf-8")) if intr_path.exists() else {}
merged_path = R / f"scan_{name}_merged.json"
scan = json.loads((merged_path if merged_path.exists() else R / f"scan_{name}.json").read_text(encoding="utf-8"))


def group(role):
    if role.startswith("candidate"):
        return "candidate"
    if role.startswith("exploratory"):
        return "exploratory"
    return role


rows, by = [], defaultdict(list)
for code, k in key.items():
    lab = labels.get(code)
    if not lab:
        continue
    uid = k["unit"]
    row = {"code": code, "unit": uid, "role": k["role"], "group": group(k["role"]),
           "intruder_pos": (intr.get(uid, {}).get("pos") or {}).get("mean_p_intruder"),
           "intruder_neg": (intr.get(uid, {}).get("neg") or {}).get("mean_p_intruder"),
           "gg_near_pos": scan["units"][uid].get("pos_near_phrase"), "gg_near_neg": scan["units"][uid].get("neg_near_phrase"),
           "dneg_removed": k["double_negative_removed_from_top40"]}
    for p in ("pos", "neg"):
        l = lab["luna"].get(p) or {}
        row[f"{p}_verdict"] = l.get("verdict")
        row[f"{p}_expl"] = l.get("explanation", "")
        row[f"{p}_covered"] = lab.get(f"jev_covered_{p}")
        row[f"{p}_random"] = lab.get(f"random_fit_{p}")
        row[f"{p}_clean"] = lab.get(f"clean_{p}")
    rows.append(row)
    by[row["group"]].append(row)

out = [f"# Experiment GG summary: {name}", "", f"Scan: {scan['tokens_scanned']:,} tokens.", "",
       "## By unit type", "",
       "| type | n | pos clean | neg clean | pos verdicts (ONE/TWO/MIX/NONE) | neg verdicts | intruder P pos | intruder P neg |",
       "|---|---|---|---|---|---|---|---|"]
for g in ("candidate", "exploratory", "random_neuron", "planted_mixture"):
    rs = by.get(g, [])
    if not rs:
        continue
    vc = lambda p, v: sum(r[f"{p}_verdict"] == v for r in rs)
    mean = lambda xs: f"{sum(xs)/len(xs):.2f}" if xs else "-"
    out.append(f"| {g} | {len(rs)} | {sum(bool(r['pos_clean']) for r in rs)} | {sum(bool(r['neg_clean']) for r in rs)} | "
               + " | ".join(f"{vc(p,'ONE')}/{vc(p,'TWO_RELATED')}/{vc(p,'MIXTURE')}/{vc(p,'NONE')}" for p in ("pos", "neg"))
               + f" | {mean([r['intruder_pos'] for r in rs if r['intruder_pos'] is not None])}"
               + f" | {mean([r['intruder_neg'] for r in rs if r['intruder_neg'] is not None])} |")
out += ["", "## Every unit", ""]
for g in ("candidate", "exploratory", "planted_mixture", "random_neuron"):
    for r in sorted(by.get(g, []), key=lambda r: r["unit"]):
        out += [f"### {r['unit']} ({r['role']}, dossier {r['code']})",
                f"- intruder P pos/neg: {r['intruder_pos']}/{r['intruder_neg']} | top-40 firings next to the phrase pos/neg: "
                f"{r['gg_near_pos']}/{r['gg_near_neg']} | double-negative firings removed: {r['dneg_removed']}"]
        for p in ("pos", "neg"):
            out.append(f"- **{p}** [{r[f'{p}_verdict']}] covered {r[f'{p}_covered']}/20, random fit "
                       f"{r[f'{p}_random']}/20, clean={r[f'{p}_clean']}: {r[f'{p}_expl']}")
        out.append("")
(R / f"summary_{name}.md").write_text("\n".join(out), encoding="utf-8")
print("\n".join(out[:14]))

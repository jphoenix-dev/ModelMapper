"""Build the published-labels file that `modelmapper lookup` reads: modelmapper/data/labels/Qwen3-8B_v3.json.gz.

    python scripts/build_label_data.py <explainer data> [--results <v3_results.json>] [--out <file.json.gz>]

The explainer data is either its per-neuron folder (data/neurons/L<k>.json: the v3 and v3x runs, 11,000 neurons) or
an older single v3.json: labels, verdicts, AUCs, generative-check words, job types, push words both ways, knockout
concepts. "Verified" is the scorecard's rule: the bin `intelligible`, or `decoys-too-easy (intelligible)` (verified
at the edge, at_edge = true, a reclassification made after the runs; modelmapper.scorecard). v3_results.json is the run's scorecard (modelmapper.scorecard);
when given, it adds what v3.json leaves out (each neuron's knockout concept count and largest drop, the verified sides
and the triage crossed with knockouts) and every shared field is checked to agree.

The output is the scorecard's per-neuron format (gate / apos / aneg, jobtype, knockout, triage), plus per part the
short label and the generative check's AUC and words, the push words per side, and the knockout concepts, so the same
reader handles this file and a user's own `<prefix>_results.json`.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

PARTS = {"gate": "gate", "apos": "pos", "aneg": "neg"}      # scorecard board -> v3.json key
META = {
    "model": "Qwen/Qwen3-8B",
    "source": "v3 and v3x labelling runs, 21 Oct 2026 (11,000 neurons: 950 random and 50 bridge in v3, 10,000 knockout-linked in v3x)",
    "labeller": "gpt-6-luna",
    "judge": "Jev",
    "method": "v3 (modelmapper 0.2.0, LABELLING.md)",
    "rules": {"verified": "exam AUC >= 0.75 and AUC - control >= 0.2, and the list exam passes (bin 'intelligible'; or "
                          "'decoys-too-easy (intelligible)', verified at the edge)",
              "exam_auc_min": 0.75, "exam_margin_min": 0.2,
              "at_edge": "both exams passed but a control scored above 0.65 (bin 'decoys-too-easy (intelligible)'): verified "
                         "with lower confidence, a reclassification made after the runs (the runs' files say not verified)",
              "generative": "AUC >= 0.75, own-label words against surface-label words; run only on verified labels",
              "generative_auc_min": 0.75,
              "job_type": "amplifier if cos(read, write) >= 0.2, eraser if <= -0.2, else redirector", "job_cut": 0.2,
              "knockout": "concepts listed: drop >= 0.3 in the published map, up to 8; profile as in modelmapper.scorecard"},
    "fields": {"set": "random / bridge (v3) or knockout-linked (v3x)",
               "gate/apos/aneg": "the gate, the + side and the - side: label, short (a heuristic summary of it), bin, verified, "
                                 "at_edge, list_pass, auc (exam), auc_control, gen_pass, gen_auc, gen_words (the first 14)",
               "jobtype": "per side, with the cosine", "pushes": "the words each side's write raises most (logit lens)",
               "knockout": "profile, n_concepts, max_drop, concepts [[name, drop], ...]", "triage": "as in modelmapper.scorecard"},
}


def r3(x):
    return None if x is None else round(float(x), 3)


def build(v3: dict, results: dict | None) -> dict:
    res = (results or {}).get("neurons", {})
    out = {}
    for x in v3["neurons"]:
        u = x["id"]
        s = res.get(u, {})
        n = {"layer": x["L"], "set": x.get("set")}
        for board, key in PARTS.items():
            p = x.get(key)
            if p is None:
                n[board] = None; continue
            n[board] = {"label": p["label"], "short": p.get("short"), "bin": p["bin"], "verified": p["bin"] in ("intelligible", "decoys-too-easy (intelligible)"),
                        "at_edge": p["bin"] == "decoys-too-easy (intelligible)",
                        "list_pass": p.get("list"), "auc": r3(p.get("exam")), "auc_control": r3(p.get("ctrl")),
                        "gen_pass": p.get("gen"), "gen_auc": r3(p.get("gen_auc")), "gen_words": p.get("words") or []}
            if s:
                q = s[board]
                assert (q["label"], q["verified"] or q["bin"] == "decoys-too-easy (intelligible)", q["auc"], q["gen_pass"]) == (p["label"], n[board]["verified"], p["exam"], p["gen"]), (u, board)
        j = x["job"]
        n["jobtype"] = {"apos": j["pos"], "c_pos": r3(j["c_pos"]), "aneg": j["neg"], "c_neg": r3(j["c_neg"])}
        n["pushes"] = {"apos": x.get("pushes") or [], "aneg": x.get("pushes_neg") or []}
        k = s.get("knockout", {})
        n["knockout"] = {"profile": x["ko_profile"], "n_concepts": k.get("n_concepts"), "max_drop": k.get("max_drop"),
                         "concepts": [[c["c"], r3(c["drop"])] for c in x.get("ko", [])]}
        t = s.get("triage", {})
        n["triage"] = {"read": x["triage"], "verified_sides": t.get("verified_sides",
                                                                     [b for b in ("apos", "aneg") if (n[b] or {}).get("verified")]),
                       "with_knockout": t.get("with_knockout")}
        if s:
            assert k.get("profile") == x["ko_profile"] and t.get("read") == x["triage"], u
            assert s.get("jobtype") == n["jobtype"], u
        out[u] = n
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("v3", help="the explainer's data/neurons folder, or an older v3.json")
    ap.add_argument("--results", help="the run's v3_results.json (scorecard)")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "modelmapper" / "data" / "labels" / "Qwen3-8B_v3.json.gz"))
    a = ap.parse_args(argv)
    src = Path(a.v3)
    if src.is_dir():
        recs = [x for f in sorted(src.glob("L*.json"), key=lambda f: int(f.stem[1:]))
                for x in json.loads(f.read_text(encoding="utf-8"))["neurons"].values()]
        v3 = {"neurons": sorted(recs, key=lambda x: (x["L"], int(x["id"].split(":")[1])))}
    else:
        v3 = json.loads(src.read_text(encoding="utf-8"))
    results = json.loads(Path(a.results).read_text(encoding="utf-8")) if a.results else None
    neurons = build(v3, results)
    data = {"_meta": {**META, "n_neurons": len(neurons), "scorecard": v3.get("scorecard")}, "neurons": neurons}
    raw = json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f, gzip.GzipFile(filename="", mode="wb", fileobj=f, compresslevel=9, mtime=0) as g:
        g.write(raw)                                              # mtime 0, no name: the same input gives the same bytes
    print(f"{len(neurons)} neurons -> {out} ({out.stat().st_size / 1e3:.0f} kB; {len(raw) / 1e6:.2f} MB uncompressed)")


if __name__ == "__main__":
    main()

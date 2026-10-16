"""Build an answer key: a few marker concept -> neuron associations an install should rediscover (src/markers.py).

A marker must hold up in EVERY independent prompt set:
  - the neuron ranks in the concept's top --max-rank by attribution;
  - knocking it out alone lowers mean P(answer) by at least --min-drop;
  - its blind label (recall pole) names the concept: it contains one of answer_keys/keywords.json's words.
One marker per concept (the one with the largest worst-case knockout drop); concepts with none are listed as such.

From two existing concept-pipeline runs with independently written prompts (how the Qwen3-8B key was built):
    python scripts/make_answer_key.py from-runs Qwen/Qwen3-8B results/concepts_v3 results/concepts_v3b \
        answer_keys/Qwen3-8B.json --names CR-2 CR-2b
Fresh: the labeller writes --sets independent prompt sets per concept, then markers are labelled from this
install's atlas (needs data/atlas_<model>.pt and the labeller; no judge):
    python scripts/make_answer_key.py fresh Qwen/Qwen3-0.6B answer_keys/Qwen3-0.6B.json "Paris" "Tokyo" ...
"""
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
KEYWORDS = json.loads((REPO / "answer_keys" / "keywords.json").read_text(encoding="utf-8"))
DEFAULT_RULES = {   # pass fractions as calibrated for the published keys (see "calibration" in each key)
    "tier1": {"what": "shipped prompts: the same measurement should reproduce near-exactly on any machine",
              "marker_ok": "rank <= rank_factor * key rank + rank_slack and drop >= drop_factor * key drop",
              "rank_factor": 2, "rank_slack": 3, "drop_factor": 0.5, "pass_fraction": 0.9},
    "tier2": {"what": "fresh recall prompts written by your labeller",
              "marker_ok": "rank <= max_rank and drop >= min_drop", "max_rank": 20, "min_drop": 0.03,
              "pass_fraction": 0.6},
    "tier3": {"what": "fresh blind labels for the marker neurons, from your own atlas and labeller",
              "marker_ok": "the recall-pole label contains one of the marker's keywords", "pass_fraction": 0.6},
}


def from_run_dir(d: Path, tag: str) -> dict:
    """{concept: {"neurons": [{rank, layer, neuron, drop, sign}], "baseline_p", "labels", "spec"}} from pipeline output."""
    out = {}
    for f in sorted(d.glob(f"*__{tag}.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        if r.get("status") != "ok":
            continue
        base = r["baseline_p"]
        drop = {(s["layer"], s["neuron"]): base - s["p_answer"] for s in r["single_ablation"]}
        neurons = [{"rank": n["rank"], "layer": n["layer"], "neuron": n["neuron"], "sign": n["sign_on_recall"],
                    "drop": round(drop[(n["layer"], n["neuron"])], 4)}
                   for n in r["top_neurons"][:20] if (n["layer"], n["neuron"]) in drop]
        spec_f = d / f"{f.name.split('__')[0]}_spec.json"
        out[r["concept"]] = {"neurons": neurons, "baseline_p": base, "labels": r.get("labels") or {},
                             "spec": json.loads(spec_f.read_text(encoding="utf-8")) if spec_f.exists() else None,
                             "recall": r.get("recall_kept")}
    return out


def select(runs: list[dict], names: list[str], labels: dict, max_rank: int, min_drop: float):
    """runs: one {concept: row} per prompt set. labels: {concept: {unit: {"pos": {...}, "neg": {...}}}}."""
    from markers import keyword_hit, uid
    markers, skipped = [], {}
    for concept in sorted(set.intersection(*(set(r) for r in runs))):
        kws = KEYWORDS.get(concept)
        if not kws:
            skipped[concept] = "no keywords in answer_keys/keywords.json"
            continue
        per = [{(n["layer"], n["neuron"]): n for n in r[concept]["neurons"]} for r in runs]
        best = None
        for u in per[0]:
            if not all(u in p and p[u]["rank"] <= max_rank and p[u]["drop"] >= min_drop for p in per):
                continue
            sign = per[0][u]["sign"]
            text = ((labels.get(concept) or {}).get(uid(*u)) or {}).get(sign, {}).get("explanation", "")
            hits = keyword_hit(text, kws)
            if not hits:
                continue
            worst = min(p[u]["drop"] for p in per)
            if best is None or worst > best[0]:
                best = (worst, u, sign, text, hits)
        if best is None:
            skipped[concept] = f"no neuron in the top {max_rank} of every set with drop >= {min_drop} and a matching label"
            continue
        worst, (l, n), sign, text, hits = best
        markers.append({
            "concept": concept, "answer": runs[0][concept]["spec"]["answer"] if runs[0][concept].get("spec") else None,
            "unit": uid(l, n), "layer": l, "neuron": n, "sign": sign, "keywords": kws, "job": text, "label_hits": hits,
            "evidence": [{"set": nm, "rank": p[(l, n)]["rank"], "drop": p[(l, n)]["drop"],
                          "baseline_p": r[concept]["baseline_p"]} for nm, p, r in zip(names, per, runs)],
            "spec": runs[0][concept].get("spec"),   # tier 1 re-uses these exact prompts
        })
    return markers, skipped


def write_key(model, out, markers, skipped, sources, args):
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    key = {"model": model, "format": 1, "built": time.strftime("%Y-%m-%d"), "git": head,
           "selection": {"max_rank": args.max_rank, "min_drop": args.min_drop, "sources": sources,
                         "rule": "in every prompt set: attribution rank <= max_rank, single-knockout P(answer) drop >= "
                                 "min_drop; label names the concept (keywords.json); best worst-case drop per concept"},
           "pass_rules": DEFAULT_RULES, "markers": markers, "concepts_without_marker": skipped}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(key, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{len(markers)} markers -> {out}")
    for m in markers:
        ev = " | ".join(f"{e['set']}: rank {e['rank']}, drop {e['drop']:.2f}" for e in m["evidence"])
        print(f"  {m['concept']:<24} {m['unit']:<11} {ev} | label hits {m['label_hits']}")
    for c, why in skipped.items():
        print(f"  (no marker) {c}: {why}")


def calibrate(key_path, checks, passes):
    """Record verify_answer_key.py results on our own machine in the key, and set each tier's pass fraction."""
    key = json.loads(Path(key_path).read_text(encoding="utf-8"))
    runs = []
    for f in checks:
        r = json.loads(Path(f).read_text(encoding="utf-8"))
        ctl = "_control" in Path(f).name
        run = {"file": Path(f).name, "tier": r["tier"], "control": ctl, "found": r["found"], "markers": r["markers"],
               "missed": [x["concept"] for x in r["rows"] if not x["ok"]] if not ctl else None, "time": r["time"]}
        if r["tier"] == 3 and r["rows"] and "control_total" in r["rows"][0]:
            run["keyword_control"] = f"{r['rows'][0]['control_matches']}/{r['rows'][0]['control_total']} label x other-marker pairings matched"
        runs.append(run)
    for tier, frac in passes.items():
        key["pass_rules"][tier]["pass_fraction"] = frac
    key["calibration"] = {
        "about": "verify_answer_key.py run on the key builders' own machine (RTX 5090). Tier 2 runs used prompt sets "
                 "written fresh by the labeller, never seen by the key; tier 3 used labels written fresh from an atlas "
                 "of FineWeb shard 001_00000, which neither the key nor the published results used. Controls look for "
                 "each marker in another marker's concept (tiers 1-2) or match each label against other markers' "
                 "keywords (tier 3); a meaningful key scores low on them. Pass fractions were set from these runs.",
        "runs": runs}
    Path(key_path).write_text(json.dumps(key, indent=1, ensure_ascii=False), encoding="utf-8")
    for run in runs:
        print(f"  tier {run['tier']}{' control' if run['control'] else ''}: {run['found']}/{run['markers']}"
              + (f" | missed {run['missed']}" if run.get("missed") else "") + (f" | {run['keyword_control']}" if run.get("keyword_control") else ""))
    print(f"pass fractions: { {t: key['pass_rules'][t]['pass_fraction'] for t in ('tier1', 'tier2', 'tier3')} } -> {key_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    a1 = sub.add_parser("from-runs"); a1.add_argument("model"); a1.add_argument("run_dirs", nargs=2); a1.add_argument("out")
    a1.add_argument("--tag", default="clamp_reply"); a1.add_argument("--names", nargs=2, default=["A", "B"])
    a3 = sub.add_parser("calibrate"); a3.add_argument("key"); a3.add_argument("checks", nargs="+")
    a3.add_argument("--tier1", type=float); a3.add_argument("--tier2", type=float); a3.add_argument("--tier3", type=float)
    a2 = sub.add_parser("fresh"); a2.add_argument("model"); a2.add_argument("out"); a2.add_argument("concepts", nargs="+")
    a2.add_argument("--sets", type=int, default=2); a2.add_argument("--workdir", help="default results/answer_key_build/<model>")
    for p in (a1, a2, a3):
        p.add_argument("--max-rank", type=int, default=5); p.add_argument("--min-drop", type=float, default=0.1)
    args = ap.parse_args()

    if args.mode == "calibrate":
        calibrate(args.key, args.checks, {t: v for t, v in (("tier1", args.tier1), ("tier2", args.tier2),
                                                            ("tier3", args.tier3)) if v is not None})
        return
    if args.mode == "from-runs":
        runs = [from_run_dir(REPO / d, args.tag) for d in args.run_dirs]
        labels = {c: r["labels"] for c, r in runs[0].items()}       # labels written in the first run
        markers, skipped = select(runs, args.names, labels, args.max_rank, args.min_drop)
        write_key(args.model, args.out, markers, skipped, [f"{n}: {d}" for n, d in zip(args.names, args.run_dirs)], args)
        return

    import markers as mk
    from config import short_name
    work = Path(args.workdir or REPO / "results" / "answer_key_build" / short_name(args.model))
    mk.setup_env(work / "set1", discovery=True)
    from pipeline_fast import FastPipeline
    pipe = FastPipeline(args.model)
    runs, names = [], [f"set{i + 1}" for i in range(args.sets)]
    for nm in names:
        mk.set_out_dir(work / nm)                    # a new folder = the labeller writes a new, independent prompt set
        res = {}
        for c in args.concepts:
            r = mk.run_concept(pipe, c)
            top = ", ".join("%s (%.2f)" % (mk.uid(n["layer"], n["neuron"]), n["drop"]) for n in r.get("neurons", [])[:3])
            print(f"[{nm}] {c:<22} {r['status']:<13} recall {r['recall']}/{r['recall_total']}"
                  + (f" | top: {top}" if top else ""), flush=True)
            if r["status"] == "ok":
                res[c] = r
        (work / nm / "marker_runs.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        runs.append(res)
    # label every candidate (top max_rank in all sets) from this install's atlas, blind, labeller only
    cands = {}
    for c in set.intersection(*(set(r) for r in runs)):
        ranks = [{(n["layer"], n["neuron"]): n["rank"] for n in r[c]["neurons"]} for r in runs]
        cands[c] = [u for u in ranks[0] if all(rk.get(u, 99) <= args.max_rank for rk in ranks)]
    units = sorted({u for us in cands.values() for u in us})
    # labels already written in an earlier run of this folder are kept (they are a fresh LLM call each time, so
    # re-labelling would quietly change earlier results); only new candidates are labelled
    lab_f = work / "labels.json"
    lab = json.loads(lab_f.read_text(encoding="utf-8")) if lab_f.exists() else {}
    todo = [u for u in units if mk.uid(*u) not in lab]
    print(f"labelling {len(todo)} new candidate neurons ({len(units) - len(todo)} labelled before)", flush=True)
    if todo:
        lab.update(mk.label_units(args.model, todo, work / "dossiers_markers"))   # FineWeb text: not published
    labels = {c: {mk.uid(*u): lab.get(mk.uid(*u), {}) for u in us} for c, us in cands.items()}
    lab_f.write_text(json.dumps(lab, indent=1, ensure_ascii=False), encoding="utf-8")
    markers, skipped = select(runs, names, labels, args.max_rank, args.min_drop)
    write_key(args.model, args.out, markers, skipped, [f"{n}: fresh prompt set ({work / n})" for n in names], args)


if __name__ == "__main__":
    main()

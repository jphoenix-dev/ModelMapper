"""Loop v2 CHECK + report over a run's drained results (no model, no API; seconds). Re-runnable at any time.

Reads <run>/results/map/<batch>/records.jsonl (every finished batch) and the frame (section, fame tier, named/common
tag). Reports, with every definition stated in the output:
  coverage   per status, and recalled share by section / fame tier / named-common
  maps       k_mean / k_flip distributions; share of concepts whose top-5 has a 'say' neuron (weights-based role);
             roles of the top-5; by named vs common
  overlap    distinct neurons in any top-30 / top-5; 'dedicated' share = a concept's top-10 neurons that are in no
             other mapped concept's top-30 (population-dependent: P2 law); neurons shared within vs across sections
  heads      the heads most often among concepts' 5 most damaging single-head knockouts
  checks     null: per concept, the larger |mean drop| of its 2 null neurons vs its top-1 single drop (must be far
             apart); P4-style: section 'family' neurons from the first half of finished batches predict the second
             half's top-30 (own-section vs other-section hit rate, as p4_family.py)
  run        GPUs, per-step time, per-concept wall time by GPU
Output: <run>/report.txt and report.json.  Usage: python src/loop_v2/report.py <run_dir> [frame.json]
"""
import glob
import json
import statistics as stx
import sys
from collections import Counter, defaultdict
from pathlib import Path

run = Path(sys.argv[1])
frame_p = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("D:/already-clean-data/loop_v2/frame/frame.json")


def main():
    F = {x["concept"]: x for x in json.loads(frame_p.read_text(encoding="utf-8"))["draw"]}
    recs, batches = [], sorted(glob.glob(str(run / "results/**/map/*/records.jsonl"), recursive=True))   # drain layout may nest
    for b in batches:
        for line in Path(b).read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line); r["_batch"] = Path(b).parent.name; recs.append(r)
    seen = {}
    for r in recs:                                       # a retried batch may repeat a concept: the later record wins
        seen[r["concept"]] = r
    recs = list(seen.values())
    kind = lambda c: F.get(c, {}).get("kind", "untagged")
    md = lambda xs: stx.median(xs) if xs else float("nan")
    L = [f"Loop v2 report: {run.name} - {len(batches)} finished batches, {len(recs)} concepts (frame {len(F)})"]
    st = Counter(r["row"]["status"] for r in recs)
    L.append(f"status: {dict(st)}")
    ok = [r for r in recs if r["row"]["status"] == "ok"]
    for key, fn in (("section", lambda c: F.get(c, {}).get("section")), ("fame tier", lambda c: F.get(c, {}).get("tier")), ("named/common", kind)):
        g = defaultdict(lambda: [0, 0])
        for r in recs:
            if r["row"]["status"] in ("ok", "not_recalled"):
                g[fn(r["concept"])][0] += r["row"]["status"] == "ok"; g[fn(r["concept"])][1] += 1
        L.append(f"recalled by {key}: " + ", ".join(f"{k} {a}/{n} ({a / n:.0%})" for k, (a, n) in sorted(g.items(), key=lambda x: str(x[0]))))
    L.append("\nmaps (recalled concepts):")
    for kd in ("named", "common", "untagged"):
        rs = [r for r in ok if kind(r["concept"]) == kd]
        if not rs:
            continue
        km = [r["row"]["k_mean"] for r in rs if r["row"].get("k_mean")]; kf = [r["row"]["k_flip"] for r in rs if r["row"].get("k_flip")]
        say5 = sum(any(x["role"] == "say" for x in r["v2"]["roles_top30"][:5]) for r in rs)
        roles5 = Counter(x["role"] for r in rs for x in r["v2"]["roles_top30"][:5])
        L.append(f"  {kd:<8} n={len(rs)}: k_mean <= 5 in {sum(k <= 5 for k in km)}/{len(rs)} (median {md(km)}; none within 50: {len(rs) - len(km)}); "
                 f"k_flip median {md(kf)}; a 'say' neuron in the top-5: {say5}/{len(rs)} ({say5 / len(rs):.0%}); top-5 roles {dict(roles5)}")
    top30 = {r["concept"]: [e["neuron"] for e in r["rank"]["top"]] for r in ok}
    cnt = Counter(u for ns in top30.values() for u in set(ns))
    L.append(f"\noverlap: {len(top30)} maps; distinct neurons in any top-30 {len(cnt)}, in any top-5 {len({u for ns in top30.values() for u in ns[:5]})}")
    for kd in ("named", "common"):
        rs = [c for c in top30 if kind(c) == kd]
        if rs:
            ded = [sum(cnt[u] == 1 for u in top30[c][:10]) / 10 for c in rs]
            L.append(f"  {kd:<8}: dedicated share of top-10 (in no other map's top-30) median {md(ded):.0%}, mean {stx.mean(ded):.0%}")
    sec_of = {c: F.get(c, {}).get("section") for c in top30}; by_u = defaultdict(set)
    for c, ns in top30.items():
        for u in ns:
            by_u[u].add(c)
    shared = [cs for cs in by_u.values() if len(cs) >= 2]
    within = sum(len({sec_of[c] for c in cs}) == 1 for cs in shared)
    L.append(f"  neurons shared by >= 2 maps: {len(shared)}; all within one section {within} ({within / max(1, len(shared)):.0%})")
    hc = Counter()
    for r in ok:
        hk = r["rank"].get("head_knockout_mean", {})
        for h, _ in sorted(hk.items(), key=lambda x: -x[1])[:5]:
            hc[h] += 1
    L.append("heads most often among a concept's 5 most damaging single-head knockouts: " + ", ".join(f"{h} {n}" for h, n in hc.most_common(12)))
    # checks
    ratios = []
    for r in ok:
        nd = max(abs(sum(v) / len(v)) for v in r["v2"]["null_single_drops"].values())
        ratios.append((nd, r["rank"]["top"][0]["single_drop"]))
    L.append(f"\ncheck null: median |null drop| {md([a for a, _ in ratios]):.3f}, p95 {sorted(a for a, _ in ratios)[int(.95 * len(ratios))] if ratios else float('nan'):.3f}; "
             f"median top-1 drop {md([b for _, b in ratios]):.2f}; concepts whose null >= 20% of top-1: {sum(a >= 0.2 * b for a, b in ratios)}")
    done_b = sorted({r["_batch"] for r in ok}); half = set(done_b[: len(done_b) // 2])
    first = [c for c in top30 if seen[c]["_batch"] in half]; second = [c for c in top30 if seen[c]["_batch"] not in half]
    fam = defaultdict(list)
    for c in first:
        fam[sec_of[c]].append(c)
    allc = Counter(u for c in first for u in set(top30[c])); gen = {u for u, k in allc.items() if k >= 0.15 * max(1, len(first))}
    F_ = {}
    for s, cs in fam.items():
        if len(cs) >= 5:
            fc = Counter(u for c in cs for u in set(top30[c]))
            F_[s] = {u for u, k in fc.items() if k >= 0.2 * len(cs) and u not in gen}
    rows = []
    for c in second:
        s = sec_of[c]
        if s in F_ and F_[s] and len([g for g in F_ if g != s and F_[g]]) >= 1:
            own = len(set(top30[c]) & F_[s]) / len(F_[s])
            oth = stx.mean(len(set(top30[c]) & F_[g]) / len(F_[g]) for g in F_ if g != s and F_[g])
            rows.append((own, oth))
    if rows:
        w = sum(a > b for a, b in rows); l_ = sum(a < b for a, b in rows)
        L.append(f"check P4-style (section neurons from the first {len(first)} maps -> the next {len(second)}): own > other in {w}, < in {l_}; "
                 f"median own {md([a for a, _ in rows]):.3f} vs other {md([b for _, b in rows]):.3f}")
    gpus = Counter(r["v2"]["provenance"]["gpu"] for r in recs)
    L.append(f"\nrun: GPUs {dict(gpus)}")
    for g in gpus:
        t = [sum(r["v2"]["timing_s"].values()) for r in ok if r["v2"]["provenance"]["gpu"] == g]
        if t:
            L.append(f"  {g}: median {md(t):.0f} s per recalled concept (n={len(t)})")
    txt = "\n".join(L)
    (run / "report.txt").write_text(txt + "\n", encoding="utf-8"); print(txt)


if __name__ == "__main__":
    main()

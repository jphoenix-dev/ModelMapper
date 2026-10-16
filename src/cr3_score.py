"""Score CR-3 against its frozen predictions (PREREG CR-3). Written and committed before any CR-3 data.

Reads results/concepts_cr3/wave_free.json (+ results/cr3_jev.json for P4b if present). Prints PASS / FAIL /
NOT TESTABLE per prediction, with the numbers.

Usage: python src/cr3_score.py
"""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def mann_whitney_greater(a, b):
    """One-sided Mann-Whitney U (a > b), normal approximation with tie correction. Returns (U, p)."""
    allv = sorted([(v, 0) for v in a] + [(v, 1) for v in b])
    ranks, i = {}, 0
    vals = [v for v, _ in allv]
    rank = [0.0] * len(allv)
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and vals[j + 1] == vals[i]:
            j += 1
        for k in range(i, j + 1):
            rank[k] = (i + j) / 2 + 1
        i = j + 1
    ra = sum(r for r, (_, g) in zip(rank, allv) if g == 0)
    n1, n2 = len(a), len(b)
    u = ra - n1 * (n1 + 1) / 2
    mu = n1 * n2 / 2
    sd = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    z = (u - mu - 0.5) / sd if sd else 0.0
    return u, 0.5 * math.erfc(z / math.sqrt(2))


def main():
    d = json.loads((ROOT / "results/concepts_cr3/wave_free.json").read_text(encoding="utf-8"))
    wave = json.loads((ROOT / "prompts/concepts_cr3.json").read_text(encoding="utf-8"))["concepts"]
    made = set(wave["made_up_controls"]); people = set(wave["people"])
    real = [c for cat, cs in wave.items() if cat != "made_up_controls" for c in cs]
    missing = [c for c in real + list(made) if c not in d]
    if missing:
        print(f"WARNING: {len(missing)} concepts have no result (crash?): {missing}")
    ok = {c: d[c] for c in real if c in d and d[c].get("status") == "ok"}
    le = lambda k, c: k is not None and k <= c  # noqa: E731
    res = []
    r_rate = len(ok) / len(real); m_rec = sum(d.get(c, {}).get("status") == "ok" for c in made)
    res.append(("P1 recall >=90% real, 0/4 made-up", r_rate >= 0.9 and m_rec == 0, f"{len(ok)}/{len(real)} real ({r_rate:.0%}); made-up recalled {m_rec}/{len(made)}"))
    n = len(ok)
    km = sum(le(v["k_mean"], 5) for v in ok.values()) / n; kd = sum(le(v["k_median"], 5) for v in ok.values()) / n
    rnd = sum(v["random_drops"][-1] < 0.5 for v in ok.values()) / n
    res.append(("P2 k_mean<=5 >=90%, k_median<=5 >=80%, random<0.5 >=95%", km >= .9 and kd >= .8 and rnd >= .95,
                f"k_mean {km:.0%}, k_median {kd:.0%}, random<0.5 {rnd:.0%} (n={n})"))
    kf = sum(le(v["k_flip"], 5) for v in ok.values()) / n
    res.append(("P3a k_flip<=5 >=70%", kf >= .7, f"{kf:.0%}"))
    hard = [v["baseline_mean_logodds"] for v in ok.values() if not le(v["k_flip"], 5)]
    easy = [v["baseline_mean_logodds"] for v in ok.values() if le(v["k_flip"], 5)]
    if len(hard) < 3:
        res.append(("P3b hard-to-flip have higher baseline log-odds", None, f"NOT TESTABLE: only {len(hard)} concepts with k_flip > 5"))
    else:
        u, p = mann_whitney_greater(hard, easy)
        res.append(("P3b hard-to-flip have higher baseline log-odds", p < 0.05, f"hard mean {sum(hard)/len(hard):.2f} vs {sum(easy)/len(easy):.2f}; U={u:.0f}, p={p:.3f}"))
    s = {c: v["steer"] for c, v in ok.items()}
    auto = sum(x["heldout"] for x in s.values()) / n; dflt = sum(x["heldout_default_R1"] for x in s.values()) / n
    rand = sum(x["heldout_random"] for x in s.values()) / n
    res.append(("P4a auto >= 1.5x default, random < 0.05", auto >= 1.5 * dflt and rand < 0.05,
                f"auto {auto:.3f}, default {dflt:.3f} ({auto / dflt if dflt else float('inf'):.2f}x), random {rand:.3f}"))
    jp = ROOT / "results/cr3_jev.json"
    if jp.exists():
        j = json.loads(jp.read_text(encoding="utf-8"))
        res.append(("P4b Jev: auto >= default (sample)", j["auto_mean"] >= j["default_mean"],
                    f"auto {j['auto_mean']:.3f} vs default {j['default_mean']:.3f} on {j['n']} concepts"))
    else:
        res.append(("P4b Jev: auto >= default (sample)", None, "NOT RUN yet (results/cr3_jev.json missing)"))
    pp = [s[c]["heldout"] for c in s if c in people]; npp = [s[c]["heldout"] for c in s if c not in people]
    if not pp:
        res.append(("P5 people >0 and below non-people", None, "NOT TESTABLE: no people recalled"))
    else:
        mp, mn = sum(pp) / len(pp), sum(npp) / len(npp)
        res.append(("P5 people >0 and below non-people", mp > 0 and mp < mn, f"people {mp:.3f} (n={len(pp)}) vs non-people {mn:.3f}"))
    for name, ok_, txt in res:
        print(f"{'PASS' if ok_ else 'FAIL' if ok_ is False else 'N/T '}  {name}: {txt}")
    (ROOT / "results/cr3_score.json").write_text(json.dumps([{"prediction": a, "pass": b, "detail": c} for a, b, c in res], indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

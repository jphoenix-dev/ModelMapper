"""Interference test summary (no model, no API). Reads results/interference.jsonl.
delta = loss(neuron zeroed) - loss(as is), in nats per predicted token: > 0 the neuron's push HELPED predict the real
text; < 0 it HURT. Reported for the next token (delta1) and the mean over the next T tokens (deltaT).
Groups: concept neurons' firings labelled A (about the concept), B (related), C (unrelated); control neurons' firings
(random neurons judged against the same concept). Null: a random same-layer neuron zeroed in the same pass.
Usage: python src/interference_summary.py -> results/interference_summary.txt
"""
import json
import math
import statistics as stx
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sign_p(pos, neg):
    n = pos + neg
    if not n:
        return 1.0
    k = max(pos, neg)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n)


def main():
    R0 = [json.loads(l) for l in (ROOT / "results/interference.jsonl").read_text(encoding="utf-8").splitlines()]
    R = [r for r in R0 if r["T"] > 0]              # a firing on a document's last token has nothing after it to predict
    n_end = len(R0) - len(R)
    n_act = sum(not r["act_ok"] for r in R)
    R = [r for r in R if r["act_ok"]]              # firings that don't reproduce the profile's activation (gate 2)
    for r in R:
        b, z, nl = r["logp"]
        r["d1"] = b[0] - z[0]; r["dT"] = stx.mean(x - y for x, y in zip(b, z))
        r["n1"] = b[0] - nl[0]; r["nT"] = stx.mean(x - y for x, y in zip(b, nl))
        L = int(r["neuron"].split(":")[0][1:]); r["band"] = "L0-15" if L < 16 else "L16-23" if L < 24 else "L24-29" if L < 30 else "L30-35"
    g = defaultdict(list)
    for r in R:
        g[("control" if r["control"] else "concept", r["label"])].append(r)
    eps = 1e-6
    L = [f"Interference test: {len(R)} contexts ({sum(not r['control'] for r in R)} concept-neuron, {sum(r['control'] for r in R)} control-neuron firings); "
         f"T = {stx.median(r['T'] for r in R)} tokens (median); left out: {n_act} whose activation didn't reproduce the profile's (gate 2); "
         f"left out: {n_end} firings on a document's last token (nothing to predict)",
         "delta = loss(zeroed) - loss(as is), nats/token: > 0 the push HELPED predict the real text, < 0 it HURT",
         f"\n{'group':<14}{'n':>6}{'median d1':>11}{'mean d1':>9}{'helped/hurt d1':>16}{'p':>9}{'median dT':>11}{'mean dT':>9}{'helped/hurt dT':>16}{'p':>9}{'null mean dT':>13}"]
    for key in sorted(g):
        rs = g[key]
        h1 = sum(r["d1"] > eps for r in rs); u1 = sum(r["d1"] < -eps for r in rs)
        hT = sum(r["dT"] > eps for r in rs); uT = sum(r["dT"] < -eps for r in rs)
        L.append(f"{key[0] + ' ' + key[1]:<14}{len(rs):>6}{stx.median(r['d1'] for r in rs):>11.4f}{stx.mean(r['d1'] for r in rs):>9.4f}{f'{h1}/{u1}':>16}{sign_p(h1, u1):>9.2g}"
                 f"{stx.median(r['dT'] for r in rs):>11.4f}{stx.mean(r['dT'] for r in rs):>9.4f}{f'{hT}/{uT}':>16}{sign_p(hT, uT):>9.2g}{stx.mean(r['nT'] for r in rs):>13.5f}")
    L.append("\nconcept-neuron firings by layer band (mean dT; helped/hurt):")
    for band in ("L0-15", "L16-23", "L24-29", "L30-35"):
        parts = []
        for lab in "ABC":
            rs = [r for r in g[("concept", lab)] if r["band"] == band]
            if rs:
                parts.append(f"{lab}: n={len(rs)} {stx.mean(r['dT'] for r in rs):+.4f} ({sum(r['dT'] > eps for r in rs)}/{sum(r['dT'] < -eps for r in rs)})")
        L.append(f"  {band:<7} " + " | ".join(parts))
    # within-neuron: concept neurons with both on-concept (A/B) and unrelated (C) firings
    by_n = defaultdict(lambda: {"on": [], "off": []})
    for r in R:
        if not r["control"]:
            by_n[r["neuron"]]["on" if r["label"] in "AB" else "off"].append(r["dT"])
    both = [(stx.mean(v["on"]), stx.mean(v["off"])) for v in by_n.values() if v["on"] and v["off"]]
    w = sum(a > b for a, b in both); l_ = sum(a < b for a, b in both)
    L.append(f"\nwithin concept neurons with both kinds of firing ({len(both)}): mean dT on-concept/related {stx.mean(a for a, _ in both):+.4f} vs unrelated "
             f"{stx.mean(b for _, b in both):+.4f}; on > off in {w}, off > on in {l_} (sign p {sign_p(w, l_):.2g})")
    # noise-aware (added 17 Oct after the partial data showed the random-neuron null is NOT exact: bf16 rounding sends
    # the two runs apart by ~0.005 nats/token whenever the zeroing changes anything): compare with the null directly
    allnull = sorted(abs(r["nT"]) for r in R); p90 = allnull[int(0.9 * len(allnull))]
    L.append(f"\nnoise-aware: null |dT| median {stx.median(allnull):.5f}, p90 {p90:.5f} (random same-layer neuron zeroed, same pass)")
    L.append(f"{'group':<14}{'n':>6}{'median |dT|':>13}{'null':>9}{'ratio':>7}{'|dT|>null p90':>15}{'mean dT - null':>16}{'+-2SE':>9}{'neuron>null':>13}{'p':>9}")
    for key in sorted(g):
        rs = g[key]
        a = stx.median(abs(r["dT"]) for r in rs); b = stx.median(abs(r["nT"]) for r in rs)
        diff = [r["dT"] - r["nT"] for r in rs]; se = (stx.pstdev(diff) / math.sqrt(len(diff))) if len(diff) > 1 else float("nan")
        w = sum(abs(r["dT"]) > abs(r["nT"]) for r in rs); l_ = sum(abs(r["dT"]) < abs(r["nT"]) for r in rs)
        L.append(f"{key[0] + ' ' + key[1]:<14}{len(rs):>6}{a:>13.5f}{b:>9.5f}{a / b if b else float('nan'):>7.2f}"
                 f"{sum(abs(r['dT']) > p90 for r in rs) / len(rs):>15.1%}{stx.mean(diff):>+16.5f}{2 * se:>9.5f}{f'{w}/{l_}':>13}{sign_p(w, l_):>9.2g}")
    big = [r for r in g[("concept", "C")] if abs(r["dT"]) >= 0.05]
    L.append(f"unrelated firings of concept neurons with |dT| >= 0.05 nats: {len(big)} ({len(big) / max(1, len(g[('concept', 'C')])):.1%}); "
             f"helped {sum(r['dT'] > 0 for r in big)}, hurt {sum(r['dT'] < 0 for r in big)}")
    txt = "\n".join(L)
    (ROOT / "results/interference_summary.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()

"""Stage C summary: Luna's view (relatedness ratings) vs Qwen's own view (distances in its representation space).
(no model, no API)

From meaning_space.json (group centroids of each unit's firing contexts in Qwen's whitened MLP-input space, the unit's
own read directions removed):
  - reference scale: 'same meaning' = mean cosine between snippets of one group (within); 'unrelated' = cosine between
    the two groups of correctly separated SPLICED units (two different neurons' firings - known unrelated)
  - concept neurons' between-meaning cosines placed on that scale, by Luna's 0-3 rating
  - agreement: Spearman between Luna's rating and the cosine (negative = they agree)
  - 'inside close, outside far' (James's Ur-concept question): pairs Luna calls unrelated (3) whose cosine is at least
    the median cosine of pairs Luna calls the same / closely related (0-1)
  - hubs: units with >= 3 meanings whose most central meaning beats the 300-permutation null (p < 0.05); 5% expected by chance
Usage: python src/meaning_space_summary.py  -> results/meaning_space_summary.txt
"""
import json
import statistics as stx
from collections import Counter, defaultdict
from pathlib import Path

from fame_summary import spearman

ROOT = Path(__file__).resolve().parents[1]


def main():
    S = json.loads((ROOT / "results/meaning_space.json").read_text(encoding="utf-8"))
    J = json.loads((ROOT / "results/meaning_judge.json").read_text(encoding="utf-8"))["jobs"]
    md = lambda xs: stx.median(xs) if xs else float("nan")
    by = defaultdict(list)
    for k, v in S.items():
        by[v["kind"]].append((k, v))
    L = [f"Stage C - meanings in Qwen's own space: " + ", ".join(f"{k} {len(v)}" for k, v in by.items()) + " units with >= 2 meaning groups\n"]
    # spliced reference: pairs of groups whose majority sources differ
    spl_between = []
    for k, v in by["spliced"]:
        src = J[k]["source"]; groups = J[k]["result"]["groups"]
        maj = [Counter(src[int(m) - 1] for m in g["members"]).most_common(1)[0][0] if g["members"] else None for g in groups]
        for p in v["pairs"]:
            if maj[p["a"]] is not None and maj[p["b"]] is not None and maj[p["a"]] != maj[p["b"]]:
                spl_between.append(p["cos"])
    within_all = [v["within_mean"] for _, v in S.items() if v["within_mean"] is not None]
    ref_same, ref_unrel = md(within_all), md(spl_between)
    L.append(f"reference: same meaning (within-group) median cosine {ref_same:+.3f}; known-unrelated (spliced, different sources) "
             f"{ref_unrel:+.3f} (n={len(spl_between)})")
    for kind in ("concept", "control"):
        pairs = [p for _, v in by[kind] for p in v["pairs"] if p["luna"] is not None]
        L.append(f"\n{kind} neurons: {len(pairs)} meaning pairs")
        for r in range(4):
            cs = [p["cos"] for p in pairs if p["luna"] == r]
            if cs:
                pos = (md(cs) - ref_unrel) / (ref_same - ref_unrel) if ref_same != ref_unrel else float("nan")
                L.append(f"  Luna rating {r}: n={len(cs):<4} median cosine {md(cs):+.3f} (position on unrelated->same scale: {pos:.2f})")
        if len(pairs) >= 10:
            rho = spearman([p["luna"] for p in pairs], [p["cos"] for p in pairs])
            close = md([p["cos"] for p in pairs if p["luna"] in (0, 1)])
            far_close = [p for p in pairs if p["luna"] == 3 and p["cos"] >= close]
            L.append(f"  Spearman(Luna rating, cosine) = {rho:+.3f} (negative = Luna and Qwen agree)")
            L.append(f"  'inside close, outside far': {len(far_close)}/{sum(p['luna'] == 3 for p in pairs)} pairs Luna calls unrelated sit at least as "
                     f"close as the median 'same/closely related' pair ({close:+.3f})")
    L.append("\nhubs (units with >= 3 meanings; a hub = the most central meaning beats the permutation null at p < 0.05; 5% by chance):")
    for kind in ("concept", "control", "spliced"):
        hs = [v for _, v in by[kind] if "hub_p" in v]
        if hs:
            sig = [v for v in hs if v["hub_p"] < 0.05]
            L.append(f"  {kind:<8} {len(sig)}/{len(hs)} ({len(sig) / len(hs):.0%})")
    L.append("\nexamples - concept neurons with a significant hub (hub meaning <- other meanings):")
    for k, v in by["concept"]:
        if v.get("hub_p", 1) < 0.05:
            others = [l for i, l in enumerate(v["labels"]) if i != v["hub_group"]]
            L.append(f"  {v['units'][0]:<10} hub '{v['hub_label']}' <- {others}  (p={v['hub_p']})")
    L.append("\nexamples - Luna says unrelated, Qwen places close (concept neurons):")
    close_all = md([p["cos"] for _, v in by["concept"] for p in v["pairs"] if p["luna"] in (0, 1)])
    for k, v in by["concept"]:
        for p in v["pairs"]:
            if p["luna"] == 3 and p["cos"] >= close_all:
                L.append(f"  {v['units'][0]:<10} '{v['labels'][p['a']]}' ~ '{v['labels'][p['b']]}' cos {p['cos']:+.3f}")
    txt = "\n".join(L)
    (ROOT / "results/meaning_space_summary.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()

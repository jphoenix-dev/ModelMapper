"""Compare recall concentration across formats on the non-saturating log-odds scale (ISSUES #27).

k_LO3 = fewest neurons (measured order) whose joint ablation lowers mean answer log-odds by >= 3.
Question: is chat/question format (CR-2, V3) really less concentrated than completion format (CR, V0) once the
measure doesn't saturate? Also: does baseline confidence still predict k?

Usage: python src/logodds_compare.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
cr = json.loads((ROOT / "results/concepts/logodds_cr_V0.json").read_text(encoding="utf-8"))
cr2 = json.loads((ROOT / "results/concepts_v3/logodds_cr2_V3.json").read_text(encoding="utf-8"))
k50 = {}
for d in ("concepts", "concepts_v3"):
    for f in (ROOT / "results" / d).glob("*__clamp_reply.json"):
        x = json.loads(f.read_text(encoding="utf-8"))
        if isinstance(x, dict) and "concept" in x:
            k50[(d, x["concept"])] = x.get("k50")


def show(name, data, d):
    print(f"\n{name}")
    print(f"  {'concept':<26} {'base P':>6} {'base LO':>7} {'k50(old)':>8} {'kLO3 attr':>9} {'kLO3 meas':>9}  drop@5 meas / random")
    for c, v in sorted(data.items(), key=lambda t: -t[1]["baseline_mean_p"]):
        r5 = v["drop_random"][3]; m5 = v["drop_measured"][3]
        print(f"  {c:<26} {v['baseline_mean_p']:>6} {v['baseline_mean_logodds']:>7} {str(k50.get((d, c))):>8} "
              f"{str(v['k_LO3_attr']):>9} {str(v['k_LO3_measured']):>9}  {m5:>5} / {r5}")
    ks = [v["k_LO3_measured"] for v in data.values()]
    print(f"  k_LO3 (measured) <= 5: {sum(k is not None and k <= 5 for k in ks)}/{len(ks)}; <= 10: "
          f"{sum(k is not None and k <= 10 for k in ks)}/{len(ks)}; none within 50: {sum(k is None for k in ks)}")


show("CR (completion format, V0)", cr, "concepts")
show("CR-2 (question format, V3)", cr2, "concepts_v3")
both = sorted(set(cr) & set(cr2))
print(f"\nSame concept, both formats ({len(both)}): k_LO3 measured CR vs CR-2")
for c in both:
    print(f"  {c:<26} {cr[c]['k_LO3_measured']} vs {cr2[c]['k_LO3_measured']}   (base P {cr[c]['baseline_mean_p']} vs {cr2[c]['baseline_mean_p']})")

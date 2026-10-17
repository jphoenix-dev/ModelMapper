"""P4 post-hoc artefact checks (NOT pre-registered): shuffled family labels, without the human family, human-family
neurons in non-people. (no model, no API)  Usage: python src/p4_checks.py  -> results/p4_checks.txt (tee)"""
import sys, json, random, math, statistics as stx
from collections import Counter
sys.path.insert(0, 'src')
from p4_family import tops, ROOT
fam = json.loads((ROOT/"results/wiki_families.json").read_text(encoding="utf-8"))
family = {c: v["p31_labels"][0] for c, v in fam.items() if c != "_unresolved" and v["p31_labels"]}
P = json.loads((ROOT/"results/p4_predictions.json").read_text(encoding="utf-8"))
F = {f: set(v["neurons"]) for f, v in P["families"].items() if v["neurons"]}
print("families:", {f: len(v) for f, v in F.items()})
new = tops(only_dirs=("concepts_fame5",))
print("coverage-4 concepts with tops:", len(new), "; with any family label:", sum(c in family for c in new))
def score(rows_cf):
    rows=[]
    for c,f in rows_cf:
        s=set(new[c]); own=len(s&F[f])/len(F[f]); other=stx.mean(len(s&F[g])/len(F[g]) for g in F if g!=f)
        rows.append((own,other))
    ratios=[(o+1e-9)/(x+1e-9) if x>0 else (float('inf') if o>0 else 1.0) for o,x in rows]
    wins=sum(o>x for o,x in rows); ties=sum(o==x for o,x in rows); k=len(rows)-ties
    p=sum(math.comb(k,i) for i in range(wins,k+1))/2**k if k else 1.0
    return stx.median(ratios), wins, k, p, rows
cf=[(c,family[c]) for c in new if family.get(c) in F]
med,w,k,p,rows=score(cf)
print(f"\nREAL: median ratio {med:.2f}, wins {w}/{k}, p {p:.2g}; own==0 in {sum(o==0 for o,_ in rows)}/{len(rows)}; own<other in {sum(o<x for o,x in rows)}")
# 1 shuffle null
rng=random.Random(7); labs=[f for _,f in cf]; meds=[]; ws=[]
for _ in range(2000):
    rng.shuffle(labs); m,ww,kk,pp,_=score([(c,l) for (c,_),l in zip(cf,labs)]); meds.append(m); ws.append(ww/kk if kk else 0)
meds_f=[m for m in meds if m!=float('inf')]
print(f"SHUFFLED labels (2000x): median ratio median {stx.median(meds):.2f}, 99th pct {sorted(meds)[int(.99*len(meds))]:.2f}, max {max(meds):.2f}; "
      f"win share median {stx.median(ws):.2f} (real {w/k:.2f}); shuffles reaching real ratio: {sum(m>=med for m in meds)}")
# 2 without human
cf2=[x for x in cf if x[1]!="human"]
med2,w2,k2,p2,_=score(cf2)
print(f"\nWITHOUT human ({len(cf2)} concepts): median ratio {med2:.2f}, wins {w2}/{k2}, p {p2:.2g}")
for f in sorted(set(x[1] for x in cf2)):
    sub=[x for x in cf2 if x[1]==f]; m,ww,kk,pp,rr=score(sub)
    print(f"  {f:<20} n={len(sub)} wins {ww}/{kk} own-mean {stx.mean(o for o,_ in rr):.3f} other-mean {stx.mean(x for _,x in rr):.3f}")
# 3 human neurons outside people
H=F["human"]; print("\nhuman family neurons:", sorted(H))
hum=[c for c in new if family.get(c)=="human"]; non=[c for c in new if c in family and family[c]!="human"]
r=lambda cs: stx.mean(len(set(new[c])&H)/len(H) for c in cs)
print(f"human-neuron hit rate: people {r(hum):.3f} (n={len(hum)}) vs non-people {r(non):.3f} (n={len(non)})")
nf=Counter(family[c] for c in non)
for f,n in nf.most_common(12):
    cs=[c for c in non if family[c]==f]; print(f"  {f:<28} n={n:<3} {r(cs):.3f}")

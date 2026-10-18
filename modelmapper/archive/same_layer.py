import glob, json, random, statistics as stx
from collections import Counter
from pathlib import Path
R=Path('C:/dev/already-clean')
tops={}; kset={}
for f in sorted(glob.glob(str(R/'results/concepts_*/rank_save_*.json'))):
    if 'rank_save_test' in f or f.endswith('_exact.json'): continue
    for c,v in json.loads(Path(f).read_text(encoding='utf-8')).items():
        if 'top' in v and c not in tops:
            tops[c]=[e['neuron'] for e in v['top']]
# k needed to move recall (k_mean) from wave_free / map_loop rows
for f in glob.glob(str(R/'results/concepts_*/wave_free.json'))+glob.glob(str(R/'results/concepts_*/map_loop_*.json')):
    if f.endswith('_meta.json'): continue
    try: d=json.loads(Path(f).read_text(encoding='utf-8'))
    except Exception: continue
    for c,r in d.items():
        if isinstance(r,dict) and r.get('k_mean') and c not in kset: kset[c]=r['k_mean']
lay=lambda u:int(u.split(':')[0][1:])
print(f"concepts with ranked neurons: {len(tops)}; with k_mean: {sum(c in kset for c in tops)}")
def stats(sets):
    shared=sum(len(set(map(lay,s)))<len(s) for s in sets)
    pairs=sum(sum(n*(n-1)//2 for n in Counter(map(lay,s)).values()) for s in sets)
    maxsame=Counter(max(Counter(map(lay,s)).values()) for s in sets)
    return shared, pairs, maxsame
rng=random.Random(5)
for K in (2,3,5,10):
    sets=[t[:K] for t in tops.values() if len(t)>=K]
    sh,pr,mx=stats(sets)
    pool=[lay(u) for s in sets for u in s]   # null: same layer distribution, layers drawn independently
    nulls=[]
    for _ in range(500):
        nulls.append(sum(len(set(rng.sample(pool,K)))<K for _ in sets))
    print(f"top-{K}: {sh}/{len(sets)} concepts ({sh/len(sets):.0%}) have >=2 neurons on the same layer; chance {stx.mean(nulls)/len(sets):.0%} "
          f"(same layer mix, random draw); max neurons on one layer: {dict(sorted(mx.items()))}")
# the k actually needed
sets=[tops[c][:kset[c]] for c in tops if c in kset and kset[c]>=2 and len(tops[c])>=kset[c]]
sh,pr,mx=stats(sets); 
print(f"\nthe k neurons each concept actually needs (k_mean, k>=2; {len(sets)} concepts, median k {stx.median(len(s) for s in sets)}): "
      f"{sh} ({sh/len(sets):.0%}) have >=2 on one layer; max per layer {dict(sorted(mx.items()))}")
L=Counter(lay(u) for t in tops.values() for u in t[:5])
print("\nlayers of top-5 neurons:", " ".join(f"L{l}:{n}" for l,n in sorted(L.items())))
# examples
ex=[(c,t[:5]) for c,t in tops.items() if max(Counter(map(lay,t[:5])).values())>=3][:6]
for c,t in ex: print(" ", c, t)

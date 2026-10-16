"""Validate the judge-free steering score against Jev on the auto-selected held-out replies (~1.2k Jev calls).
Jev asks the pipeline's two questions (lenient on-topic, fluent); compares per reply and per concept."""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import jev  # noqa: E402
from cache import Cache  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
cache = Cache(str(ROOT / "results/auto_steer_jev.cache.jsonl"))
d = json.loads((ROOT / "results/auto_steer_scale.json").read_text(encoding="utf-8"))


def judge(job):
    concept, r = job
    k = f"{concept}|{r}"
    if k in cache:
        return
    q = {"on": {"type": "noul", "instructions": f"Does this text mention, describe, or seem preoccupied with {concept}?"},
         "fluent": {"type": "noul", "instructions": "Is this text fluent, grammatical English that makes sense?"}}
    try:
        a = jev.ask(r or "(empty)", q)
        cache.put(k, [a["on"]["noul"], a["fluent"]["noul"]])
    except Exception:  # noqa: BLE001
        pass


jobs = [(t.split("/", 1)[1], r) for t, x in d.items() for r in x["all_heldout"]]
for w in (32, 8):
    with ThreadPoolExecutor(w) as ex:
        list(ex.map(judge, [j for j in jobs if f"{j[0]}|{j[1]}" not in cache]))
per = []
for t, x in d.items():
    c = t.split("/", 1)[1]
    js = [cache.get(f"{c}|{r}") for r in x["all_heldout"]]
    js = [v for v in js if v]
    jev_joint = sum(on >= .5 and fl >= .5 for on, fl in js) / max(1, len(js))
    per.append((x["heldout"], jev_joint, t))
a = [p[0] for p in per]; b = [p[1] for p in per]
n = len(per); ma, mb = sum(a) / n, sum(b) / n
r = sum((x - ma) * (y - mb) for x, y in per[:0] or zip(a, b)) / ((sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** .5)
print(f"{n} concepts | mean judge-free {ma:.3f} vs Jev lenient joint {mb:.3f} | per-concept Pearson r {r:.2f}")
print(f"concepts >= 0.25: judge-free {sum(x >= .25 for x in a)}, Jev {sum(y >= .25 for y in b)}")
for x, y, t in sorted(per, key=lambda p: -abs(p[0] - p[1]))[:6]:
    print(f"  biggest gap {t:<40} judge-free {x:.2f} Jev {y:.2f}")
print(f"Jev {jev.usage}, ~${jev.cost():.3f}")

# both sides judged by Jev, if the default replies have been regenerated (default_heldout_regen.py)
dp = ROOT / "results/default_heldout.json"
if dp.exists():
    dd = json.loads(dp.read_text(encoding="utf-8"))
    jobs = [(t.split("/", 1)[1], r) for t, rs in dd.items() for r in rs]
    for w in (32, 8):
        with ThreadPoolExecutor(w) as ex:
            list(ex.map(judge, [j for j in jobs if f"{j[0]}|{j[1]}" not in cache]))

    def jj(c, rs):
        js = [cache.get(f"{c}|{r}") for r in rs]; js = [v for v in js if v]
        return sum(on >= .5 and fl >= .5 for on, fl in js) / max(1, len(js))
    au = {t: jj(t.split("/", 1)[1], x["all_heldout"]) for t, x in d.items()}
    de = {t: jj(t.split("/", 1)[1], rs) for t, rs in dd.items()}
    both = sorted(set(au) & set(de))
    print(f"\nJev lenient joint, held-out, {len(both)} concepts: auto {sum(au[t] for t in both)/len(both):.3f} vs default "
          f"{sum(de[t] for t in both)/len(both):.3f} | auto better {sum(au[t] > de[t] for t in both)}, worse "
          f"{sum(au[t] < de[t] for t in both)}, tie {sum(au[t] == de[t] for t in both)}")
    print(f"Jev {jev.usage}, ~${jev.cost():.3f}")

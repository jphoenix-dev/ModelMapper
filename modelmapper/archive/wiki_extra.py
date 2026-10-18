"""Coverage expansion: a second, larger draw from the fame tiers where the model actually recalls (no Luna, no web
except one creation-date/category lookup per drawn title, same as wiki_sitelinks.py).

Same candidate pool, language-edition counts, filters and rejection log as wiki_sitelinks.py (its cache); excludes every
title already drawn into prompts/concepts_fame.json. Draws EXTRA[tier] new titles per tier (seed 8, a different walk
from the first draw's seed 7). Writes prompts/concepts_fame2.json (wave format) and updates results/wiki_sitelinks.json.
Usage: python src/wiki_extra.py
"""
import json
import random
from pathlib import Path

import httpx

from wiki_sample import ADULT, CREATED_BY, UA
from wiki_sitelinks import TIERS, get

ROOT = Path(__file__).resolve().parents[1]
import os
import sys
# usage: python src/wiki_extra.py [out_name=concepts_fame2] [seed=8] [t1 t2 t3 counts=60 60 40]
OUT_NAME = sys.argv[1] if len(sys.argv) > 1 else "concepts_fame2"
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 8
EXTRA = dict(zip(("t1_128+", "t2_64-127", "t3_32-63"), [int(x) for x in sys.argv[3:6]] if len(sys.argv) > 5 else (60, 60, 40)))


def main():
    st_p = ROOT / "results/wiki_sitelinks.json"
    st = json.loads(st_p.read_text(encoding="utf-8"))
    import glob as _g
    first = set()                                            # every title already drawn in any earlier fame/coverage draw
    for fp in _g.glob(str(ROOT / "prompts/concepts_fame*.json")):
        if Path(fp).stem != OUT_NAME:
            first |= {c for cs in json.loads(Path(fp).read_text(encoding="utf-8"))["concepts"].values() for c in cs}
    dis = set(st["disamb"])
    n_of = {t: st["links"].get(q) for t, q in st["qid"].items() if t not in dis and q}
    rng = random.Random(SEED); wave = {"concepts": {}}; st.setdefault("draw_log_extra", {}); st["draw_log_extra"].setdefault(OUT_NAME, {})
    with httpx.Client(headers=UA, timeout=30) as c:
        def info(t):
            if t not in st["meta"]:
                d = get(c, "https://en.wikipedia.org/w/api.php", action="query", prop="revisions|categories", titles=t,
                        rvlimit=1, rvdir="newer", rvprop="timestamp", cllimit="max", redirects=1, format="json")
                pg = next(iter(d["query"]["pages"].values()))
                st["meta"][t] = {"created": (pg.get("revisions") or [{}])[0].get("timestamp"),
                                 "cats": [x["title"] for x in pg.get("categories", [])]}
            return st["meta"][t]
        for name, lo, hi in TIERS:
            if name not in EXTRA:
                continue
            pool = sorted(t for t, v in n_of.items() if v is not None and lo <= v < hi and t not in first)
            rng.shuffle(pool); take = []; log = st["draw_log_extra"][OUT_NAME][name] = []
            for t in pool:
                if len(take) >= EXTRA[name]:
                    break
                m = info(t)
                if not m["created"]:
                    log.append((t, "no creation date")); continue
                if m["created"][:10] > CREATED_BY:
                    log.append((t, f"created {m['created'][:10]}")); continue
                if any(ADULT.search(x) for x in m["cats"]):
                    log.append((t, "adult-industry category")); continue
                take.append(t)
            wave["concepts"][name] = take
            print(f"{name:<10} pool {len(pool):>5} (excluding the first draw)  drew {len(take)}, rejected {len(log)}", flush=True)
    st_p.write_text(json.dumps(st, indent=1, ensure_ascii=False), encoding="utf-8")
    (ROOT / f"prompts/{OUT_NAME}.json").write_text(json.dumps(wave, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

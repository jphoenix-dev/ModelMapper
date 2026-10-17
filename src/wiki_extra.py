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
EXTRA = {"t1_128+": 60, "t2_64-127": 60, "t3_32-63": 40}


def main():
    st_p = ROOT / "results/wiki_sitelinks.json"
    st = json.loads(st_p.read_text(encoding="utf-8"))
    first = {c for cs in json.loads((ROOT / "prompts/concepts_fame.json").read_text(encoding="utf-8"))["concepts"].values() for c in cs}
    dis = set(st["disamb"])
    n_of = {t: st["links"].get(q) for t, q in st["qid"].items() if t not in dis and q}
    rng = random.Random(8); wave = {"concepts": {}}; st.setdefault("draw_log_extra", {})
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
            rng.shuffle(pool); take = []; log = st["draw_log_extra"][name] = []
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
    (ROOT / "prompts/concepts_fame2.json").write_text(json.dumps(wave, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

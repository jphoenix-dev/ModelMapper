"""Fame-stratified concept sample, v4: fame = number of Wikipedia language editions (Wikidata sitelinks). No Luna.

Why v4: per-article page-view lookups (v2: 2025 views; v3: 2023 views) are one request per title, and Wikimedia
throttled us (HTTP 429, Retry-After 33 s) - hours of waiting. Language-edition counts come 50 titles per request, are
a standard notability proxy, and are not tied to one year (less recency bias than 2025 page views).
Candidates: the same pool as wiki_sample.py (2025 and 2023 monthly top-1000 lists + 5,000 random articles), with the
same title filters (non-articles, lists, years/dates, 2024+ years in the title) plus disambiguation pages (pageprops).
Tiers (fixed before looking, doubling scale): >= 128 editions, 64-127, 32-63, 16-31, 8-15, 4-7, 1-3.
Draw: walk each tier in a fixed random order (seed 7) and accept a title only if its article existed by 2023-06-30
(first revision) and none of its categories marks it pornographic / adult-industry; every rejection is logged.
Requests one at a time with a pause; Retry-After honoured. Writes results/wiki_sitelinks.json and prompts/concepts_fame.json.
Usage: python src/wiki_sitelinks.py
"""
import json
import random
import re
import time
from pathlib import Path

import httpx

from modelmapper.core.wiki_sample import ADULT, BAD, CREATED_BY, RECENT_YEAR, UA
from modelmapper.core.settings import REPO_ROOT

ROOT = REPO_ROOT
TIERS = [("t1_128+", 128, 10 ** 6), ("t2_64-127", 64, 128), ("t3_32-63", 32, 64), ("t4_16-31", 16, 32),
         ("t5_8-15", 8, 16), ("t6_4-7", 4, 8), ("t7_1-3", 1, 4)]
TIER_N = 30
NOT_WP = {"commonswiki", "specieswiki", "metawiki", "wikidatawiki", "mediawikiwiki", "sourceswiki", "outreachwiki",
          "wikimaniawiki", "foundationwiki", "incubatorwiki", "wikifunctionswiki", "testwiki", "test2wiki", "testwikidatawiki"}


def get(c, url, **params):
    for attempt in range(12):
        try:
            r = c.get(url, params=params)
        except httpx.HTTPError:
            time.sleep(min(60, 2 ** attempt)); continue
        if r.status_code == 200:
            time.sleep(0.5)                                         # polite pacing
            return r.json()
        ra = r.headers.get("retry-after", "")
        time.sleep(float(ra) + 1 if ra.replace(".", "").isdigit() else min(60, 2 ** attempt))
    raise RuntimeError(f"{url}: still failing after 12 attempts ({r.status_code})")


def main():
    src = json.loads((ROOT / "results/wiki_sample.json").read_text(encoding="utf-8"))["sources"]
    titles = [t for t in src if not BAD.search(t) and not RECENT_YEAR.search(t)]
    out_p = ROOT / "results/wiki_sitelinks.json"
    st = json.loads(out_p.read_text(encoding="utf-8")) if out_p.exists() else {"qid": {}, "disamb": [], "links": {}, "meta": {}}
    with httpx.Client(headers=UA, timeout=30) as c:
        todo = [t for t in titles if t not in st["qid"]]
        for i in range(0, len(todo), 50):
            chunk = todo[i:i + 50]
            d = get(c, "https://en.wikipedia.org/w/api.php", action="query", prop="pageprops", titles="|".join(chunk),
                    redirects=1, format="json")
            q = d.get("query", {}); back = {}
            for x in q.get("normalized", []) + q.get("redirects", []):
                back[x["to"]] = back.get(x["from"], x["from"])
            for pg in q.get("pages", {}).values():
                orig = back.get(pg.get("title"), pg.get("title")); pp = pg.get("pageprops", {})
                if orig in chunk:
                    st["qid"][orig] = pp.get("wikibase_item")
                    if "disambiguation" in pp:
                        st["disamb"].append(orig)
            for t in chunk:
                st["qid"].setdefault(t, None)
            if (i // 50) % 20 == 0:
                print(f"  ids {len(st['qid'])}/{len(titles)}", flush=True)
                out_p.write_text(json.dumps(st, indent=1, ensure_ascii=False), encoding="utf-8")
        qids = sorted({q for q in st["qid"].values() if q and q not in st["links"]})
        for i in range(0, len(qids), 50):
            d = get(c, "https://www.wikidata.org/w/api.php", action="wbgetentities", ids="|".join(qids[i:i + 50]),
                    props="sitelinks", format="json")
            for q, e in d.get("entities", {}).items():
                st["links"][q] = sum(1 for k in e.get("sitelinks", {}) if k.endswith("wiki") and k not in NOT_WP)
            if (i // 50) % 20 == 0:
                print(f"  sitelinks {len(st['links'])}/{len(qids)}", flush=True)
                out_p.write_text(json.dumps(st, indent=1, ensure_ascii=False), encoding="utf-8")
        out_p.write_text(json.dumps(st, indent=1, ensure_ascii=False), encoding="utf-8")
        dis = set(st["disamb"])
        n_of = {t: st["links"].get(st["qid"].get(t)) for t in titles if t not in dis}
        print(f"{len(titles)} titles; {len(dis)} disambiguation; {sum(v is None for v in n_of.values())} without a Wikidata count (excluded)", flush=True)

        def info(t):
            if t not in st["meta"]:
                d = get(c, "https://en.wikipedia.org/w/api.php", action="query", prop="revisions|categories", titles=t,
                        rvlimit=1, rvdir="newer", rvprop="timestamp", cllimit="max", redirects=1, format="json")
                pg = next(iter(d["query"]["pages"].values()))
                st["meta"][t] = {"created": (pg.get("revisions") or [{}])[0].get("timestamp"),
                                 "cats": [x["title"] for x in pg.get("categories", [])]}
            return st["meta"][t]
        rng = random.Random(7); wave = {"concepts": {}}; st["draw_log"] = {}
        for name, lo, hi in TIERS:
            pool = sorted(t for t, v in n_of.items() if v is not None and lo <= v < hi)
            rng.shuffle(pool); take = []; log = st["draw_log"][name] = []
            for t in pool:
                if len(take) >= TIER_N:
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
            print(f"{name:<10} pool {len(pool):>5}  drew {len(take)} (rejected {len(log)}: {sum('created' in r for _, r in log)} too recent, "
                  f"{sum(r.startswith('adult') for _, r in log)} adult): {take}", flush=True)
        out_p.write_text(json.dumps(st, indent=1, ensure_ascii=False), encoding="utf-8")
    (ROOT / "prompts/concepts_fame.json").write_text(json.dumps(wave, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

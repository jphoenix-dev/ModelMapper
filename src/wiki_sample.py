"""Fame-stratified random concept sample from English Wikipedia page views (agreed with James 16 Oct 2026; no Luna).

Candidates: Wikimedia's monthly top-1000 most-viewed articles for every month of 2025 (the famous end) plus random
articles from the API's list=random (the obscure end). Fame = total 2025 user page views (per-article monthly API).
Filtered out mechanically before sampling: non-article namespaces, Main Page, lists ("List of"), "Deaths in",
"(disambiguation)", years / dates / pure numbers. Tiers by annual views; TIER_N concepts drawn per tier (seed 7).
Read-only public APIs; user agent names the project's public page as contact (Wikimedia policy), no personal details;
2 requests at a time, honouring Retry-After. Writes prompts/concepts_fame.json (wave format,
categories = tiers) and results/wiki_sample.json (every candidate with its views, for audit).
Usage: python src/wiki_sample.py [n_random_batches=10]   (500 random articles per batch)
"""
import json
import random
import re
import sys
import time
from pathlib import Path
from urllib.parse import quote

import httpx
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
UA = {"User-Agent": "already-clean-research/0.1 (https://eiffel-tower-qwen.vercel.app/; interpretability research, low volume) httpx"}
TIERS = [("t1_5M+", 5e6, 1e12), ("t2_1M-5M", 1e6, 5e6), ("t3_200k-1M", 2e5, 1e6), ("t4_50k-200k", 5e4, 2e5),
         ("t5_10k-50k", 1e4, 5e4), ("t6_2k-10k", 2e3, 1e4), ("t7_under_2k", 1, 2e3)]
TIER_N = 30
FAME_YEAR = 2023               # fame = page views in a year inside the model's training data (v3; v2 used 2025 views)
CREATED_BY = "2023-06-30"     # article must exist by then
RECENT_YEAR = re.compile(r"\b20(2[4-9]|[3-9]\d)\b")   # titles naming 2024 or later (events, seasons, films) are excluded
ADULT = re.compile(r"pornograph|adult (film|video|model|website|industry|entertainment)|erotic", re.I)
BAD = re.compile(r"^(Main Page|Special:|Wikipedia:|Portal:|File:|Category:|Template:|Help:|Talk:|User:|List of|Lists of|"
                 r"Deaths in|\d{1,4}$|\d{4} in |(January|February|March|April|May|June|July|August|September|October|"
                 r"November|December) \d)|\(disambiguation\)|^-$")


def get(c, url, **params):
    """GET JSON politely: on 429/5xx wait the server's Retry-After (or back off, up to 60 s), 10 attempts.
    Returns None for 404 and for a request that still fails (the caller counts those; never silent)."""
    for attempt in range(10):
        try:
            r = c.get(url, params=params or None)
        except httpx.HTTPError:
            time.sleep(min(60, 2 ** attempt)); continue
        if r.status_code == 200:
            return r.json()
        if r.status_code == 404:
            return None
        ra = r.headers.get("retry-after", "")
        time.sleep(float(ra) if ra.replace(".", "").isdigit() else min(60, 2 ** attempt))
    return FAILED


FAILED = {"failed": True}


def main():
    n_rand = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    out_p = ROOT / "results/wiki_sample.json"
    cache = json.loads(out_p.read_text(encoding="utf-8")) if out_p.exists() else {"views": {}, "sources": {}}
    with httpx.Client(headers=UA, timeout=30) as c:
        if not cache["sources"]:
            for y, m in [(y, m) for y in (2025,) for m in range(1, 13)]:
                d = get(c, f"https://wikimedia.org/api/rest_v1/metrics/pageviews/top/en.wikipedia/all-access/{y}/{m:02d}/all-days")
                for a in d["items"][0]["articles"]:
                    cache["sources"].setdefault(a["article"].replace("_", " "), "top")
            for _ in range(n_rand):
                d = get(c, "https://en.wikipedia.org/w/api.php", action="query", list="random", rnnamespace=0, rnlimit=500, format="json")
                for a in d["query"]["random"]:
                    cache["sources"].setdefault(a["title"], "random")
            out_p.write_text(json.dumps(cache, indent=1, ensure_ascii=False), encoding="utf-8")
        if not cache.get("sources_2023_added"):
            for m in range(1, 13):
                for _try in range(5):
                    d = get(c, f"https://wikimedia.org/api/rest_v1/metrics/pageviews/top/en.wikipedia/all-access/{FAME_YEAR}/{m:02d}/all-days")
                    if d and "items" in d:
                        break
                    time.sleep(10)
                else:
                    print(f"top list {FAME_YEAR}-{m:02d} unavailable after 5 tries - skipped", flush=True); continue
                for a in d["items"][0]["articles"]:
                    cache["sources"].setdefault(a["article"].replace("_", " "), f"top{FAME_YEAR}")
            cache["sources_2023_added"] = True
            out_p.write_text(json.dumps(cache, indent=1, ensure_ascii=False), encoding="utf-8")
        titles = [t for t in cache["sources"] if not BAD.search(t) and not RECENT_YEAR.search(t)]
        print(f"{len(cache['sources'])} candidates, {len(titles)} after filtering", flush=True)
        vk = f"views{FAME_YEAR}"; cache.setdefault(vk, {})
        todo = [t for t in titles if t not in cache[vk]]

        def views(t):
            d = get(c, "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/"
                    f"{quote(t.replace(' ', '_'), safe='')}/monthly/{FAME_YEAR}0101/{FAME_YEAR}1231")
            if d is FAILED:
                return t, None                                              # unknown: excluded from tiers, counted
            return t, (sum(x["views"] for x in d["items"]) if d else 0)
        with ThreadPoolExecutor(2) as ex:                                   # 2 at a time (4 drew HTTP 429)
            for i, (t, v) in enumerate(ex.map(views, todo)):
                cache[vk][t] = v
                if i % 500 == 0:
                    print(f"  {FAME_YEAR} views fetched {len(cache[vk])}/{len(titles)}", flush=True)
                    out_p.write_text(json.dumps(cache, indent=1, ensure_ascii=False), encoding="utf-8")
        out_p.write_text(json.dumps(cache, indent=1, ensure_ascii=False), encoding="utf-8")
    V = cache[f"views{FAME_YEAR}"]
    unknown = [t for t in titles if V.get(t) is None]
    print(f"views unknown after retries (excluded from every tier): {len(unknown)}", flush=True)
    # Draw (v2, after the first draw showed the problems): walk each tier's pool in a fixed random order and accept a
    # title only if (a) its article existed by CREATED_BY (first revision) - so the model could have seen it; 2025 page
    # views otherwise favour things that only became notable after Qwen3's training data (recency, not fame) - and
    # (b) none of its Wikipedia categories marks it as pornographic / adult-industry. Every rejection is logged.
    rng = random.Random(7); wave = {"concepts": {}}; meta = cache.setdefault("meta", {}); draw_log = cache["draw_log"] = {}
    with httpx.Client(headers=UA, timeout=30) as c:
        def info(t):
            if t not in meta:
                d = get(c, "https://en.wikipedia.org/w/api.php", action="query", prop="revisions|categories", titles=t,
                        rvlimit=1, rvdir="newer", rvprop="timestamp", cllimit="max", redirects=1, format="json")
                if d is None or d is FAILED:
                    meta[t] = {"created": None, "cats": []}
                else:
                    pg = next(iter(d["query"]["pages"].values()))
                    meta[t] = {"created": (pg.get("revisions") or [{}])[0].get("timestamp"),
                               "cats": [x["title"] for x in pg.get("categories", [])]}
            return meta[t]
        for name, lo, hi in TIERS:
            pool = sorted(t for t in titles if V.get(t) is not None and lo <= V[t] < hi)
            rng.shuffle(pool); take = []; log = draw_log[name] = []
            for t in pool:
                if len(take) >= TIER_N:
                    break
                m = info(t)
                if m["created"] is None:
                    log.append((t, "no creation date")); continue
                if m["created"][:10] > CREATED_BY:
                    log.append((t, f"created {m['created'][:10]}")); continue
                if any(ADULT.search(x) for x in m["cats"]):
                    log.append((t, "adult-industry category")); continue
                take.append(t)
            wave["concepts"][name] = take
            print(f"{name:<12} pool {len(pool):>6}  drew {len(take)} (rejected {len(log)}: "
                  f"{sum('created' in r for _, r in log)} too recent, {sum(r.startswith('adult') for _, r in log)} adult): {take}", flush=True)
    out_p.write_text(json.dumps(cache, indent=1, ensure_ascii=False), encoding="utf-8")
    (ROOT / "prompts/concepts_fame.json").write_text(json.dumps(wave, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

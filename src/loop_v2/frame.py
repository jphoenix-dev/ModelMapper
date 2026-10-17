"""Loop v2 sampling frame: which concepts go in. Stratified by FAME (language editions) x TYPE (Wikidata P31).
(web, read-only; no model)

James (17 Oct): broaden away from famous names / people. Pool = the concept-atlas names (English Wikipedia titles
known in >= 16 language editions, filtered as concept_atlas.fetch), minus every concept already mapped.
  types    fetch each name's Wikidata 'instance of' (P31) values and their English labels -> types.json
  draw     map P31 labels to broad buckets (rules below, fixed after looking at the label frequencies only, before any
           draw), then draw up to N_PER_CELL names per (bucket x fame tier) cell at random (seed 71), filling to the
           target from the remaining names round-robin over buckets; saves frame.json with the full draw log.
Usage: python src/loop_v2/frame.py types | draw [target=1500]
"""
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("AC_DATA_DIR", "D:/already-clean-data"))
OUT = DATA / "loop_v2" / "frame"
TIERS = [("128+", 128, 10 ** 9), ("64-127", 64, 128), ("32-63", 32, 64), ("16-31", 16, 32)]


def types():
    from wiki_sample import UA
    from wiki_sitelinks import get
    OUT.mkdir(parents=True, exist_ok=True)
    st = json.loads((ROOT / "results/wiki_sitelinks.json").read_text(encoding="utf-8"))
    intros = json.loads((DATA / "concept_atlas/intros.json").read_text(encoding="utf-8"))
    pool = {t: st["qid"][t] for t, v in intros.items() if t in st["qid"] and not v.get("excluded") and v["extract"]}
    p = OUT / "types.json"
    T = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"p31": {}, "labels": {}}
    todo = sorted(q for q in set(pool.values()) if q not in T["p31"])
    print(f"pool {len(pool)} names; {len(todo)} qids to fetch", flush=True)
    with httpx.Client(headers=UA, timeout=60) as cl:
        for i in range(0, len(todo), 50):
            d = get(cl, "https://www.wikidata.org/w/api.php", action="wbgetentities", ids="|".join(todo[i:i + 50]), props="claims", format="json")
            for q, e in d.get("entities", {}).items():
                T["p31"][q] = [c["mainsnak"].get("datavalue", {}).get("value", {}).get("id") for c in e.get("claims", {}).get("P31", [])]
            if (i // 50) % 20 == 0:
                print(f"{i + 50}/{len(todo)}", flush=True); p.write_text(json.dumps(T), encoding="utf-8")
        vals = sorted({v for vs in T["p31"].values() for v in vs if v} - set(T["labels"]))
        for i in range(0, len(vals), 50):
            d = get(cl, "https://www.wikidata.org/w/api.php", action="wbgetentities", ids="|".join(vals[i:i + 50]), props="labels", languages="en", format="json")
            for q, e in d.get("entities", {}).items():
                T["labels"][q] = e.get("labels", {}).get("en", {}).get("value", q)
    T["pool"] = pool; T["links"] = {t: st["links"].get(q) for t, q in pool.items()}
    p.write_text(json.dumps(T, ensure_ascii=False), encoding="utf-8")
    c = Counter(T["labels"].get(v, v) for t, q in pool.items() for v in (T["p31"].get(q) or [None])[:1])
    print("most common first P31 labels:"); print("\n".join(f"  {n:>5} {k}" for k, n in c.most_common(60)))


VA = "Wikipedia:Vital articles/Level/4"
SECTIONS = ["People", "History", "Geography", "Arts", "Philosophy and religion", "Everyday life", "Society and social sciences",
            "Biology and health sciences", "Physical sciences", "Technology", "Mathematics"]
SKIP = ("Draft", "draft", "Removed", "Article alerts", "Candidates")


def vital():
    """James (17 Oct): build the frame from Vital Articles Level 4 (~10,000 articles curated to cover knowledge evenly).
    1 article links (namespace 0) from every current Level-4 list page (redirects resolved; drafts, removed,
      candidates and alert pages skipped), section = the top-level section of the page it was listed on
    2 Wikidata id, number of language editions (fame), P31 and P279
    3 named vs common, fixed here before any drawing: the name (title without a parenthetical) is looked for in the
      article's plain-text intro AFTER its first word; 'common' if it appears there in lowercase (a bicycle, oxygen,
      democracy), 'named' if only capitalised; if it never appears mid-sentence, 'common' if Wikidata has 'subclass of'
      (P279), else 'named'. Both signals are saved.
    -> vital.json"""
    from wiki_sample import ADULT, UA
    from wiki_sitelinks import get
    import re
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "vital.json"
    V = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    with httpx.Client(headers=UA, timeout=60) as cl:
        if "articles" not in V:
            d = get(cl, "https://en.wikipedia.org/w/api.php", action="query", list="allpages", apprefix="Vital articles/Level 4/",   # current lists ("Level/4/..." are old redirects)
                    apnamespace=4, aplimit=500, format="json")
            pages = [x["title"] for x in d["query"]["allpages"] if not any(s in x["title"] for s in SKIP)]
            arts, seen_final, log = {}, set(), []
            for pg in pages:
                d = get(cl, "https://en.wikipedia.org/w/api.php", action="parse", page=pg, prop="links", redirects=1, format="json")
                if "parse" not in d:
                    log.append((pg, "no parse")); continue
                final = d["parse"]["title"]
                if final in seen_final:
                    log.append((pg, f"duplicate of {final}")); continue
                seen_final.add(final)
                rest = final.replace("Level/4/", "Level 4/").split("Level 4/", 1)[-1]
                sec = next((s for s in SECTIONS if rest.startswith(s)), None)
                if sec is None:
                    log.append((pg, f"not a current section: {final}")); continue
                links = [x["*"] for x in d["parse"]["links"] if x["ns"] == 0 and "exists" in x]
                log.append((pg, f"{final}: {len(links)} links -> {sec}"))
                for t in links:
                    arts.setdefault(t, sec)
            V = {"articles": arts, "page_log": log}
            p.write_text(json.dumps(V, ensure_ascii=False), encoding="utf-8")
            print("\n".join(f"  {a} | {b}" for a, b in log), flush=True)
        arts = V["articles"]; titles = sorted(arts)
        print(f"{len(titles)} articles; by section {dict(Counter(arts.values()))}", flush=True)
        V.setdefault("qid", {}); V.setdefault("intro", {})
        todo = [t for t in titles if t not in V["qid"]]
        for i in range(0, len(todo), 50):
            d = get(cl, "https://en.wikipedia.org/w/api.php", action="query", prop="pageprops", ppprop="wikibase_item",
                    titles="|".join(todo[i:i + 50]), redirects=1, format="json")
            q = d.get("query", {}); back = {}
            for k in ("normalized", "redirects"):
                for x in q.get(k, []):
                    back[x["to"]] = back.get(x["from"], x["from"])
            for pg in q.get("pages", {}).values():
                src = back.get(pg.get("title"), pg.get("title"))
                V["qid"][src] = pg.get("pageprops", {}).get("wikibase_item")
            for t in todo[i:i + 50]:
                V["qid"].setdefault(t, None)
        print(f"qids: {sum(v is not None for v in V['qid'].values())}/{len(titles)}", flush=True)
        p.write_text(json.dumps(V, ensure_ascii=False), encoding="utf-8")
        V.setdefault("wd", {})
        qids = sorted({q for q in V["qid"].values() if q and q not in V["wd"]})
        for i in range(0, len(qids), 50):
            d = get(cl, "https://www.wikidata.org/w/api.php", action="wbgetentities", ids="|".join(qids[i:i + 50]), props="claims|sitelinks", format="json")
            for q, e in d.get("entities", {}).items():
                cl_ = e.get("claims", {})
                ids = lambda prop: [c["mainsnak"].get("datavalue", {}).get("value", {}).get("id") for c in cl_.get(prop, [])]
                V["wd"][q] = {"links": len(e.get("sitelinks", {})), "p31": ids("P31"), "p279": ids("P279")}
            if (i // 50) % 20 == 0:
                print(f"wikidata {i + 50}/{len(qids)}", flush=True); p.write_text(json.dumps(V, ensure_ascii=False), encoding="utf-8")
    p.write_text(json.dumps(V, ensure_ascii=False), encoding="utf-8")
    print(f"wikidata done: {len(V['wd'])} items", flush=True)


def fetch_intros(cl, V, titles, p):
    """Plain-text intros, 20 per request (only for the titles that need them: ~100 per minute)."""
    from wiki_sitelinks import get
    todo = [t for t in titles if t not in V["intro"]]
    for i in range(0, len(todo), 20):
        d = get(cl, "https://en.wikipedia.org/w/api.php", action="query", prop="extracts", exintro=1, explaintext=1, exlimit=20,
                titles="|".join(todo[i:i + 20]), redirects=1, format="json")
        q = d.get("query", {}); back = {}
        for k in ("normalized", "redirects"):
            for x in q.get(k, []):
                back[x["to"]] = back.get(x["from"], x["from"])
        for pg in q.get("pages", {}).values():
            V["intro"][back.get(pg.get("title"), pg.get("title"))] = (pg.get("extract") or "")[:3000]
        for t in todo[i:i + 20]:
            V["intro"].setdefault(t, "")
        if (i // 20) % 50 == 0:
            print(f"intros {i + 20}/{len(todo)}", flush=True); p.write_text(json.dumps(V, ensure_ascii=False), encoding="utf-8")
    p.write_text(json.dumps(V, ensure_ascii=False), encoding="utf-8")


def tag_titles(V, titles):
    """named vs common (rule fixed before any drawing): the name (title without a parenthetical) is looked for in the
    intro AFTER its first word; 'common' if it appears there in lowercase, 'named' if only capitalised; if it never
    appears mid-sentence, 'common' if Wikidata has 'subclass of' (P279), else 'named'. Also flags adult intros."""
    import re
    from wiki_sample import ADULT
    out = {}
    for t in titles:
        name = re.sub(r"\s*\(.*?\)\s*$", "", t).strip(); intro = V["intro"].get(t, "")
        body = intro.split(" ", 1)[1] if " " in intro else ""
        lower_hit = re.search(r"" + re.escape(name[:1].lower() + name[1:]) + r"", body) is not None
        upper_hit = re.search(r"" + re.escape(name[:1].upper() + name[1:]) + r"", body) is not None
        wd = V["wd"].get(V["qid"].get(t) or "", {})
        if lower_hit and name[:1].isalpha():
            tag, why = "common", "lowercase in intro"
        elif upper_hit:
            tag, why = "named", "capitalised in intro"
        else:
            tag, why = ("common", "P279 (no mid-sentence mention)") if wd.get("p279") else ("named", "no P279 (no mid-sentence mention)")
        out[t] = {"tag": tag, "why": why, "adult": bool(ADULT.search(intro[:600]))}
    return out


VTIERS = [("128+", 128, 10 ** 9), ("64-127", 64, 128), ("32-63", 32, 64), ("<32", 0, 32)]


def draw():
    """Stratified draw from vital.json: equal share per section; within a section, equal share per fame tier; a cell
    short of articles passes its remainder to the section's other tiers, then to other sections (round-robin). Seed
    71. Excluded (logged): no Wikidata id, adult intro, BAD titles, already mapped concepts. -> frame.json"""
    from concept_search_feasibility import maps
    from wiki_sample import BAD
    target = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
    V = json.loads((OUT / "vital.json").read_text(encoding="utf-8"))
    mapped = set(maps()); excl = []
    cells = defaultdict(list)
    for t, sec in sorted(V["articles"].items()):
        q = V["qid"].get(t); wd = V["wd"].get(q or "", {})
        why = "no wikidata id" if not q or not wd else "bad title" if BAD.search(t) else "already mapped" if t in mapped else None
        if why:
            excl.append((t, why)); continue
        n = wd["links"]; tier = next(nm for nm, lo, hi in VTIERS if lo <= n < hi)
        cells[(sec, tier)].append({"concept": t, "section": sec, "tier": tier, "links": n, "qid": q})
    rng = random.Random(71)
    for k in sorted(cells):
        rng.shuffle(cells[k])
    take = defaultdict(int); per_sec = target // len(SECTIONS)
    quota = {s: per_sec + (1 if i < target - per_sec * len(SECTIONS) else 0) for i, s in enumerate(SECTIONS)}
    left = 0
    for s in SECTIONS:                                   # within section: equal per tier, remainder to other tiers
        need = quota[s]; tiers = [nm for nm, _, _ in VTIERS]
        while need and any(take[(s, t)] < len(cells[(s, t)]) for t in tiers):
            for t in tiers:
                if need and take[(s, t)] < len(cells[(s, t)]):
                    take[(s, t)] += 1; need -= 1
        left += need
    while left and any(take[k] < len(v) for k, v in cells.items()):     # sections short of articles: round-robin
        for k in sorted(cells):
            if left and take[k] < len(cells[k]):
                take[k] += 1; left -= 1
    drawn = [x for k in sorted(cells) for x in cells[k][:take[k]]]
    rng.shuffle(drawn)
    # named/common tags and the adult-intro check need intros (~100 per minute): done separately by 'tag', so the draw
    # (and preparing / mapping) doesn't wait on them; analyses read the tags from frame.json once 'tag' has run
    out = {"target": target, "seed": 71, "rules": draw.__doc__, "draw": drawn, "excluded": excl,
           "cell_sizes": {f"{a}|{b}": len(v) for (a, b), v in cells.items()}, "cell_taken": {f"{a}|{b}": take[(a, b)] for (a, b) in cells}}
    (OUT / "frame.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    c = Counter(x["section"] for x in drawn); ct = Counter(x["tier"] for x in drawn)
    print(f"drew {len(drawn)} (target {target}); excluded {dict(Counter(w for _, w in excl))}; tiers {dict(ct)}; sections {dict(c)}")


def tag():
    """Fetch intros for the drawn concepts and add named/common tags + the adult-intro flag to frame.json (run after
    'draw', in parallel with prepare). Flagged-adult concepts stay mapped but are listed for exclusion from analyses."""
    import httpx
    from wiki_sample import UA
    F = json.loads((OUT / "frame.json").read_text(encoding="utf-8")); V = json.loads((OUT / "vital.json").read_text(encoding="utf-8"))
    titles = [x["concept"] for x in F["draw"]]
    with httpx.Client(headers=UA, timeout=60) as cl:
        fetch_intros(cl, V, titles, OUT / "vital.json")
    tags = tag_titles(V, titles)
    for x in F["draw"]:
        x["kind"], x["kind_why"], x["adult_intro"] = tags[x["concept"]]["tag"], tags[x["concept"]]["why"], tags[x["concept"]]["adult"]
    (OUT / "frame.json").write_text(json.dumps(F, indent=1, ensure_ascii=False), encoding="utf-8")
    c = Counter((x["section"], x["kind"]) for x in F["draw"])
    print(f"tagged {len(titles)}; adult-flagged {sum(x['adult_intro'] for x in F['draw'])}")
    for sct in SECTIONS:
        print(f"  {sct:<30} named {c[(sct, 'named')]:>4}  common {c[(sct, 'common')]:>4}")
    rng = random.Random(1)
    for k in ("named", "common"):
        ex = [x["concept"] for x in F["draw"] if x["kind"] == k]
        print(f"examples {k}: {rng.sample(ex, min(25, len(ex)))}")


if __name__ == "__main__":
    {"types": types, "vital": vital, "draw": draw, "tag": tag}[sys.argv[1]]()

"""Luna-free concept families from Wikidata 'instance of' (P31), for PREREG_MAP P4. (web: Wikipedia/Wikidata APIs, read-only)

Concepts: every mapped concept (rank_save_*.json of all sets) plus the coverage-4 list (prompts/concepts_fame5.json).
Wikidata item: from wiki_sitelinks.json for Wikipedia-drawn concepts; for the hand-picked ones, the English Wikipedia
title is tried as given, then without a leading 'the/a/an', then capitalised (redirects followed); unresolved concepts
are listed and left out. P31 values and their English labels are fetched 50 items per request.
Output: results/wiki_families.json {concept: {"qid", "p31": [qid...], "p31_labels": [...]}, "_unresolved": [...]}
Usage: python src/wiki_families.py
"""
import glob
import json
import time
from pathlib import Path

import httpx

from wiki_sample import UA
from wiki_sitelinks import get

ROOT = Path(__file__).resolve().parents[1]


def main():
    concepts = set()
    for f in glob.glob(str(ROOT / "results/concepts_*/rank_save_*.json")):
        if "rank_save_test" in f or f.endswith("_exact.json"):
            continue
        concepts |= {c for c, v in json.loads(Path(f).read_text(encoding="utf-8")).items() if "top" in v}
    concepts |= {c for cs in json.loads((ROOT / "prompts/concepts_fame5.json").read_text(encoding="utf-8"))["concepts"].values() for c in cs}
    sl = json.loads((ROOT / "results/wiki_sitelinks.json").read_text(encoding="utf-8"))["qid"]
    out_p = ROOT / "results/wiki_families.json"
    out = json.loads(out_p.read_text(encoding="utf-8")) if out_p.exists() else {}
    unresolved = []
    with httpx.Client(headers=UA, timeout=30) as c:
        need = [x for x in sorted(concepts) if x not in out]
        q_of = {x: sl[x] for x in need if sl.get(x)}
        rest = [x for x in need if x not in q_of]
        for x in rest:                                            # hand-picked names: try a few title forms
            forms = [x]
            for art in ("the ", "a ", "an "):
                if x.lower().startswith(art):
                    forms.append(x[len(art):])
            forms += [f[:1].upper() + f[1:] for f in list(forms)]
            for t in dict.fromkeys(forms):
                d = get(c, "https://en.wikipedia.org/w/api.php", action="query", prop="pageprops", titles=t, redirects=1, format="json")
                pg = next(iter(d["query"]["pages"].values())); pp = pg.get("pageprops", {})
                if pp.get("wikibase_item") and "disambiguation" not in pp:
                    q_of[x] = pp["wikibase_item"]; break
            else:
                unresolved.append(x)
        items = sorted(set(q_of.values())); p31 = {}
        for i in range(0, len(items), 50):
            d = get(c, "https://www.wikidata.org/w/api.php", action="wbgetentities", ids="|".join(items[i:i + 50]), props="claims", format="json")
            for q, e in d.get("entities", {}).items():
                p31[q] = [cl["mainsnak"]["datavalue"]["value"]["id"] for cl in e.get("claims", {}).get("P31", [])
                          if cl.get("mainsnak", {}).get("datavalue")]
        labs = sorted({v for vs in p31.values() for v in vs}); lab = {}
        for i in range(0, len(labs), 50):
            d = get(c, "https://www.wikidata.org/w/api.php", action="wbgetentities", ids="|".join(labs[i:i + 50]), props="labels", languages="en", format="json")
            for q, e in d.get("entities", {}).items():
                lab[q] = e.get("labels", {}).get("en", {}).get("value", q)
    for x, q in q_of.items():
        out[x] = {"qid": q, "p31": p31.get(q, []), "p31_labels": [lab.get(v, v) for v in p31.get(q, [])]}
    out["_unresolved"] = sorted(set(out.get("_unresolved", [])) | set(unresolved))
    out_p.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    from collections import Counter
    cnt = Counter(l for k, v in out.items() if k != "_unresolved" for l in v["p31_labels"][:1])
    print(f"{len(out) - 1} concepts labelled; unresolved {len(out['_unresolved'])}: {out['_unresolved'][:20]}")
    print("most common first P31:", cnt.most_common(15))


if __name__ == "__main__":
    main()

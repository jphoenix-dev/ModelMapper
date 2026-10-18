"""Question-side test + specificity gradient (queued after run 1; design in RESEARCH_STATE.md). (web; Luna; GPU)

Are a concept's key neurons 'say it' neurons or meaning neurons, and how specific are they?
Sources: 15 mapped landmarks (top-5 neurons and k_mean from their maps).
Neighbours per source, from Wikidata (not by hand), 3 per level:
  L1 same place      notable items within 3 km (coordinates; >= 30 language editions; no administrative areas / people)
  L2 same kind       same first 'instance of' (P31) as the source, > 500 km away, >= 30 editions
  L3 same section    Vital Articles: same section as the source, different P31 (seeded draw)
  L4 unrelated       Vital Articles: a different section (seeded draw)
Luna writes: 8 FACT questions per source (the name IN the question, a short factual answer), and 8 NAMING questions
(answer = the name) for each neighbour and for sources whose map saved no prompts. Qwen keeps the questions it answers
(P >= 0.2, leak filter on naming questions).
Ablations (each zeroed at every position, every row in the same pass as its as-is copy):
  src_k     the source's top-k_mean neurons         src5   its top-5
  src5_late its top-5 in layers >= 20                src5_early its top-5 in layers < 20
  rand5     5 random neurons from the same layers (2 draws)
  other5    another source's top-5 (specificity control)
measured on: the source's naming questions, its fact questions, and every neighbour's naming questions.
Predictions (stated before running, RESEARCH_STATE.md): late 'say' neurons leave facts intact while naming drops; early
neurons may not; damage falls with distance L1 -> L4; attribute neurons partly damage neighbours sharing the attribute.
Output: D:/already-clean-data/specificity/ (neighbours.json, questions.json, results.jsonl, summary.txt).
Usage: python src/loop_v2/specificity.py neighbours | questions | run | summary
"""
import glob
import json
import os
import random
import re
import statistics as stx
import sys
import time
from pathlib import Path
from modelmapper.core.settings import DATA_DIR, REPO_ROOT

ROOT = REPO_ROOT
OUT = DATA_DIR / "specificity"
SOURCES = ["Eiffel Tower", "Statue of Liberty", "Louvre", "Notre-Dame de Paris", "Taj Mahal", "Colosseum", "Sydney Opera House",
           "Burj Khalifa", "Forbidden City", "Sagrada Família", "Hagia Sophia", "St. Peter's Basilica", "Machu Picchu", "Parthenon",
           "Golden Gate Bridge"]
ADMIN = {"Q515", "Q484170", "Q702842", "Q5119", "Q6256", "Q1549591", "Q200250", "Q1637706", "Q5", "Q3624078", "Q90", "Q1093829", "Q747074",
         "Q15284", "Q123705", "Q174844", "Q3957", "Q532", "Q2264924", "Q22865"}
from modelmapper.core.wiki_sample import UA  # noqa: E402  (the project's descriptive User-Agent; a shorter one got HTTP 403)


def maps():
    out = {}
    for f in glob.glob(str(ROOT / "results/concepts_*/rank_save_*.json")):
        if "test" in f or f.endswith("_exact.json"):
            continue
        for c, v in json.loads(Path(f).read_text(encoding="utf-8")).items():
            if "top" in v:
                out.setdefault(c, {"top": [(e["neuron"], e["act_answer_pos"]) for e in v["top"][:5]], "k_mean": v.get("k_mean"),
                                   "prompts": v.get("prompts"), "answer_token": v.get("answer_token")})
    for b in glob.glob((DATA_DIR.as_posix() + "/loop_v2/run1/results/**/map/*/records.jsonl"), recursive=True):
        for line in Path(b).read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["row"]["status"] == "ok":
                out[r["concept"]] = {"top": [(e["neuron"], e["act_answer_pos"]) for e in r["rank"]["top"][:5]], "k_mean": r["row"]["k_mean"],
                                     "prompts": r["rank"]["prompts"], "answer_ids": r["rank"]["answer_ids"], "form": r["row"]["answer_form"]}
    return out


def sparql(cl, q):
    for attempt in range(6):
        r = cl.get("https://query.wikidata.org/sparql", params={"query": q, "format": "json"})
        if r.status_code == 200:
            time.sleep(1.0); return r.json()["results"]["bindings"]
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"WDQS failed: {r.status_code}")


def neighbours():
    import httpx
    from modelmapper.core.wiki_sitelinks import get
    OUT.mkdir(parents=True, exist_ok=True)
    V = json.loads((DATA_DIR / "loop_v2/frame/vital.json").read_text(encoding="utf-8"))
    rng = random.Random(151); res = {}
    with httpx.Client(headers=UA, timeout=90) as cl:
        d = get(cl, "https://en.wikipedia.org/w/api.php", action="query", prop="pageprops", ppprop="wikibase_item", titles="|".join(SOURCES), redirects=1, format="json")
        qid = {pg["title"]: pg.get("pageprops", {}).get("wikibase_item") for pg in d["query"]["pages"].values()}
        def ents(ids):
            """P31, coordinates and English Wikipedia title for Wikidata ids (regular API, 50 per call)."""
            out = {}
            for i in range(0, len(ids), 50):
                d = get(cl, "https://www.wikidata.org/w/api.php", action="wbgetentities", ids="|".join(ids[i:i + 50]), props="claims|sitelinks", format="json")
                for k_, e in d.get("entities", {}).items():
                    cl_ = e.get("claims", {})
                    p31 = [c["mainsnak"].get("datavalue", {}).get("value", {}).get("id") for c in cl_.get("P31", [])]
                    co = (cl_.get("P625") or [{}])[0].get("mainsnak", {}).get("datavalue", {}).get("value")
                    out[k_] = {"p31": p31, "coord": (co["latitude"], co["longitude"]) if co else None,
                               "title": e.get("sitelinks", {}).get("enwiki", {}).get("title"), "links": len(e.get("sitelinks", {}))}
            return out

        def dist(a, b):
            import math
            la1, lo1, la2, lo2 = map(math.radians, (*a, *b))
            return 6371 * 2 * math.asin(math.sqrt(math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2))
        for s in SOURCES:
            q = qid.get(s); se = ents([q])[q]; t0 = se["p31"][0] if se["p31"] else None
            near = sparql(cl, f"""SELECT ?item ?links WHERE {{ wd:{q} wdt:P625 ?loc .
              SERVICE wikibase:around {{ ?item wdt:P625 ?l . bd:serviceParam wikibase:center ?loc . bd:serviceParam wikibase:radius "3" . }}
              ?item wikibase:sitelinks ?links . FILTER(?links >= 30) }} ORDER BY DESC(?links) LIMIT 60""")
            ids = [b_["item"]["value"].rsplit("/", 1)[-1] for b_ in near if b_["item"]["value"].rsplit("/", 1)[-1] != q]
            E = ents(ids); l1 = []
            # physical places only (after a first pass picked UNESCO, Île-de-France, the French language...): an architectural
            # structure, or a mountain / island / square / park, via the type hierarchy
            phys = set()
            if ids:
                vals = " ".join(f"wd:{i}" for i in ids)
                for b_ in sparql(cl, f"""SELECT DISTINCT ?item WHERE {{ VALUES ?item {{ {vals} }} VALUES ?root {{ wd:Q811979 wd:Q8502 wd:Q23442 wd:Q174782 wd:Q22698 }}
                                       ?item wdt:P31/wdt:P279* ?root . }}"""):
                    phys.add(b_["item"]["value"].rsplit("/", 1)[-1])
            for i in ids:
                e = E.get(i)
                if e and e["title"] and i in phys and not (set(e["p31"]) & ADMIN):
                    l1.append(e["title"])
                if len(l1) == 3:
                    break
            l2 = []
            for t_ in se["p31"]:                        # first type first; fall back to the source's other types if needed
                if len(l2) == 3:
                    break
                same = sparql(cl, f"""SELECT ?item ?links WHERE {{ ?item wdt:P31 wd:{t_} ; wikibase:sitelinks ?links . FILTER(?links >= 30) }}
                  ORDER BY DESC(?links) LIMIT 150""")
                ids2 = [b_["item"]["value"].rsplit("/", 1)[-1] for b_ in same if b_["item"]["value"].rsplit("/", 1)[-1] != q]
                E2 = ents(ids2)
                for i in ids2:
                    e = E2.get(i)
                    if e and e["title"] and e["title"] not in l2 and e["coord"] and se["coord"] and dist(e["coord"], se["coord"]) > 500:
                        l2.append(e["title"])
                    if len(l2) == 3:
                        break
            sec = V["articles"].get(s)
            pool_same = [t for t, sc in V["articles"].items() if sc == sec and t != s and (V["wd"].get(V["qid"].get(t) or "", {}).get("links") or 0) >= 50
                         and (V["wd"].get(V["qid"].get(t) or "", {}).get("p31") or [None])[0] != t0]
            pool_diff = [t for t, sc in V["articles"].items() if sc != sec and sc != "People" and (V["wd"].get(V["qid"].get(t) or "", {}).get("links") or 0) >= 50]
            l3 = rng.sample(sorted(pool_same), 3) if len(pool_same) >= 3 else []
            l4 = rng.sample(sorted(pool_diff), 3)
            res[s] = {"qid": q, "p31": t0, "section": sec, "L1": l1, "L2": l2, "L3": l3, "L4": l4}
            print(s, res[s], flush=True)
    (OUT / "neighbours.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")


def questions():
    from concurrent.futures import ThreadPoolExecutor
    from modelmapper.core import oai
    N = json.loads((OUT / "neighbours.json").read_text(encoding="utf-8")); M = maps()
    SYS = "You write quiz questions. Reply in JSON only."
    jobs = []
    for s, v in N.items():
        jobs.append(("fact", s))
        if not (M.get(s) or {}).get("prompts"):
            jobs.append(("name", s))
        for lv in ("L1", "L2", "L3", "L4"):
            jobs += [("name", t) for t in v[lv]]
    jobs = sorted(set(jobs))

    def ask(job):
        kind, name = job
        if kind == "fact":
            q = (f'Write 8 different quiz questions ABOUT "{name}" that each NAME it in the question and have a short factual answer '
                 '(one to three words: a place, person, material, number, date, style...), e.g. "In which city is X?". The answer must not '
                 'contain the name. Reply {"items": [{"q": "...", "a": "..."}, ... 8]}')
        else:
            q = (f'Write 8 different quiz questions whose correct answer is exactly "{name}" (as named on English Wikipedia). None may '
                 'contain the name or an obvious part of it. Reply {"items": [{"q": "...", "a": "' + name + '"}, ... 8]}')
        for _ in range(3):
            try:
                return job, oai.chat_json(q, SYS).get("items", [])
            except Exception:  # noqa: BLE001
                pass
        return job, []
    with ThreadPoolExecutor(32) as ex:
        out = {f"{k}|{n}": items for (k, n), items in ex.map(ask, jobs)}
    (OUT / "questions.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{len(out)} question sets; Luna {oai.usage}")


def run():
    import torch
    from modelmapper.core import concept_pipeline as cp
    from modelmapper.core.config import DEFAULT_MODEL
    from modelmapper.core.neurons import DEVICE, load, mlps
    from modelmapper.core.pipeline_fast import leaks
    N = json.loads((OUT / "neighbours.json").read_text(encoding="utf-8")); Q = json.loads((OUT / "questions.json").read_text(encoding="utf-8")); M = maps()
    model, tok = load(DEFAULT_MODEL); W = model.config.intermediate_size; rng = random.Random(152)
    chat = lambda q: tok.encode(tok.apply_chat_template([{"role": "user", "content": q + " Answer with just the name." if not q.endswith("?") or True else q}],
                                                        tokenize=False, add_generation_prompt=True, enable_thinking=False))
    chat_fact = lambda q: tok.encode(tok.apply_chat_template([{"role": "user", "content": q + " Answer in as few words as possible."}],
                                                             tokenize=False, add_generation_prompt=True, enable_thinking=False))

    def logodds(groups):
        """groups: list of [(ids, answer_ids, kill_units or None)] - one pass per group (as-is rows included by caller)."""
        out = []
        for g in groups:
            seqs = [p + a[:-1] for p, a, _ in g]; T = max(map(len, seqs)); pad = tok.pad_token_id or 0
            x = torch.full((len(g), T), pad, device=DEVICE); mask = torch.zeros_like(x)
            for i, s_ in enumerate(seqs):
                x[i, :len(s_)] = torch.tensor(s_, device=DEVICE); mask[i, :len(s_)] = 1
            by = {}
            for i, (_, _, kill) in enumerate(g):
                for (l, n) in (kill or []):
                    by.setdefault(l, []).append((i, n))
            hks = []
            for l, items in by.items():
                rows = torch.tensor([a for a, _ in items], device=DEVICE); idx = torch.tensor([b for _, b in items], device=DEVICE)

                def h(mod, a, rows=rows, idx=idx):
                    y = a[0].clone(); y[rows, :, idx] = 0.0; return (y,)
                hks.append(mlps(model)[l].down_proj.register_forward_pre_hook(h))
            try:
                with torch.no_grad():
                    hs = model.model(input_ids=x, attention_mask=mask).last_hidden_state
            finally:
                for hk in hks:
                    hk.remove()
            vals = []
            for i, (p, a, _) in enumerate(g):
                pos = torch.arange(len(p) - 1, len(p) - 1 + len(a), device=DEVICE)
                lp = model.lm_head(hs[i, pos]).double().log_softmax(-1)
                s_ = min(lp[torch.arange(len(a), device=DEVICE), torch.tensor(a, device=DEVICE)].sum().item(), -1e-15)
                vals.append(s_ - float(torch.log(-torch.expm1(torch.tensor(s_, dtype=torch.float64)))))
            out.append(vals)
        return out

    def recalled(name, items, fact=False):
        """keep questions Qwen answers: best answer form (naming) / Luna's answer (fact); P >= MIN_P."""
        if not items:
            return []
        keep = []
        if fact:
            for it in items:
                ids = tok.encode(str(it.get("a", "")).strip())
                if not ids:
                    continue
                p = chat_fact(it["q"]); lo = logodds([[(p, ids, None)]])[0][0]
                if 1 / (1 + 2.718281828 ** (-lo)) >= cp.MIN_P:
                    keep.append((p, ids, it["q"], it["a"]))
            return keep
        base = name[4:].strip() if name.lower().startswith("the ") else name
        forms = list(dict.fromkeys([name, base, re.sub(r"\s*\(.*?\)\s*$", "", name).strip()]))
        best = None
        for fm in forms:
            ids = tok.encode(fm); ps = [chat(it["q"]) for it in items]
            lo = logodds([[(p, ids, None) for p in ps]])[0]
            k = [(p, ids, it["q"], fm) for p, l_, it in zip(ps, lo, items) if 1 / (1 + 2.718281828 ** (-l_)) >= cp.MIN_P and not leaks(it["q"], name)]
            if best is None or len(k) > len(best):
                best = k
        return best
    f = open(OUT / "results.jsonl", "w", encoding="utf-8")
    src_units = {}
    for s in N:
        m = M.get(s) or M.get("the " + s)
        src_units[s] = [((int(u[1:].split(":")[0]), int(u.split(":")[1])), a) for u, a in m["top"]] if m else []
    for si, s in enumerate(N):
        m = M.get(s) or M.get("the " + s)
        if not m:
            print("no map for", s); continue
        units = src_units[s]; k = max(1, min(int(m.get("k_mean") or 5), 5))
        top5 = [u for u, _ in units]
        other = src_units[list(N)[(si + 1) % len(N)]]
        abl = {"src_k": top5[:k], "src5": top5, "src5_late": [u for u in top5 if u[0] >= 20], "src5_early": [u for u in top5 if u[0] < 20],
               "rand5_a": [(l, rng.randrange(W)) for l, _ in top5], "rand5_b": [(l, rng.randrange(W)) for l, _ in top5],
               "other5": [u for u, _ in other]}
        abl = {a: v for a, v in abl.items() if v}
        sets = {}
        if m.get("prompts"):
            ids = m.get("answer_ids") or tok.encode(m.get("form") or s)
            sets["src_name"] = [(tok.encode(p), ids) for p in m["prompts"][:12]]
        else:
            sets["src_name"] = [(p, ids) for p, ids, _, _ in recalled(s, Q.get(f"name|{s}", []))]
        sets["src_fact"] = [(p, ids) for p, ids, _, _ in recalled(s, Q.get(f"fact|{s}", []), fact=True)]
        for lv in ("L1", "L2", "L3", "L4"):
            for t in N[s][lv]:
                sets[f"{lv}|{t}"] = [(p, ids) for p, ids, _, _ in recalled(t, Q.get(f"name|{t}", []))]
        for sname, rows in sets.items():
            if len(rows) < 2:
                continue
            res = {}
            for a, kill in abl.items():
                g = [(p, ids, None) for p, ids in rows] + [(p, ids, kill) for p, ids in rows]
                v = logodds([g])[0]; n_ = len(rows)
                res[a] = round(stx.mean(b - z for b, z in zip(v[:n_], v[n_:])), 4)
            rec = {"source": s, "set": sname, "n_questions": len(rows), "drops": res, "units": {a: [f"L{l}:{n}" for l, n in v] for a, v in abl.items()}}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
        print(f"done {s}: sets {[(k_, len(v)) for k_, v in sets.items()]}", flush=True)


def summary():
    R = [json.loads(l) for l in (OUT / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    md = stx.median
    L = [f"Question-side test + specificity gradient: {len({r['source'] for r in R})} sources, {len(R)} question sets"]
    groups = [("source naming", lambda r: r["set"] == "src_name"), ("source FACTS", lambda r: r["set"] == "src_fact")] + \
             [(f"neighbours {lv}", lambda r, lv=lv: r["set"].startswith(lv + "|")) for lv in ("L1", "L2", "L3", "L4")]
    abls = ["src_k", "src5", "src5_late", "src5_early", "rand5_a", "rand5_b", "other5"]
    L.append(f"median log-odds drop when zeroed (rows) on each question set (columns: ablations)\n{'':<18}" + "".join(f"{a:>11}" for a in abls))
    for name, fn in groups:
        rs = [r for r in R if fn(r)]
        L.append(f"{name:<14}(n={len(rs):<3})" + "".join(f"{md([r['drops'][a] for r in rs if a in r['drops']]) if any(a in r['drops'] for r in rs) else float('nan'):>11.2f}" for a in abls))
    L.append("\nper source (src5 drop): naming | facts | L1 | L2 | L3 | L4 neighbour medians")
    for s in sorted({r["source"] for r in R}):
        rs = [r for r in R if r["source"] == s]
        g = lambda pref: md([r["drops"]["src5"] for r in rs if r["set"].startswith(pref) and "src5" in r["drops"]]) if any(r["set"].startswith(pref) for r in rs) else float("nan")
        L.append(f"  {s:<22} {g('src_name'):6.2f} | {g('src_fact'):6.2f} | {g('L1'):6.2f} | {g('L2'):6.2f} | {g('L3'):6.2f} | {g('L4'):6.2f}")
    txt = "\n".join(L)
    (OUT / "summary.txt").write_text(txt + "\n", encoding="utf-8"); print(txt)


if __name__ == "__main__":
    {"neighbours": neighbours, "questions": questions, "run": run, "summary": summary}[sys.argv[1]]()

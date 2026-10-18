"""Job-finding for knowledge neurons: a neuron's job as its place in an ENTITY x RELATION grid. (Luna; GPU)

James (18 Oct): naming maps are 'say it' machinery; knowledge about a concept seems to live earlier - find mechanisms
for the job of earlier neurons. Ground truth: facts. 30 landmarks x 5 fixed relations (city, country, designer,
century built, main material), 3 fixed phrasings each (same wording for every landmark), answers from Luna, kept only
where Qwen answers (P >= MIN_P on >= 2 of 3 phrasings).
Stages:
  entities  Luna gives the 5 short answers per landmark                                  -> entities.json
  build     recall filter; subject-token positions (the landmark's name in each prompt)   -> facts.json
  map       per fact: attribution (activation x gradient of the answer's log-prob, every neuron, summed over prompts; split
            into subject-token positions vs the rest) -> top-100 -> each zeroed alone (every position) -> drop -> fact map
  grid      each fact's top-3 neurons zeroed alone, measured on EVERY fact: same entity / same relation / neither ->
            job class per neuron (entity, relation, fact-specific, generic)
  position  each fact's top-3 zeroed only at the subject tokens vs only at the last position vs everywhere
  summary   layers of fact neurons vs naming neurons; job classes by layer; positions; overlap with naming maps
Predictions (stated before running): fact neurons sit earlier than naming neurons; entity neurons are early and act at
the subject tokens; relation neurons ('city of') sit mid-to-late.
Output: D:/already-clean-data/facts/.  Usage: python src/loop_v2/facts.py entities | build | map | grid | position | summary
"""
import json
import os
import random
import statistics as stx
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
OUT = Path(os.environ.get("AC_DATA_DIR", "D:/already-clean-data")) / "facts"
ENTITIES = ["Eiffel Tower", "Statue of Liberty", "Louvre", "Notre-Dame de Paris", "Taj Mahal", "Colosseum", "Sydney Opera House",
            "Burj Khalifa", "Forbidden City", "Sagrada Família", "Hagia Sophia", "St. Peter's Basilica", "Parthenon", "Golden Gate Bridge",
            "Arc de Triomphe", "Sydney Harbour Bridge", "Tokyo Tower", "Empire State Building", "Tower Bridge", "Brooklyn Bridge",
            "Palace of Versailles", "Alhambra", "Park Güell", "Casa Batlló", "Sistine Chapel", "Blue Mosque", "Petronas Towers", "CN Tower",
            "Leaning Tower of Pisa", "Milan Cathedral"]
TEMPLATES = {
    "city": ["In which city is {X} located?", "{X} stands in which city?", "Name the city where {X} is."],
    "country": ["In which country is {X}?", "{X} is located in which country?", "Which country is home to {X}?"],
    "designer": ["Who designed {X}?", "Who was the architect of {X}?", "{X} was designed by whom?"],
    "century": ["In which century was {X} built?", "{X} was completed in which century?", "During which century was {X} constructed?"],
    "material": ["What material is {X} mainly made of?", "{X} is built chiefly from which material?", "What is the main building material of {X}?"]}
SUFFIX = " Answer in as few words as possible."


def entities():
    import oai
    from concurrent.futures import ThreadPoolExecutor
    OUT.mkdir(parents=True, exist_ok=True)
    SYS = "You are a precise encyclopedia. Reply in JSON only."

    def ask(e):
        q = (f'For "{e}", give the short standard answers: {{"city": "...", "country": "...", "designer": "<the main architect or designer, '
             f'surname or full name as usually cited>", "century": "<ordinal only, e.g. 19th>", "material": "<one or two words, e.g. wrought iron, '
             f'marble, steel>"}}. Use the most widely cited answer.')
        for _ in range(3):
            try:
                return e, oai.chat_json(q, SYS)
            except Exception:  # noqa: BLE001
                pass
        return e, None
    with ThreadPoolExecutor(16) as ex:
        res = dict(ex.map(ask, ENTITIES))
    (OUT / "entities.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False)[:1500]); print(oai.usage)


def setup():
    import torch
    from config import DEFAULT_MODEL
    from neurons import load
    model, tok = load(DEFAULT_MODEL)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def encode_with_subject(tok, question, name):
    text = tok.apply_chat_template([{"role": "user", "content": question + SUFFIX}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    a = text.find(name); b = a + len(name)
    subj = [i for i, (s, e) in enumerate(enc["offset_mapping"]) if a >= 0 and s < b and e > a]
    return enc["input_ids"], subj


def logodds_groups(model, tok, groups):
    """groups: [[(prompt_ids, answer_ids, kills)]], kills = list of (layer, neuron, positions or None=all). One pass per group."""
    import torch
    from neurons import DEVICE, mlps
    out = []
    for g in groups:
        seqs = [p + a[:-1] for p, a, _ in g]; T = max(map(len, seqs)); pad = tok.pad_token_id or 0
        x = torch.full((len(g), T), pad, device=DEVICE); mask = torch.zeros_like(x)
        for i, s in enumerate(seqs):
            x[i, :len(s)] = torch.tensor(s, device=DEVICE); mask[i, :len(s)] = 1
        by = defaultdict(list)
        for i, (_, _, kills) in enumerate(g):
            for (l, n, pos) in (kills or []):
                by[l].append((i, n, pos))
        hks = []
        for l, items in by.items():
            def h(mod, a, items=items):
                y = a[0].clone()
                for i, n, pos in items:
                    if pos is None:
                        y[i, :, n] = 0.0
                    else:
                        y[i, pos, n] = 0.0
                return (y,)
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
            s = min(lp[torch.arange(len(a), device=DEVICE), torch.tensor(a, device=DEVICE)].sum().item(), -1e-15)
            vals.append(s - float(torch.log(-torch.expm1(torch.tensor(s, dtype=torch.float64)))))
        out.append(vals)
    return out


def build():
    import concept_pipeline as cp
    E = json.loads((OUT / "entities.json").read_text(encoding="utf-8")); model, tok = setup()
    facts = []
    for e, ans in E.items():
        if not ans:
            continue
        for rel, tpls in TEMPLATES.items():
            a = str(ans.get(rel, "")).strip()
            if not a or "unknown" in a.lower():           # e.g. Colosseum designer "Unknown": not a fact to map
                continue
            forms = list(dict.fromkeys([a, a[:1].upper() + a[1:], a + " century" if rel == "century" else a]))
            enc = [encode_with_subject(tok, t.format(X=e), e) for t in tpls]
            best = None
            for fm in forms:
                ids = tok.encode(fm)
                lo = logodds_groups(model, tok, [[(p, ids, None) for p, _ in enc]])[0]
                ok = [i for i, l in enumerate(lo) if 1 / (1 + 2.718281828 ** (-l)) >= cp.MIN_P]
                if best is None or len(ok) > len(best[1]):
                    best = (fm, ok, ids, lo)
            fm, ok, ids, lo = best
            rec = {"entity": e, "relation": rel, "answer": a, "form": fm, "answer_ids": ids, "base": [round(x, 3) for x in lo],
                   "prompts": [enc[i][0] for i in ok], "subject_pos": [enc[i][1] for i in ok], "recalled": len(ok)}
            if len(ok) >= 2:
                facts.append(rec)
    (OUT / "facts.json").write_text(json.dumps(facts, ensure_ascii=False), encoding="utf-8")
    c = Counter(f["relation"] for f in facts)
    print(f"{len(facts)} facts kept (>= 2 of 3 phrasings recalled) of {sum(1 for v in E.values() if v) * 5}; by relation {dict(c)}")


def fact_map():
    import torch
    from neurons import DEVICE, mlps
    F = json.loads((OUT / "facts.json").read_text(encoding="utf-8")); model, tok = setup()
    L, W = model.config.num_hidden_layers, model.config.intermediate_size
    f = open(OUT / "maps.jsonl", "w", encoding="utf-8")
    for k, fa in enumerate(F):
        attr = torch.zeros(L, W, device=DEVICE); attr_subj = torch.zeros(L, W, device=DEVICE)
        for p, subj in zip(fa["prompts"], fa["subject_pos"]):
            acts = {}

            def keep(l):
                def h(mod, a):
                    a0 = a[0]
                    if not a0.requires_grad:
                        a0 = a0.detach().requires_grad_(True); acts[l] = a0; return (a0,)
                    a0.retain_grad(); acts[l] = a0
                return h
            hks = [mlps(model)[l].down_proj.register_forward_pre_hook(keep(l)) for l in range(L)]
            ids = torch.tensor([p + fa["answer_ids"][:-1]], device=DEVICE); n = len(p)
            try:
                with torch.enable_grad():
                    lp = model(input_ids=ids).logits[0, n - 1:].float().log_softmax(-1)
                    s = lp[torch.arange(len(fa["answer_ids"]), device=DEVICE), torch.tensor(fa["answer_ids"], device=DEVICE)].sum()
                    s.backward()
            finally:
                for h in hks:
                    h.remove()
            for l in range(L):
                pa = (acts[l][0, :n].float() * acts[l].grad[0, :n].float())                # [n, W] (prompt positions only)
                attr[l] += pa.sum(0) / len(fa["prompts"])
                if subj:
                    attr_subj[l] += pa[subj].sum(0) / len(fa["prompts"])
        top = [divmod(int(i), W) for i in attr.flatten().topk(100).indices]
        groups = []
        for c0 in range(0, len(top), 32):
            g = [(p, fa["answer_ids"], None) for p in fa["prompts"]]
            for (l, nn) in top[c0:c0 + 32]:
                g += [(p, fa["answer_ids"], [(l, nn, None)]) for p in fa["prompts"]]
            groups.append(g)
        drops = {}
        for c0, v in zip(range(0, len(top), 32), logodds_groups(model, tok, groups)):
            P = len(fa["prompts"]); base = v[:P]
            for j, (l, nn) in enumerate(top[c0:c0 + 32]):
                drops[f"L{l}:{nn}"] = round(stx.mean(b - x for b, x in zip(base, v[P * (j + 1):P * (j + 2)])), 4)
        ranked = sorted(drops, key=lambda u: -drops[u])
        rec = {"entity": fa["entity"], "relation": fa["relation"], "answer": fa["answer"], "n_prompts": len(fa["prompts"]),
               "top": [{"neuron": u, "drop": drops[u], "attr": round(float(attr[int(u[1:].split(':')[0]), int(u.split(':')[1])]), 4),
                        "attr_subject_share": round(float(attr_subj[int(u[1:].split(':')[0]), int(u.split(':')[1])] /
                                                          (attr[int(u[1:].split(':')[0]), int(u.split(':')[1])] + 1e-9)), 3)} for u in ranked[:30]]}
        f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
        if k % 10 == 0:
            print(f"map {k + 1}/{len(F)}: {fa['entity']} / {fa['relation']} top {[(x['neuron'], x['drop']) for x in rec['top'][:3]]}", flush=True)


def grid():
    F = json.loads((OUT / "facts.json").read_text(encoding="utf-8")); M = [json.loads(l) for l in (OUT / "maps.jsonl").read_text(encoding="utf-8").splitlines()]
    model, tok = setup()
    units = []
    for m in M:
        for x in m["top"][:3]:
            if x["drop"] > 0.2:
                units.append((m["entity"], m["relation"], x["neuron"]))
    f = open(OUT / "grid.jsonl", "w", encoding="utf-8")
    uniq = sorted({u for _, _, u in units})
    for k, fa in enumerate(F):
        P = len(fa["prompts"]); res = {}
        for c0 in range(0, len(uniq), 48):
            chunk = uniq[c0:c0 + 48]
            g = [(p, fa["answer_ids"], None) for p in fa["prompts"]]
            for u in chunk:
                l, n = int(u[1:].split(":")[0]), int(u.split(":")[1])
                g += [(p, fa["answer_ids"], [(l, n, None)]) for p in fa["prompts"]]
            v = logodds_groups(model, tok, [g])[0]; base = v[:P]
            for j, u in enumerate(chunk):
                res[u] = round(stx.mean(b - x for b, x in zip(base, v[P * (j + 1):P * (j + 2)])), 4)
        f.write(json.dumps({"entity": fa["entity"], "relation": fa["relation"], "drops": res}, ensure_ascii=False) + "\n"); f.flush()
        if k % 20 == 0:
            print(f"grid {k + 1}/{len(F)}", flush=True)
    (OUT / "grid_units.json").write_text(json.dumps(units, ensure_ascii=False), encoding="utf-8")


def position():
    F = {(x["entity"], x["relation"]): x for x in json.loads((OUT / "facts.json").read_text(encoding="utf-8"))}
    M = [json.loads(l) for l in (OUT / "maps.jsonl").read_text(encoding="utf-8").splitlines()]
    model, tok = setup()
    f = open(OUT / "position.jsonl", "w", encoding="utf-8")
    for m in M:
        fa = F[(m["entity"], m["relation"])]; P = len(fa["prompts"])
        for x in m["top"][:3]:
            l, n = int(x["neuron"][1:].split(":")[0]), int(x["neuron"].split(":")[1])
            g = [(p, fa["answer_ids"], None) for p in fa["prompts"]]
            g += [(p, fa["answer_ids"], [(l, n, None)]) for p in fa["prompts"]]
            g += [(p, fa["answer_ids"], [(l, n, sp)] if sp else []) for p, sp in zip(fa["prompts"], fa["subject_pos"])]
            g += [(p, fa["answer_ids"], [(l, n, [len(p) - 1])]) for p in fa["prompts"]]
            v = logodds_groups(model, tok, [g])[0]; base = v[:P]
            d = lambda j: round(stx.mean(b - y for b, y in zip(base, v[P * j:P * (j + 1)])), 4)
            f.write(json.dumps({"entity": m["entity"], "relation": m["relation"], "neuron": x["neuron"], "layer": l,
                                "drop_all": d(1), "drop_subject_only": d(2), "drop_last_only": d(3)}, ensure_ascii=False) + "\n")
    f.flush()


def summary():
    import glob
    M = [json.loads(l) for l in (OUT / "maps.jsonl").read_text(encoding="utf-8").splitlines()]
    G = [json.loads(l) for l in (OUT / "grid.jsonl").read_text(encoding="utf-8").splitlines()]
    Pz = [json.loads(l) for l in (OUT / "position.jsonl").read_text(encoding="utf-8").splitlines()] if (OUT / "position.jsonl").exists() else []
    units = json.loads((OUT / "grid_units.json").read_text(encoding="utf-8"))
    md = stx.median; lay = lambda u: int(u[1:].split(":")[0])
    L = [f"Facts experiment: {len(M)} facts mapped ({dict(Counter(m['relation'] for m in M))})"]
    top1 = [lay(m["top"][0]["neuron"]) for m in M if m["top"][0]["drop"] > 0.2]
    top5 = [lay(x["neuron"]) for m in M for x in m["top"][:5] if x["drop"] > 0.2]
    naming = []
    for b in glob.glob("D:/already-clean-data/loop_v2/run1/results/**/map/*/records.jsonl", recursive=True):
        for line in Path(b).read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["row"]["status"] == "ok":
                naming += [lay(e["neuron"]) for e in r["rank"]["top"][:5]]
    L.append(f"layers of fact neurons (top-5 with drop > 0.2; n={len(top5)}): median {md(top5)}; share < 20: {sum(l < 20 for l in top5) / len(top5):.0%}; "
             f"share >= 28: {sum(l >= 28 for l in top5) / len(top5):.0%}")
    L.append(f"layers of NAMING neurons (run-1 top-5, n={len(naming)}): median {md(naming)}; share < 20: {sum(l < 20 for l in naming) / len(naming):.0%}; "
             f"share >= 28: {sum(l >= 28 for l in naming) / len(naming):.0%}")
    L.append(f"fact top-1 drop median {md([m['top'][0]['drop'] for m in M]):.2f}; facts whose top-1 drop > 1: {sum(m['top'][0]['drop'] > 1 for m in M)}/{len(M)}")
    # job classes from the grid: for each unit (from fact e,r), mean drop on same-entity-other-relation, same-relation-other-entity, neither
    dg = {(g["entity"], g["relation"]): g["drops"] for g in G}
    cls = []; seen = set()
    for e, r, u in units:
        if (e, r, u) in seen:
            continue
        seen.add((e, r, u))
        own = dg[(e, r)].get(u, 0)
        se = [d[u] for (e2, r2), d in dg.items() if e2 == e and r2 != r and u in d]
        sr = [d[u] for (e2, r2), d in dg.items() if r2 == r and e2 != e and u in d]
        ne = [d[u] for (e2, r2), d in dg.items() if e2 != e and r2 != r and u in d]
        mse, msr, mne = (stx.mean(x) if x else 0 for x in (se, sr, ne))
        thr = max(0.2, 0.25 * own)
        kind = ("generic" if mne > thr else "entity+relation" if mse > thr and msr > thr else "entity" if mse > thr else "relation" if msr > thr else "fact-specific")
        cls.append({"unit": u, "layer": lay(u), "fact": f"{e}|{r}", "own": own, "same_entity": round(mse, 3), "same_relation": round(msr, 3),
                    "neither": round(mne, 3), "class": kind})
    (OUT / "grid_classes.json").write_text(json.dumps(cls, indent=1, ensure_ascii=False), encoding="utf-8")
    L.append(f"\njob classes of fact neurons (grid: each zeroed on every fact; class threshold max(0.2, 25% of its own-fact drop)): {dict(Counter(c['class'] for c in cls))}")
    for b in (0, 10, 20, 28):
        cs = [c for c in cls if b <= c["layer"] < {0: 10, 10: 20, 20: 28, 28: 36}[b]]
        if cs:
            L.append(f"  layers {b}-{ {0: 9, 10: 19, 20: 27, 28: 35}[b]}: n={len(cs)} {dict(Counter(c['class'] for c in cs))}")
    rel = defaultdict(Counter)
    for c in cls:
        rel[c["fact"].split("|")[1]][c["class"]] += 1
    L.append("  by relation: " + "; ".join(f"{r}: {dict(v)}" for r, v in rel.items()))
    ex = [c for c in cls if c["class"] in ("entity", "relation")][:12]
    L.append("  examples: " + "; ".join(f"{c['unit']} ({c['fact']}): {c['class']} own {c['own']:.2f}, same-entity {c['same_entity']:.2f}, same-relation {c['same_relation']:.2f}" for c in ex))
    if Pz:
        L.append(f"\nposition (fact top-3, n={len(Pz)}): median drop zeroed everywhere {md(p['drop_all'] for p in Pz):.2f}, only at the subject tokens "
                 f"{md(p['drop_subject_only'] for p in Pz):.2f}, only at the last position {md(p['drop_last_only'] for p in Pz):.2f}")
        for b in (0, 10, 20, 28):
            ps = [p for p in Pz if b <= p["layer"] < {0: 10, 10: 20, 20: 28, 28: 36}[b]]
            if ps:
                L.append(f"  layers {b}+: n={len(ps)} all {md(p['drop_all'] for p in ps):.2f} | subject {md(p['drop_subject_only'] for p in ps):.2f} | last {md(p['drop_last_only'] for p in ps):.2f}")
    txt = "\n".join(L)
    (OUT / "summary.txt").write_text(txt + "\n", encoding="utf-8"); print(txt)


if __name__ == "__main__":
    {"entities": entities, "build": build, "map": fact_map, "grid": grid, "position": position, "summary": summary}[sys.argv[1]]()

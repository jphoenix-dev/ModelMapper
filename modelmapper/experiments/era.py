"""Era neurons across formats: is L31:8930 'the 1800s' whether written '19th', '18xx', '1880s' or 'Victorian'? (Luna; GPU)

James (18 Oct). numeric.py found L31:8930 hurts 18 of 27 years 18xx and nothing else - the same neuron facts2 found as the
'19th century' answer-value neuron; L28:9936 is facts2's '18th' neuron and was top for Mozart 1756 / Magic Flute 1791.
Era neurons tested: every facts2 answer-value neuron whose answer is a century (chain.jsonl), plus numeric.py's L33:12258
(17xx) and L33:12280 (18xx, mixed). Controls: 12 random neurons from layers 28-33.
Entities: numeric.py's 75 year facts + Luna lists of well-known landmarks / people (birth) / works per century, 14th-21st;
kept where Qwen's own year answer falls in the same century as Luna's year.
Stages:
  entities  Luna lists                                                                   -> era_entities.json
  build     per entity 4 question formats - year / century / decade / historical period ('In which ... was X completed /
            born / created?') + a control 'Tell me about X.'; Qwen's greedy answers; recall filter  -> era_build.json
  acts      each era neuron's activation at the last prompt position, every entity x format          -> era_acts.jsonl
  text      plain sentences that MENTION a date, read (no question): '... completed in 1889.' / 'in the 1880s.' / 'in the
            19th century.' / 'in the nineteenth century.' / 'in the eighteen-hundreds.' / 'in the Victorian era.' (named eras
            mapped to a century) - 3 subjects x 8 years per century; activation at the last token and max over the date span
  readout   each era neuron's direct output: lm_head(final-norm weight * down_proj column), top / bottom 15 tokens
  causal    each era neuron (and the controls) zeroed at every position; drop in log-odds of Qwen's own answer in the year /
            century / decade formats, for entities in vs out of the neuron's era
  summary
Selectivity = AUC of activation, in-era entities vs others; the sign is fixed in the YEAR format and reused unchanged for
every other format (so the other formats are out-of-sample).
Predictions (stated before running):
  B1  L31:8930: AUC >= 0.8 in the year, century AND decade question formats.
  B2  zeroing L31:8930 drops in-era answers by a median >= 1.0 log-odds in each of the year / century / decade formats, and
      out-of-era answers by less than the random-neuron 99th percentile.
  B3  of the facts2 century neurons with >= 5 in-era entities, at least half reach AUC >= 0.8 in >= 2 of the 3 formats.
  B4  reading dates in plain text is less selective than answering: median text AUC < median question AUC (era neurons
      are output-side; cf. polysemantic inputs, S4b).
  B5  L31:8930's direct output includes era / digit tokens ('8', '19', 'nineteenth' or similar) among its top 15.
Output: D:/already-clean-data/era/.  Usage: python src/loop_v2/era.py entities | build | acts | text | readout | causal | summary
"""
import json
import random
import re
import statistics as stx
import sys
from collections import Counter, defaultdict
from pathlib import Path
from modelmapper.core.settings import DATA_DIR, REPO_ROOT

ROOT = REPO_ROOT

from modelmapper.experiments import facts as F1  # noqa: E402

OUT = (DATA_DIR / "era")
NUM = (DATA_DIR / "numeric")
CENTS = list(range(14, 22))
VERB = {"landmark": "completed", "person": "born", "work": "created"}
SUF_NUM, SUF_FEW = " Answer with just the number.", " Answer in as few words as possible."
FORMATS = {"year": ("In which year was {X} {V}?", SUF_NUM), "century": ("In which century was {X} {V}?", SUF_FEW),
           "decade": ("In which decade was {X} {V}?", SUF_FEW), "period": ("In which historical period was {X} {V}?", SUF_FEW),
           "control": ("Tell me about {X}.", "")}
WORDS = {14: "fourteenth", 15: "fifteenth", 16: "sixteenth", 17: "seventeenth", 18: "eighteenth", 19: "nineteenth", 20: "twentieth", 21: "twenty-first"}
HUNDREDS = {14: "thirteen", 15: "fourteen", 16: "fifteen", 17: "sixteen", 18: "seventeen", 19: "eighteen", 20: "nineteen"}
ERAS = [("Tudor", 16), ("Elizabethan", 16), ("Jacobean", 17), ("Baroque", 17), ("Georgian", 18), ("Enlightenment", 18),
        ("Napoleonic", 19), ("Victorian", 19), ("Edwardian", 20), ("interwar", 20), ("Cold War", 20), ("digital", 21)]
SUBJ = ["The building was completed", "She was born", "The novel was published"]


def ordinal(c):
    return f"{c}{'th' if 10 <= c % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(c % 10, 'th')}"


def century(y):
    return (y - 1) // 100 + 1


def era_neurons():
    R = [json.loads(l) for l in ((DATA_DIR / "facts2") / "chain.jsonl").read_text(encoding="utf-8").splitlines()]
    lab = defaultdict(set)
    for r in R:
        m = re.fullmatch(r"(\d+)(st|nd|rd|th)( century)?", r["answer"])
        if m:
            lab[r["value_neuron"]].add(int(m.group(1)))
    lab["L33:12258"].add(18); lab["L33:12280"].add(19)
    return {u: sorted(v) for u, v in lab.items()}


def nulls(model):
    rng = random.Random(223); W = model.config.intermediate_size
    return [f"L{rng.randrange(28, 34)}:{rng.randrange(W)}" for _ in range(12)]


def entities():
    from modelmapper.core import oai
    from concurrent.futures import ThreadPoolExecutor
    OUT.mkdir(parents=True, exist_ok=True)
    SYS = "You are a precise encyclopedia. Reply in JSON only."

    def ask(c):
        lo, hi = (c - 1) * 100 + 1, c * 100
        q = (f"List 36 very well-known things from the {ordinal(c)} century (years {lo}-{min(hi, 2024)}): 12 buildings or monuments "
             f"(year completed), 12 people (birth year in this range), 12 artworks, books or pieces of music (year created). Reply JSON "
             f'{{"items": [{{"name": "...", "kind": "landmark|person|work", "year": 1234}}]}}. Most widely cited years; no duplicates.')
        for _ in range(3):
            try:
                return c, oai.chat_json(q, SYS)["items"]
            except Exception:  # noqa: BLE001
                pass
        return c, []
    with ThreadPoolExecutor(8) as ex:
        got = dict(ex.map(ask, CENTS))
    ents = {}
    for f in json.loads((NUM / "facts.json").read_text(encoding="utf-8")):
        if f["relation"] == "year" and len(f["answer"]) == 4:
            ents[f["entity"]] = {"kind": f["kind"], "year": int(f["answer"]), "source": "numeric"}
    for c, items in got.items():
        for it in items:
            try:
                y = int(it["year"])
            except Exception:  # noqa: BLE001
                continue
            if it.get("kind") in VERB and century(y) == c and it.get("name") and it["name"] not in ents:
                ents[it["name"]] = {"kind": it["kind"], "year": y, "source": "luna"}
    (OUT / "era_entities.json").write_text(json.dumps(ents, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{len(ents)} entities; by century {dict(sorted(Counter(century(v['year']) for v in ents.values()).items()))}; Luna {oai.usage}")


def _chat(tok, q):
    return tok(tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=False),
               add_special_tokens=False)["input_ids"]


def _generate(model, tok, prompts, n=10, bs=48):
    import torch
    out = []; stop = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}
    for i in range(0, len(prompts), bs):
        ch = prompts[i:i + bs]; T = max(map(len, ch)); pad = tok.pad_token_id or 0
        x = torch.full((len(ch), T), pad, device=model.device); m = torch.zeros_like(x)
        for j, p in enumerate(ch):
            x[j, T - len(p):] = torch.tensor(p); m[j, T - len(p):] = 1
        with torch.no_grad():
            g = model.generate(input_ids=x, attention_mask=m, max_new_tokens=n, do_sample=False, pad_token_id=pad)[:, T:].tolist()
        for s in g:
            a = []
            for t in s:
                if t in stop:
                    break
                a.append(t)
            out.append(a)
    return out


def build():
    E = json.loads((OUT / "era_entities.json").read_text(encoding="utf-8")); model, tok = F1.setup()
    names = sorted(E); rows = []
    for e in names:
        for fm, (t, suf) in FORMATS.items():
            rows.append((e, fm, _chat(tok, t.format(X=e, V=VERB[E[e]["kind"]]) + suf)))
    gen = _generate(model, tok, [p for e, fm, p in rows if fm != "control"])
    it = iter(gen); B = defaultdict(dict)
    for e, fm, p in rows:
        a = next(it) if fm != "control" else []
        B[e][fm] = {"prompt": p, "answer_ids": a, "answer": tok.decode(a).strip()}
    keep = {}
    for e in names:
        m = re.search(r"\d{3,4}", B[e]["year"]["answer"])
        qy = int(m.group()) if m else None
        rec = {**E[e], "century": century(E[e]["year"]), "qwen_year": qy, "formats": B[e]}
        rec["recalled"] = qy is not None and century(qy) == century(E[e]["year"])
        cw = B[e]["century"]["answer"].lower(); c = century(E[e]["year"])
        rec["century_answer_ok"] = ordinal(c) in cw or WORDS.get(c, "~") in cw
        keep[e] = rec
    (OUT / "era_build.json").write_text(json.dumps(keep, ensure_ascii=False), encoding="utf-8")
    ok = [v for v in keep.values() if v["recalled"]]
    print(f"{len(keep)} entities; recalled (Qwen's year in Luna's century) {len(ok)}; by century "
          f"{dict(sorted(Counter(v['century'] for v in ok).items()))}; century answer consistent {sum(v['century_answer_ok'] for v in ok)}/{len(ok)}")
    for e in names[:6]:
        print("  ", e, E[e]["year"], {fm: keep[e]["formats"][fm]["answer"] for fm in ("year", "century", "decade", "period")})


def _capture(model, tok, seqs, units, where):
    """Activations of units at given positions. seqs: token lists; where(i, seq) -> list of positions; returns per seq
    {unit: [acts at those positions]}."""
    import torch
    from modelmapper.core.neurons import DEVICE, mlps
    by = defaultdict(list)
    for u in units:
        by[int(u[1:].split(":")[0])].append((u, int(u.split(":")[1])))
    out = []
    for i0 in range(0, len(seqs), 64):
        ch = seqs[i0:i0 + 64]; T = max(map(len, ch)); pad = tok.pad_token_id or 0
        x = torch.full((len(ch), T), pad, device=DEVICE); m = torch.zeros_like(x)
        for j, s in enumerate(ch):
            x[j, :len(s)] = torch.tensor(s, device=DEVICE); m[j, :len(s)] = 1
        cap = {}
        hks = [mlps(model)[l].down_proj.register_forward_pre_hook(lambda mod, a, l=l, us=us: cap.__setitem__(l, a[0][:, :, [n for _, n in us]].float().cpu()))
               for l, us in by.items()]
        try:
            with torch.no_grad():
                model.model(input_ids=x, attention_mask=m)
        finally:
            for h in hks:
                h.remove()
        for j, s in enumerate(ch):
            ps = where(i0 + j, s); d = {}
            for l, us in by.items():
                for k, (u, _) in enumerate(us):
                    d[u] = [round(cap[l][j, p, k].item(), 4) for p in ps]
            out.append(d)
    return out


def acts():
    B = json.loads((OUT / "era_build.json").read_text(encoding="utf-8")); model, tok = F1.setup()
    units = list(era_neurons()) + nulls(model)
    rows = [(e, fm, v["formats"][fm]["prompt"]) for e, v in B.items() if v["recalled"] for fm in FORMATS]
    A = _capture(model, tok, [p for _, _, p in rows], units, lambda i, s: [len(s) - 1])
    with open(OUT / "era_acts.jsonl", "w", encoding="utf-8") as f:
        for (e, fm, _), d in zip(rows, A):
            f.write(json.dumps({"entity": e, "format": fm, "century": B[e]["century"], "acts": {u: v[0] for u, v in d.items()}}, ensure_ascii=False) + "\n")
    print(f"{len(rows)} rows x {len(units)} neurons")


def text():
    model, tok = F1.setup(); units = list(era_neurons()) + nulls(model); rng = random.Random(227)
    items = []
    for c in CENTS:
        lo, hi = (c - 1) * 100 + 1, min(c * 100, 2024)
        for y in rng.sample(range(lo, hi + 1), 8):
            for s in SUBJ:
                items += [(c, "digits", f"{s} in {y}.", f"{y}"), (c, "decade", f"{s} in the {y // 10 * 10}s.", f"{y // 10 * 10}s"),
                          (c, "ordinal", f"{s} in the {ordinal(c)} century.", f"{ordinal(c)} century"),
                          (c, "word", f"{s} in the {WORDS[c]} century.", f"{WORDS[c]} century")]
                if c in HUNDREDS:
                    items.append((c, "hundreds", f"{s} in the {HUNDREDS[c]}-hundreds.", f"{HUNDREDS[c]}-hundreds"))
    for name, c in ERAS:
        for s in SUBJ:
            items.append((c, "named era", f"{s} in the {name} era.", f"{name} era"))
    items = list(dict.fromkeys(items))
    encs = [tok(t, return_offsets_mapping=True, add_special_tokens=False) for _, _, t, _ in items]
    spans = []
    for (_, _, t, span), enc in zip(items, encs):
        a = t.find(span); b = a + len(span)
        spans.append([i for i, (s, e) in enumerate(enc["offset_mapping"]) if s < b and e > a] + [len(enc["input_ids"]) - 1])
    A = _capture(model, tok, [enc["input_ids"] for enc in encs], units, lambda i, s: spans[i])
    with open(OUT / "era_text.jsonl", "w", encoding="utf-8") as f:
        for (c, fm, t, _), d in zip(items, A):
            f.write(json.dumps({"century": c, "format": fm, "text": t, "acts_span_then_end": d}, ensure_ascii=False) + "\n")
    print(f"{len(items)} sentences x {len(units)} neurons; formats {dict(Counter(fm for _, fm, _, _ in items))}")


def readout():
    import torch
    from modelmapper.core.neurons import mlps
    model, tok = F1.setup(); res = {}
    g = model.model.norm.weight.float()
    for u in era_neurons():
        l, n = int(u[1:].split(":")[0]), int(u.split(":")[1])
        w = mlps(model)[l].down_proj.weight[:, n].float()
        lg = model.lm_head.weight.float() @ (g * w)
        res[u] = {"top": [tok.decode([i]) for i in lg.topk(15).indices.tolist()], "bottom": [tok.decode([i]) for i in (-lg).topk(15).indices.tolist()]}
    (OUT / "era_readout.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    for u, r in res.items():
        print(u, "top:", r["top"][:10], "| bottom:", r["bottom"][:8])


def causal():
    B = json.loads((OUT / "era_build.json").read_text(encoding="utf-8")); model, tok = F1.setup()
    units = list(era_neurons()) + nulls(model)
    kills = [(int(u[1:].split(":")[0]), int(u.split(":")[1])) for u in units]
    f = open(OUT / "era_causal.jsonl", "w", encoding="utf-8")
    for e, v in B.items():
        if not v["recalled"]:
            continue
        for fm in ("year", "century", "decade"):
            p, a = v["formats"][fm]["prompt"], v["formats"][fm]["answer_ids"]
            if not a:
                continue
            g = [(p, a, None)] + [(p, a, [(l, n, None)]) for l, n in kills]
            lo = F1.logodds_groups(model, tok, [g])[0]
            f.write(json.dumps({"entity": e, "format": fm, "century": v["century"], "answer": v["formats"][fm]["answer"], "base": round(lo[0], 4),
                                "drops": {u: round(lo[0] - x, 4) for u, x in zip(units, lo[1:])}}, ensure_ascii=False) + "\n")
        f.flush()


def _auc(pos, neg):
    if not pos or not neg:
        return None
    s = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return s / (len(pos) * len(neg))


def summary():
    md = lambda xs: stx.median(xs) if len(xs) else float("nan"); EN = era_neurons()
    A = [json.loads(l) for l in (OUT / "era_acts.jsonl").read_text(encoding="utf-8").splitlines()]
    T = [json.loads(l) for l in (OUT / "era_text.jsonl").read_text(encoding="utf-8").splitlines()]
    C = [json.loads(l) for l in (OUT / "era_causal.jsonl").read_text(encoding="utf-8").splitlines()]
    RO = json.loads((OUT / "era_readout.json").read_text(encoding="utf-8"))
    units = list(A[0]["acts"]); nul = [u for u in units if u not in EN]
    cent = Counter(a["century"] for a in A if a["format"] == "year")
    L = [f"ERA NEURONS (era.py): {len(EN)} era neurons + {len(nul)} random controls; {sum(cent.values())} recalled entities, by century {dict(sorted(cent.items()))}"]
    by = defaultdict(lambda: defaultdict(list))
    for a in A:
        for u, x in a["acts"].items():
            by[(u, a["format"])][a["century"]].append(x)
    tb = defaultdict(lambda: defaultdict(list))
    for t in T:
        for u, xs in t["acts_span_then_end"].items():
            tb[(u, t["format"])][t["century"]].append((max(xs), min(xs), xs[-1]))
    rows = []
    for u in units:
        lab = EN.get(u, [19])                     # controls scored against the 19th century (arbitrary fixed label)
        ins = lambda d: [x for c, xs in d.items() if c in lab for x in xs]
        outs = lambda d: [x for c, xs in d.items() if c not in lab for x in xs]
        a_year = _auc(ins(by[(u, "year")]), outs(by[(u, "year")]))
        if a_year is None:
            continue
        sgn = 1 if a_year >= 0.5 else -1
        r = {"u": u, "label": lab, "is_control": u not in EN, "n_in": len(ins(by[(u, "year")])), "sign": sgn}
        for fm in FORMATS:
            p, n = ins(by[(u, fm)]), outs(by[(u, fm)])
            r[fm] = _auc([sgn * x for x in p], [sgn * x for x in n])
        for fm in ("digits", "decade", "ordinal", "word", "hundreds", "named era"):
            d = tb[(u, fm)]
            p = [(x[0] if sgn > 0 else -x[1]) for c, xs in d.items() if c in lab for x in xs]
            n = [(x[0] if sgn > 0 else -x[1]) for c, xs in d.items() if c not in lab for x in xs]
            r["text_" + fm] = _auc(p, n)
        rows.append(r)
    fmt = lambda v: "  -  " if v is None else f"{v:.2f}"
    L.append("\nSELECTIVITY (AUC in-era vs other entities; sign fixed in the YEAR format; question formats at the last prompt position;"
             " text = max over the date span in a plain sentence):")
    L.append(f"   {'neuron':<10} {'era':<10} n_in | year  cent  dec   period ctrl | text: digits decade ord   word  hundr named")
    for r in sorted(rows, key=lambda r: (r["is_control"], r["label"])):
        L.append(f"   {r['u']:<10} {','.join(ordinal(c) for c in r['label']) if not r['is_control'] else 'control':<10} {r['n_in']:>4} | "
                 + " ".join(fmt(r[fm]) for fm in FORMATS) + " | " + " ".join(fmt(r['text_' + fm]) for fm in ("digits", "decade", "ordinal", "word", "hundreds", "named era")))
    era_r = [r for r in rows if not r["is_control"]]; ctl_r = [r for r in rows if r["is_control"]]
    for name, rs in (("era neurons", era_r), ("controls", ctl_r)):
        L.append(f"   median over {name} (n={len(rs)}): " + ", ".join(f"{fm} {md([r[fm] for r in rs if r[fm] is not None]):.2f}" for fm in FORMATS) + " | text "
                 + ", ".join(f"{fm} {md([r['text_' + fm] for r in rs if r['text_' + fm] is not None]):.2f}" for fm in ("digits", "decade", "ordinal", "word", "hundreds", "named era")
                             if [r for r in rs if r['text_' + fm] is not None]))
    # causal
    nd = sorted(abs(c["drops"][u]) for c in C for u in nul); q99 = nd[int(0.99 * len(nd))]
    L.append(f"\nCAUSAL (each neuron zeroed everywhere; drop in log-odds of Qwen's own answer). Random-neuron |drop| 99th pct {q99:.3f}, max {nd[-1]:.3f}")
    L.append(f"   {'neuron':<10} {'era':<10} | year in/out        | century in/out     | decade in/out")
    caus = {}
    for u in EN:
        cells = []
        for fm in ("year", "century", "decade"):
            ci = [c["drops"][u] for c in C if c["format"] == fm and c["century"] in EN[u]]
            co = [c["drops"][u] for c in C if c["format"] == fm and c["century"] not in EN[u]]
            caus[(u, fm)] = (md(ci) if ci else None, md(co) if co else None, len(ci))
            cells.append(f"{fmt(caus[(u, fm)][0])} / {fmt(caus[(u, fm)][1])} (n={len(ci)})")
        L.append(f"   {u:<10} {','.join(ordinal(c) for c in EN[u]):<10} | " + " | ".join(cells))
    # predictions
    K89 = "L31:8930"
    r89 = next((r for r in rows if r["u"] == K89), None)
    b1 = r89 and all((r89[fm] or 0) >= 0.8 for fm in ("year", "century", "decade"))
    b2 = all(caus[("L31:8930", fm)][0] is not None and caus[("L31:8930", fm)][0] >= 1.0 and abs(caus[("L31:8930", fm)][1]) < q99 for fm in ("year", "century", "decade"))
    f2 = [r for r in era_r if r["u"] not in ("L33:12258", "L33:12280") and r["n_in"] >= 5]
    b3n = sum(sum((r[fm] or 0) >= 0.8 for fm in ("year", "century", "decade")) >= 2 for r in f2)
    qa = md([r[fm] for r in era_r for fm in ("year", "century", "decade") if r[fm] is not None])
    ta = md([r["text_" + fm] for r in era_r for fm in ("digits", "decade", "ordinal", "word", "hundreds", "named era") if r["text_" + fm] is not None])
    toks = [t.strip().lower() for t in RO.get("L31:8930", {}).get("top", []) + RO.get("L31:8930", {}).get("bottom", [])]
    L.append("\nPREDICTIONS:")
    L.append(f"   B1 L31:8930 AUC >= 0.8 in year, century, decade: {'HOLDS' if b1 else 'FAILS'} ({', '.join(f'{fm} {fmt(r89[fm])}' for fm in ('year', 'century', 'decade')) if r89 else 'n/a'})")
    L.append(f"   B2 L31:8930 in-era drop >= 1.0 in all three formats, out-of-era < random 99th pct: {'HOLDS' if b2 else 'FAILS'} "
             "(" + "; ".join(f"{fm} {fmt(caus[(K89, fm)][0])}/{fmt(caus[(K89, fm)][1])}" for fm in ("year", "century", "decade")) + ")")
    L.append(f"   B3 facts2 century neurons with >= 5 in-era entities reaching AUC >= 0.8 in >= 2 of 3 formats: {b3n}/{len(f2)} -> {'HOLDS' if f2 and b3n >= len(f2) / 2 else 'FAILS'}")
    L.append(f"   B4 text less selective than answering: median question AUC {qa:.2f} vs text {ta:.2f} -> {'HOLDS' if ta < qa else 'FAILS'}")
    L.append(f"   B5 L31:8930 direct output top/bottom 15 contain era/digit tokens: {RO.get('L31:8930')}")
    L.append("\nDIRECT OUTPUT (top 10 tokens pushed up | pushed down) - informal reading only:")
    for u in sorted(EN, key=lambda u: EN[u]):
        L.append(f"   {u:<10} {','.join(ordinal(c) for c in EN[u]):<10} {RO[u]['top'][:10]} | {RO[u]['bottom'][:6]}")
    txt = "\n".join(L)
    (OUT / "era_rows.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    (OUT / "era_summary.txt").write_text(txt + "\n", encoding="utf-8"); print(txt)


if __name__ == "__main__":
    {"entities": entities, "build": build, "acts": acts, "text": text, "readout": readout, "causal": causal, "summary": summary}[sys.argv[1]]()

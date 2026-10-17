"""Concept atlas: what does each neuron do when Qwen is about to SAY a name? (web read-only; GPU; no Luna)

James (17 Oct): search for concept neurons by asking Qwen, not web text (search_verify.py showed that proposing
concepts from a neuron's web firings finds only 4 of 50 known concept neurons).
Stages (each resumable; everything saved under D:/already-clean-data/concept_atlas/):
  fetch    names = English Wikipedia titles from the earlier samples (wiki_sitelinks.json) known in >= 16 language
           editions, minus disambiguation pages, BAD titles, titles naming 2024+, and any whose intro matches ADULT;
           plus every Wikipedia-titled concept we have mapped. Intros fetched 20 per request.
           (No creation-date filter here, unlike the fame sample: the recall filter drops names Qwen doesn't know.)
  build    cloze prompts as in P1 (p1_prompts.py): each intro sentence that names the title becomes '<sentence with the
           name as ___> Which name fills the blank? Answer with just the name.', up to 4 per name; Qwen's chat format.
           Recall: best answer form, P(answer) >= MIN_P (as map_loop). For every RECALLED prompt, all 442,368 neurons'
           activations at the answer position (last prompt token), fp16, saved per prompt (acts.npy, memmap
           [prompts, 36, 12288]) with an index (index.json).
Usage: python src/concept_atlas.py fetch | build
"""
import glob
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

import httpx
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("AC_DATA_DIR", "D:/already-clean-data")) / "concept_atlas"
MIN_LINKS, PER_REQ, MAX_PROMPTS = 16, 20, 4


def fetch():
    from wiki_sample import ADULT, BAD, RECENT_YEAR, UA
    from wiki_sitelinks import get
    OUT.mkdir(parents=True, exist_ok=True)
    st = json.loads((ROOT / "results/wiki_sitelinks.json").read_text(encoding="utf-8"))
    dis = set(st["disamb"])
    titles = {t for t, q in st["qid"].items() if t not in dis and (st["links"].get(q) or 0) >= MIN_LINKS
              and not BAD.search(t) and not RECENT_YEAR.search(t)}
    mapped = set()
    for f in glob.glob(str(ROOT / "results/concepts_fame*/rank_save_*.json")):
        for c, v in json.loads(Path(f).read_text(encoding="utf-8")).items():
            if "top" in v:
                mapped.add(c)
    titles |= {t for t in mapped if not t.islower() and not t.startswith("the ")}        # Wikipedia-titled mapped concepts
    p = OUT / "intros.json"
    intros = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    todo = sorted(t for t in titles if t not in intros)
    print(f"{len(titles)} names ({len(mapped)} mapped concepts considered); {len(todo)} to fetch", flush=True)
    with httpx.Client(headers=UA, timeout=60) as cl:
        for i in range(0, len(todo), PER_REQ):
            batch = todo[i:i + PER_REQ]
            d = get(cl, "https://en.wikipedia.org/w/api.php", action="query", prop="extracts", exintro=1, explaintext=1,
                    exlimit=PER_REQ, titles="|".join(batch), redirects=1, format="json")
            q = d.get("query", {})
            back = {}
            for k in ("normalized", "redirects"):
                for x in q.get(k, []):
                    back[x["to"]] = back.get(x["from"], x["from"])
            for pg in q.get("pages", {}).values():
                t = pg.get("title"); src = back.get(t, t)
                while src in back and back[src] != src:
                    src = back[src]
                intros[src] = {"title": t, "extract": pg.get("extract", "") or "", "missing": "missing" in pg}
            for t in batch:
                intros.setdefault(t, {"title": t, "extract": "", "missing": True, "note": "not returned"})
            if (i // PER_REQ) % 25 == 0:
                p.write_text(json.dumps(intros, ensure_ascii=False), encoding="utf-8")
                print(f"{min(i + PER_REQ, len(todo))}/{len(todo)}", flush=True)
    adult = [t for t, v in intros.items() if ADULT.search(v["extract"][:600])]
    for t in adult:
        intros[t]["excluded"] = "adult-industry intro"
    p.write_text(json.dumps(intros, ensure_ascii=False), encoding="utf-8")
    print(f"done: {len(intros)} intros; empty {sum(not v['extract'] for v in intros.values())}; excluded (adult) {len(adult)}")


def cloze(title, text):
    name = re.sub(r"\s*\(.*?\)\s*$", "", title).strip()
    out = []
    for s in re.split(r"(?<=[.!?])\s+", text.replace("\n", " ")):
        if name.lower() not in s.lower():
            continue
        cz = re.sub(re.escape(name), "___", s, flags=re.I)
        if name.lower() in cz.lower() or len(cz.split()) < 6:
            continue
        out.append(cz.strip() + " Which name fills the blank? Answer with just the name.")
        if len(out) >= MAX_PROMPTS:
            break
    return name, out


def build():
    import concept_pipeline as cp
    from config import DEFAULT_MODEL
    from neurons import DEVICE, mlps
    from pipeline_fast import FastPipeline
    intros = json.loads((OUT / "intros.json").read_text(encoding="utf-8"))
    pipe = FastPipeline(DEFAULT_MODEL, mode="add", reply_only=True); m, tok = pipe.model, pipe.tok
    L, W = m.config.num_hidden_layers, m.config.intermediate_size
    items = []
    for t, v in sorted(intros.items()):
        if v.get("excluded") or not v["extract"]:
            continue
        name, ps = cloze(t, v["extract"])
        if ps:
            items.append((t, name, ps))
    print(f"{len(items)} names with >= 1 cloze prompt; {sum(len(p) for _, _, p in items)} prompts", flush=True)
    idx_p = OUT / "index.json"
    index = json.loads(idx_p.read_text(encoding="utf-8")) if idx_p.exists() else {"rows": [], "names": {}}
    cap = sum(len(p) for _, _, p in items)
    acts = np.lib.format.open_memmap(OUT / "acts.npy", mode="r+" if (OUT / "acts.npy").exists() else "w+", dtype=np.float16, shape=(cap, L, W))
    row = len(index["rows"]); t0 = time.time()
    for k, (t, name, ps) in enumerate(items):
        if t in index["names"]:
            continue
        chat = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=False) for q in ps]
        base_ = name[4:].strip() if name.lower().startswith("the ") else name
        forms = list(dict.fromkeys([name, name[:1].upper() + name[1:], base_, base_[:1].upper() + base_[1:]]))
        best = None
        for fm in forms:
            ids = tok.encode(fm); pr = pipe.batch_probs([(q, ids) for q in chat])
            sc = sum(x >= cp.MIN_P for x in pr)
            if best is None or sc > best[0]:
                best = (sc, fm, pr, ids)
        sc, fm, pr, ids = best
        rec = [i for i, x in enumerate(pr) if x >= cp.MIN_P]
        entry = {"title": t, "name": name, "form": fm, "answer_ids": ids, "prompts": ps, "probs": [round(x, 4) for x in pr], "rows": []}
        if rec:
            enc = [tok.encode(chat[i]) for i in rec]; T = max(map(len, enc)); pad = tok.pad_token_id or 0
            x = torch.full((len(enc), T), pad, device=DEVICE); mask = torch.zeros_like(x)
            for j, e in enumerate(enc):
                x[j, :len(e)] = torch.tensor(e, device=DEVICE); mask[j, :len(e)] = 1
            last = torch.tensor([len(e) - 1 for e in enc], device=DEVICE); got = {}
            hks = [mlps(m)[l].down_proj.register_forward_pre_hook(
                lambda mod, a, l=l: got.__setitem__(l, a[0][torch.arange(len(enc), device=DEVICE), last].half().cpu())) for l in range(L)]
            try:
                with torch.no_grad():
                    m.model(input_ids=x, attention_mask=mask)
            finally:
                for h in hks:
                    h.remove()
            block = torch.stack([got[l] for l in range(L)], 1).numpy()          # [n, L, W]
            acts[row:row + len(rec)] = block
            for j, i in enumerate(rec):
                index["rows"].append({"name": t, "prompt_i": i}); entry["rows"].append(row + j)
            row += len(rec)
        index["names"][t] = entry
        if k % 200 == 0:
            acts.flush(); idx_p.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
            el = time.time() - t0
            print(f"{k + 1}/{len(items)} names, {row} recalled prompts, {sum(bool(e['rows']) for e in index['names'].values())} names recalled; "
                  f"{el / 60:.1f} min", flush=True)
    acts.flush(); index["n_rows"] = row
    idx_p.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    print(f"done: {row} recalled prompts from {sum(bool(e['rows']) for e in index['names'].values())} names")


def propose():
    """Per neuron: its activation per recalled name (mean over the name's recalled prompts), standardised across names
    (z); its top names by z in EITHER sign. Saved: top-10 names and z per neuron and sign (proposals.npz).
    Check (the step that failed in search_verify): for every mapped concept present in the atlas, is it the top-1 /
    within the top-5 names of its own top-5 neurons? Chance = 1/N and 5/N."""
    from concept_search_feasibility import maps, uid
    index = json.loads((OUT / "index.json").read_text(encoding="utf-8"))
    names = [t for t, e in index["names"].items() if e["rows"]]
    acts = np.load(OUT / "acts.npy", mmap_mode="r")
    N = len(names); L, W = acts.shape[1], acts.shape[2]
    rows_of = [index["names"][t]["rows"] for t in names]
    topn = np.zeros((L * W, 2, 10), np.int32); topz = np.zeros((L * W, 2, 10), np.float32)
    for l in range(L):
        a = torch.tensor(np.asarray(acts[:index["n_rows"], l, :]), dtype=torch.float32, device="cuda")   # [rows, W]
        per = torch.stack([a[r].mean(0) for r in rows_of])                                   # [N, W]
        z = (per - per.mean(0)) / (per.std(0) + 1e-6)
        for s, sg in enumerate((1.0, -1.0)):
            v, i = (sg * z).topk(10, dim=0)                                                 # [10, W]
            topn[l * W:(l + 1) * W, s] = i.T.cpu().numpy(); topz[l * W:(l + 1) * W, s] = v.T.cpu().numpy()
        print(f"propose: layer {l}", flush=True)
    np.savez_compressed(OUT / "proposals.npz", topn=topn, topz=topz, names=np.array(names))
    M = maps(); pos = {t: i for i, t in enumerate(names)}
    hit1 = hit5 = tot = 0
    for c, v in M.items():
        if c not in pos:
            continue
        for u in v["top5"]:
            x = uid(u); tot += 1
            best = [list(topn[x, s, :5]) for s in (0, 1)]
            hit1 += pos[c] in (topn[x, 0, 0], topn[x, 1, 0]); hit5 += any(pos[c] in b for b in best)
    msg = (f"proposals: {N} recalled names. Known concept neurons whose own concept is their top-1 name (either sign): "
           f"{hit1}/{tot} ({hit1 / max(1, tot):.1%}; chance ~{2 / N:.2%}); within top-5: {hit5}/{tot} ({hit5 / max(1, tot):.1%}; chance ~{10 / N:.2%})")
    (OUT / "proposals_check.txt").write_text(msg + "\n", encoding="utf-8"); print(msg)


def verify():
    """Verify proposals causally on INDEPENDENT prompts (Luna-written quiz questions, not the cloze prompts that made the
    proposal). Groups (shuffled; Luna sees only a name):
      top     100 neurons in no concept's top-30 with the highest top-1 z (either sign), one per name (most specific first)
      random  100 neurons in no top-30, layer-matched to 'top', each with its own top-1 name
      known   100 known concept neurons (a mapped top-5 neuron) whose concept is in the atlas, each with its top-1 name
    Per neuron: Luna writes 8 questions whose answer is the name; Qwen answers (best form, P >= MIN_P, leak filter);
    with >= 3 recalled, zero the neuron everywhere: mean drop in exact log-odds; null = 2 random same-layer neurons in no
    top-30. verified = drop >= 1.0 and above the pooled null's 99th percentile.
    Criterion (stated before running): verified rate top >= 3x random, and top >= 20%."""
    import random
    import concept_pipeline as cp
    import oai
    from concept_search_feasibility import maps, uid
    from config import DEFAULT_MODEL
    from neurons import steer
    from pipeline_fast import FastPipeline, leaks
    P = np.load(OUT / "proposals.npz"); topn, topz, names = P["topn"], P["topz"], list(P["names"])
    M = maps(); top30 = {uid(u) for v in M.values() for u in v["top30"]}
    best_s = topz[:, :, 0].argmax(1); z1 = topz[np.arange(len(topz)), best_s, 0]; n1 = topn[np.arange(len(topn)), best_s, 0]
    rng = random.Random(61)
    order = np.argsort(-z1); top, seen = [], set()
    for x in order:
        x = int(x)
        if x not in top30 and n1[x] not in seen:
            top.append(x); seen.add(n1[x])
        if len(top) == 100:
            break
    rnd = []
    for t in top:
        L = t // 12288
        while True:
            x = L * 12288 + rng.randrange(12288)
            if x not in top30 and x not in rnd:
                rnd.append(x); break
    pos = set(names)
    known = sorted({uid(u) for c, v in M.items() if c in pos for u in v["top5"]}); known = rng.sample(known, min(100, len(known)))
    items = [("top", x) for x in top] + [("random", x) for x in rnd] + [("known", x) for x in known]
    rng.shuffle(items)
    vp = OUT / "verify.jsonl"; done = set()
    if vp.exists():
        done = {json.loads(l)["neuron"] for l in vp.read_text(encoding="utf-8").splitlines()}
    qp = OUT / "verify_questions.json"
    Q = json.loads(qp.read_text(encoding="utf-8")) if qp.exists() else {}
    need = sorted({names[n1[x]] for _, x in items} - set(Q))
    SYS = "You write quiz questions. Reply in JSON only."

    def ask(nm):
        for _ in range(3):
            try:
                r = oai.chat_json(f'Write 8 different quiz questions whose correct answer is exactly "{nm}" (as named on English '
                                  'Wikipedia). None may contain the name or an obvious part of it. Reply {"questions": [8 strings]}', SYS)
                return nm, r.get("questions", [])
            except Exception as e:
                err = repr(e)
        return nm, {"error": err}
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(8) as ex:
        for nm, qs in ex.map(ask, need):
            Q[nm] = qs
    qp.write_text(json.dumps(Q, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"questions for {len(Q)} names; Luna {oai.usage}", flush=True)
    index = json.loads((OUT / "index.json").read_text(encoding="utf-8"))
    pipe = FastPipeline(DEFAULT_MODEL, mode="add", reply_only=True); m, tok = pipe.model, pipe.tok
    f = open(vp, "a", encoding="utf-8")
    for grp, x in items:
        u = f"L{x // 12288}:{x % 12288}"
        if u in done:
            continue
        nm = names[n1[x]]; qs = Q.get(nm) if isinstance(Q.get(nm), list) else []
        row = {"neuron": u, "group": grp, "name": nm, "z1": float(z1[x]), "sign": "+" if best_s[x] == 0 else "-"}
        if len(qs) < 3:
            row["status"] = "no_questions"; f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush(); continue
        chat = [tok.apply_chat_template([{"role": "user", "content": q + " Answer with just the name."}], tokenize=False,
                                        add_generation_prompt=True, enable_thinking=False) for q in qs]
        e = index["names"][nm]; a = e["name"]; base_ = a[4:].strip() if a.lower().startswith("the ") else a
        forms = list(dict.fromkeys([e["form"], a, a[:1].upper() + a[1:], base_, base_[:1].upper() + base_[1:]]))
        best = None
        for fm in forms:
            ids = tok.encode(fm); pr = pipe.batch_probs([(q, ids) for q in chat]); sc = sum(p >= cp.MIN_P for p in pr)
            if best is None or sc > best[0]:
                best = (sc, ids, pr, fm)
        _, ids, pr, fm = best
        rec = [q for q, p, raw in zip(chat, pr, qs) if p >= cp.MIN_P and not leaks(raw, a)]
        row.update({"form": fm, "recalled": len(rec)})
        if len(rec) < 3:
            row["status"] = "not_recalled"; f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush(); continue
        pairs = [(q, ids) for q in rec]; base = pipe.batch_logodds(pairs); L, n = x // 12288, x % 12288

        def drop(units):
            with steer(m, [(LL, nn, 0.0) for LL, nn in units]):
                lo = pipe.batch_logodds(pairs)
            return [b_ - l_ for b_, l_ in zip(base, lo)]
        dn = drop([(L, n)]); nulls = []
        while len(nulls) < 2:
            y = rng.randrange(12288)
            if L * 12288 + y not in top30 and y != n:
                nulls.append(y)
        nd = [drop([(L, y)]) for y in nulls]
        row.update({"status": "tested", "base": [round(b_, 4) for b_ in base], "drops": [round(v, 4) for v in dn], "drop_mean": float(np.mean(dn)),
                    "null_neurons": [f"L{L}:{y}" for y in nulls], "null_drop_means": [float(np.mean(v)) for v in nd]})
        f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
        print(f"{grp:<7} {u:<10} z {z1[x]:5.1f} {nm[:40]:<40} recalled {len(rec)} drop {np.mean(dn):+.2f} null {np.mean(nd[0]):+.2f}/{np.mean(nd[1]):+.2f}", flush=True)
    R = [json.loads(l) for l in vp.read_text(encoding="utf-8").splitlines()]
    nulls = [v for r in R if r["status"] == "tested" for v in r["null_drop_means"]]; thr = max(1.0, float(np.percentile(nulls, 99)))
    lines = [f"verify: null 99th percentile {np.percentile(nulls, 99):.3f} (n={len(nulls)}); verified = drop >= {thr:.3f}"]
    rate = {}
    for g in ("top", "random", "known"):
        rs = [r for r in R if r["group"] == g]; ver = [r for r in rs if r.get("drop_mean", -9) >= thr]
        rate[g] = len(ver) / max(1, len(rs))
        lines.append(f"  {g:<7} n={len(rs)} tested {sum(r['status'] == 'tested' for r in rs)} verified {len(ver)} ({rate[g]:.0%}): "
                     + ", ".join(f"{r['name']} ({r['neuron']}, {r['drop_mean']:+.1f})" for r in sorted(ver, key=lambda r: -r['drop_mean'])[:12]))
    ok = rate["top"] >= 3 * rate["random"] and rate["top"] >= 0.2
    lines.append(f"CRITERION {'MET' if ok else 'NOT MET'} (top >= 3x random and >= 20%)")
    (OUT / "verify_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8"); print("\n".join(lines))


def verify_items():
    """The same selection as verify(): (group, neuron id, proposed name, z1, sign) - top / random / known, shuffled."""
    import random
    from concept_search_feasibility import maps, uid
    P = np.load(OUT / "proposals.npz"); topn, topz, names = P["topn"], P["topz"], list(P["names"])
    M = maps(); top30 = {uid(u) for v in M.values() for u in v["top30"]}
    best_s = topz[:, :, 0].argmax(1); z1 = topz[np.arange(len(topz)), best_s, 0]; n1 = topn[np.arange(len(topn)), best_s, 0]
    rng = random.Random(61)
    order = np.argsort(-z1); top, seen = [], set()
    for x in order:
        x = int(x)
        if x not in top30 and n1[x] not in seen:
            top.append(x); seen.add(n1[x])
        if len(top) == 100:
            break
    rnd = []
    for t in top:
        L = t // 12288
        while True:
            x = L * 12288 + rng.randrange(12288)
            if x not in top30 and x not in rnd:
                rnd.append(x); break
    pos = set(names)
    known = sorted({uid(u) for c, v in M.items() if c in pos for u in v["top5"]}); known = rng.sample(known, min(100, len(known)))
    items = [("top", x) for x in top] + [("random", x) for x in rnd] + [("known", x) for x in known]
    rng.shuffle(items)
    return [(g, x, names[n1[x]], float(z1[x]), "+" if best_s[x] == 0 else "-") for g, x in items], top30


def rows_logodds(model, tok, rows, chunk_rows=128):
    """rows: list of groups; a group = list of (prompt_ids, answer_ids, kill) sharing ONE pass (its as-is rows and its
    knockout rows), kill = (layer, neuron) zeroed at every position, or None. Groups are packed into passes of up to
    chunk_rows sequences, never split. Returns, per group, the exact log-odds of the whole answer for every row."""
    from neurons import DEVICE, mlps
    out = [None] * len(rows); i = 0
    while i < len(rows):
        pack = [i]; n = len(rows[i]); i += 1
        while i < len(rows) and n + len(rows[i]) <= chunk_rows:
            pack.append(i); n += len(rows[i]); i += 1
        flat = [(gi, r) for gi in pack for r in rows[gi]]
        seqs = [p + a[:-1] for _, (p, a, _) in flat]; T = max(map(len, seqs)); pad = tok.pad_token_id or 0
        x = torch.full((len(seqs), T), pad, device=DEVICE); mask = torch.zeros_like(x)
        for j, s in enumerate(seqs):
            x[j, :len(s)] = torch.tensor(s, device=DEVICE); mask[j, :len(s)] = 1
        by_layer = {}
        for j, (_, (_, _, k)) in enumerate(flat):
            if k is not None:
                by_layer.setdefault(k[0], []).append((j, k[1]))
        hks = []
        for L, lst in by_layer.items():
            rr = torch.tensor([j for j, _ in lst], device=DEVICE); nn = torch.tensor([n_ for _, n_ in lst], device=DEVICE)

            def h(mod, a, rr=rr, nn=nn):
                y = a[0].clone(); y[rr, :, nn] = 0.0; return (y,)
            hks.append(mlps(model)[L].down_proj.register_forward_pre_hook(h))
        try:
            with torch.no_grad():
                hs = model.model(input_ids=x, attention_mask=mask).last_hidden_state
        finally:
            for h in hks:
                h.remove()
        res = {}
        for j, (gi, (p, a, _)) in enumerate(flat):
            pos = torch.arange(len(p) - 1, len(p) - 1 + len(a), device=DEVICE)
            lp = model.lm_head(hs[j, pos]).double().log_softmax(-1)
            s = lp[torch.arange(len(a), device=DEVICE), torch.tensor(a, device=DEVICE)].sum().item()
            s = min(s, -1e-15)
            res.setdefault(gi, []).append(s - float(np.log(-np.expm1(s))))
        for gi in pack:
            out[gi] = res[gi]
    return out


def verify_fast():
    """verify() rebuilt with the loop's lessons (17 Oct): Luna writes questions in background threads while the GPU
    works on names as soon as their questions arrive (append-only saving); recall for all answer forms x questions of
    up to NB names in one batched pass; every knockout measured against an as-is copy in the SAME pass (bf16 lesson).
    Same groups, selection, criteria and nulls as verify(). Timing: GPU busy vs waiting on Luna. Gate: compare with the
    unbatched path on a sample (verify_gate) before reading results.
    Output: verify_fast.jsonl, verify_questions.jsonl, verify_fast_summary.txt."""
    import random
    import concept_pipeline as cp
    import oai
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from config import DEFAULT_MODEL
    from pipeline_fast import FastPipeline, leaks
    NB = int(os.environ.get("NB", "8")); limit = int(os.environ.get("VF_LIMIT", "0"))
    items, top30 = verify_items()
    if limit:
        items = items[:limit]
    tag = os.environ.get("VF_TAG", "verify_fast")
    vp, qp = OUT / f"{tag}.jsonl", OUT / "verify_questions.jsonl"
    done = {json.loads(l)["neuron"] for l in vp.read_text(encoding="utf-8").splitlines()} if vp.exists() else set()
    Q = {}
    if qp.exists():
        for l in qp.read_text(encoding="utf-8").splitlines():
            r = json.loads(l); Q[r["name"]] = r["questions"]
    by_name = {}
    for it in items:
        if f"L{it[1] // 12288}:{it[1] % 12288}" not in done:
            by_name.setdefault(it[2], []).append(it)
    index = json.loads((OUT / "index.json").read_text(encoding="utf-8"))
    pipe = FastPipeline(DEFAULT_MODEL, mode="add", reply_only=True); m, tok = pipe.model, pipe.tok
    SYS = "You write quiz questions. Reply in JSON only."; qlock = threading.Lock(); qf = open(qp, "a", encoding="utf-8")

    def ask(nm):
        if nm in Q:
            return nm, Q[nm]
        qs, err = None, None
        for _ in range(3):
            try:
                qs = oai.chat_json(f'Write 8 different quiz questions whose correct answer is exactly "{nm}" (as named on English '
                                   'Wikipedia). None may contain the name or an obvious part of it. Reply {"questions": [8 strings]}', SYS).get("questions", [])
                break
            except Exception as e:                                       # recorded, never dropped silently
                err = repr(e)
        rec = {"name": nm, "questions": qs if qs is not None else {"error": err}}
        with qlock:
            qf.write(json.dumps(rec, ensure_ascii=False) + "\n"); qf.flush()
        return nm, rec["questions"]
    rng = random.Random(62); f = open(vp, "a", encoding="utf-8")
    t_start = time.time(); gpu_busy = 0.0; buf = []

    def chat(q):
        return tok.encode(tok.apply_chat_template([{"role": "user", "content": q + " Answer with just the name."}], tokenize=False,
                                                  add_generation_prompt=True, enable_thinking=False))

    def process(batch):
        nonlocal gpu_busy
        t0 = time.time()
        prep = []
        for nm, qs in batch:
            qs = qs if isinstance(qs, list) else []
            e = index["names"][nm]; a = e["name"]; base_ = a[4:].strip() if a.lower().startswith("the ") else a
            forms = list(dict.fromkeys([e["form"], a, a[:1].upper() + a[1:], base_, base_[:1].upper() + base_[1:]]))
            prep.append((nm, qs, a, forms, [chat(q) for q in qs]))
        groups, keys = [], []
        for pi, (nm, qs, a, forms, ch) in enumerate(prep):
            for fm in forms:
                ids = tok.encode(fm); groups.append([(c, ids, None) for c in ch]); keys.append((pi, fm, ids))
        lo = rows_logodds(m, tok, [g for g in groups if g]) if any(groups) else []
        it = iter(lo); best = {}
        for g, (pi, fm, ids) in zip(groups, keys):
            if not g:
                continue
            v = next(it); pr = [1 / (1 + np.exp(-x)) for x in v]; sc = sum(p >= cp.MIN_P for p in pr)
            if pi not in best or sc > best[pi][0]:
                best[pi] = (sc, fm, ids, pr)
        kgroups, kmeta = [], []
        for pi, (nm, qs, a, forms, ch) in enumerate(prep):
            sc, fm, ids, pr = best.get(pi, (0, None, None, []))
            rec_i = [i for i, (q, p) in enumerate(zip(qs, pr)) if p >= cp.MIN_P and not leaks(q, a)]
            for grp, x, _, z1, sg in by_name[nm]:
                u = f"L{x // 12288}:{x % 12288}"
                row = {"neuron": u, "group": grp, "name": nm, "z1": z1, "sign": sg, "form": fm, "recalled": len(rec_i),
                       "recalled_i": rec_i, "probs": [round(float(p), 4) for p in pr]}                 # provenance
                if len(qs) < 3:
                    row["status"] = "no_questions"; f.write(json.dumps(row, ensure_ascii=False) + "\n"); continue
                if len(rec_i) < 3:
                    row["status"] = "not_recalled"; f.write(json.dumps(row, ensure_ascii=False) + "\n"); continue
                L, n = x // 12288, x % 12288; nulls = []
                while len(nulls) < 2:
                    y = rng.randrange(12288)
                    if L * 12288 + y not in top30 and y != n:
                        nulls.append(y)
                g = []
                for kill in (None, (L, n), (L, nulls[0]), (L, nulls[1])):
                    g += [(ch[i], ids, kill) for i in rec_i]
                kgroups.append(g); kmeta.append((row, len(rec_i), nulls, L))
        for (row, k, nulls, L), v in zip(kmeta, rows_logodds(m, tok, kgroups) if kgroups else []):
            b, z, n0, n1_ = v[:k], v[k:2 * k], v[2 * k:3 * k], v[3 * k:]
            dn = [x - y for x, y in zip(b, z)]
            row.update({"status": "tested", "base": [round(x, 4) for x in b], "drops": [round(x, 4) for x in dn], "drop_mean": float(np.mean(dn)),
                        "null_neurons": [f"L{L}:{y}" for y in nulls],
                        "null_drop_means": [float(np.mean([x - y for x, y in zip(b, n0)])), float(np.mean([x - y for x, y in zip(b, n1_)]))]})
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush(); gpu_busy += time.time() - t0
    with ThreadPoolExecutor(8) as ex:
        futs = [ex.submit(ask, nm) for nm in by_name]
        for fu in as_completed(futs):
            buf.append(fu.result())
            if len(buf) >= NB:
                process(buf); buf = []
        if buf:
            process(buf)
    el = time.time() - t_start
    R = [json.loads(l) for l in vp.read_text(encoding="utf-8").splitlines()]
    nulls = [v for r in R if r["status"] == "tested" for v in r["null_drop_means"]]
    thr = max(1.0, float(np.percentile(nulls, 99))) if nulls else 1.0
    lines = [f"{tag}: {len(R)} neurons; wall {el / 60:.1f} min, GPU busy {gpu_busy / 60:.1f} min ({gpu_busy / max(el, 1e-9):.0%}); Luna {oai.usage}",
             f"null 99th percentile {np.percentile(nulls, 99) if nulls else float('nan'):.3f} (n={len(nulls)}); verified = drop >= {thr:.3f}"]
    rate = {}
    for g in ("top", "random", "known"):
        rs = [r for r in R if r["group"] == g]; ver = [r for r in rs if r.get("drop_mean", -9) >= thr]
        rate[g] = len(ver) / max(1, len(rs))
        lines.append(f"  {g:<7} n={len(rs)} tested {sum(r['status'] == 'tested' for r in rs)} verified {len(ver)} ({rate[g]:.0%}): "
                     + ", ".join(f"{r['name']} ({r['neuron']}, {r['drop_mean']:+.1f})" for r in sorted(ver, key=lambda r: -r['drop_mean'])[:15]))
    ok = rate.get("top", 0) >= 3 * rate.get("random", 0) and rate.get("top", 0) >= 0.2
    lines.append(f"CRITERION {'MET' if ok else 'NOT MET'} (top >= 3x random and >= 20%)")
    (OUT / f"{tag}_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8"); print("\n".join(lines))


def verify_gate():
    """Batched path (rows_logodds, in-pass baseline, many names per pass) vs the unbatched path (pipe.batch_logodds +
    steer, separate passes, as in verify()) on the neurons verify_fast already tested (VF_TAG file), first 30.
    Rule (stated before running): drops agree (Spearman >= 0.95) and the verified / not-verified call is the same for
    >= 95% of neurons (threshold 1.0)."""
    from config import DEFAULT_MODEL
    from neurons import steer
    from pipeline_fast import FastPipeline
    tag = os.environ.get("VF_TAG", "verify_fast")
    R = [json.loads(l) for l in (OUT / f"{tag}.jsonl").read_text(encoding="utf-8").splitlines()]
    R = [r for r in R if r["status"] == "tested"][:30]
    Q = {}
    for l in (OUT / "verify_questions.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(l); Q[r["name"]] = r["questions"]
    index = json.loads((OUT / "index.json").read_text(encoding="utf-8"))
    pipe = FastPipeline(DEFAULT_MODEL, mode="add", reply_only=True); m, tok = pipe.model, pipe.tok
    a_, b_ = [], []
    for r in R:
        L, n = (int(x) for x in r["neuron"][1:].split(":"))
        ids = tok.encode(r["form"]); qs = Q[r["name"]]
        import concept_pipeline as cp
        from pipeline_fast import leaks
        chat = [tok.apply_chat_template([{"role": "user", "content": q + " Answer with just the name."}], tokenize=False,
                                        add_generation_prompt=True, enable_thinking=False) for q in qs]
        pr = pipe.batch_probs([(q, ids) for q in chat])
        if "recalled_i" in r:                       # same question set as the batched run: isolates the knockout measurement
            rec = [chat[i] for i in r["recalled_i"]]
        else:
            rec = [c for c, q, p in zip(chat, qs, pr) if p >= cp.MIN_P and not leaks(q, index["names"][r["name"]]["name"])]
        pairs = [(c, ids) for c in rec]; base = pipe.batch_logodds(pairs)
        with steer(m, [(L, n, 0.0)]):
            lo = pipe.batch_logodds(pairs)
        a_.append(r["drop_mean"]); b_.append(float(np.mean([x - y for x, y in zip(base, lo)])) if rec else float("nan"))
    from fame_summary import spearman
    ok = [(x, y) for x, y in zip(a_, b_) if y == y]
    rho = spearman([x for x, _ in ok], [y for _, y in ok]); same = sum((x >= 1.0) == (y >= 1.0) for x, y in ok)
    msg = (f"verify gate ({len(ok)} neurons): Spearman(batched, unbatched drop) {rho:.3f}; same verified call {same}/{len(ok)}; "
           f"max |diff| {max(abs(x - y) for x, y in ok):.3f}\n" + "\n".join(f"  {r['neuron']:<10} {r['name'][:30]:<30} batched {x:+.3f} unbatched {y:+.3f}"
                                                                         for r, x, y in zip(R, a_, b_)))
    (OUT / f"{tag}_gate.txt").write_text(msg + "\n", encoding="utf-8"); print(msg)


if __name__ == "__main__":
    {"fetch": fetch, "build": build, "propose": propose, "verify": verify, "verify_fast": verify_fast, "verify_gate": verify_gate}[sys.argv[1]]()

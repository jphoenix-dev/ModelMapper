"""Search-then-verify pilot: of neurons that cheap profiles rank as concept-like but that appear in NO map, how many can
be given a concept and VERIFIED causally? (GPU; Luna proposes, Qwen verifies)

James (17 Oct): grind the right thing. concept_search_feasibility.py ranks all 442k neurons; this measures what the
ranking is worth after verification - the hit rate that decides whether search-then-verify is the loop's core.
Groups (Luna sees them shuffled, unlabelled):
  top      100 drawn at random from the top 1% of all neurons by score (scores.npz) that are in no concept's top-30
  random   100 layer-matched neurons in no top-30 (one per 'top' neuron, same layer)
  size     100 size-matched neurons in no top-30 (same layer, closest corpus peak = max(stats max, -min))
  known    50 positive controls drawn from the NEW coverage-4 top-5 neurons (in no earlier top-30) - how often the
           procedure re-finds a known concept neuron (its recall, roughly)
Per neuron:
  1 Luna reads its 15 strongest positive and 15 strongest negative firings in web text (atlas, +-12 tokens, firing
    token in [[ ]]) and the 20 tokens its weights push most each way, and either names ONE specific concept (with the
    exact answer and 8 questions whose answer is that name) or says none.
  2 Qwen answers the questions (chat format, 'Answer with just the name.', best answer form, P >= MIN_P, leak filter,
    as in map_loop). Needs >= 3 recalled questions (map_loop needs 5 of ~10; stated).
  3 knock-out (zero at every position): mean drop in exact log-odds over the recalled questions, for the neuron and for
    2 random same-layer neurons in no top-30 (null).
  verified = drop >= 1.0 AND above the 99th percentile of all null drops (pooled)
Criterion (stated before running): verified rate top >= 3x random. Also reported: size-matched, known, Luna's 'none'
rate per group, and the verified neurons with their concepts.
Output: D:/already-clean-data/search_verify/ (luna.json, verify.jsonl, summary.txt). Usage: python src/search_verify.py
"""
import glob
import json
import os
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

import concept_pipeline as cp
import oai
from concept_search_feasibility import maps, uid
from config import DEFAULT_MODEL
from neurons import steer
from pipeline_fast import FastPipeline, leaks

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("AC_DATA_DIR", "D:/already-clean-data"))
OUT = Path(os.environ.get("SV_OUT", str(DATA / "search_verify")))       # SV_OUT / SV_LIMIT: trial runs
CTX, HALF, NSNIP, NPUSH, MIN_REC = 1024, 12, 15, 20, 3
SYSTEM = ("You study individual neurons inside a language model. You see where one neuron fires most strongly in web text "
          "and which output tokens it pushes. Decide whether it is about ONE specific, nameable concept - a particular "
          "person, place, organisation, work, event, object, species, product or idea that has a name - and not a general "
          "pattern (grammar, punctuation, a letter, a broad topic, a word fragment). Most neurons are NOT about one specific "
          "concept; say so when that is the case. Reply in JSON only.")


def nid(x):
    return f"L{x // 12288}:{x % 12288}"


def select():
    M = maps(); top30_all = {uid(u) for v in M.values() for u in v["top30"]}
    pre30 = {uid(u) for v in M.values() if v["set"] != "concepts_fame5" for u in v["top30"]}
    pos_new = sorted({uid(u) for v in M.values() if v["set"] == "concepts_fame5" for u in v["top5"]} - pre30)
    S = np.load(DATA / "concept_search/scores.npz"); score = S["score"]
    F = np.load(DATA / "concept_search/features.npz"); names = list(F["names"]); X = F["X"]
    size = np.maximum(X[:, names.index("stats_log_max")], X[:, names.index("stats_log_neg_min")])
    rng = random.Random(51)
    order = np.argsort(-score); top1 = [int(i) for i in order[: len(score) // 100] if int(i) not in top30_all]
    top = rng.sample(top1, 100); used = set(top)
    rnd, siz = [], []
    for t in top:
        L = t // 12288
        while True:
            x = L * 12288 + rng.randrange(12288)
            if x not in top30_all and x not in used:
                rnd.append(x); used.add(x); break
        ids = np.arange(L * 12288, (L + 1) * 12288); o = ids[np.argsort(np.abs(size[ids] - size[t]))]
        for x in o:
            x = int(x)
            if x not in top30_all and x not in used:
                siz.append(x); used.add(x); break
    known = rng.sample(pos_new, 50)
    return [("top", x) for x in top] + [("random", x) for x in rnd] + [("size", x) for x in siz] + [("known", x) for x in known], top30_all


def snippets(A, toks, starts, lens, tok, x, side):
    L, n = x // 12288, x % 12288
    out = []
    for c in A[f"{side}_c"][L, n, :NSNIP].tolist():
        d, p = c // CTX, c % CTX; g = starts[d] + p; lo = max(starts[d], g - HALF); hi = min(starts[d] + lens[d], g + HALF + 1)
        out.append((tok.decode(toks[lo:g]) + "[[" + tok.decode(toks[g:g + 1]) + "]]" + tok.decode(toks[g + 1:hi])).replace("\n", " "))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    items, top30_all = select()
    order = items[:]; random.Random(52).shuffle(order)
    if int(os.environ.get("SV_LIMIT", "0")):
        order = order[: int(os.environ["SV_LIMIT"])]
    pipe = FastPipeline(DEFAULT_MODEL, mode="add", reply_only=True); m, tok = pipe.model, pipe.tok
    A = torch.load(ROOT / f"data/atlas_{DEFAULT_MODEL.split('/')[-1]}.pt", map_location="cpu", weights_only=False)
    d = np.load(ROOT / f"data/atlas_{DEFAULT_MODEL.split('/')[-1]}_docs.npz")
    toks, lens = d["tokens"].astype(np.int64), d["lengths"]; starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    WU = m.lm_head.weight.detach(); g = m.model.norm.weight.detach().float()
    # ---- Luna proposals (resumable) ----
    lp = OUT / "luna.json"
    luna = json.loads(lp.read_text(encoding="utf-8")) if lp.exists() else {}
    jobs = []
    for grp, x in order:
        if nid(x) in luna:
            continue
        L, n = x // 12288, x % 12288
        col = (m.model.layers[L].mlp.down_proj.weight[:, n].detach().float() * g).to(WU.dtype)
        lg = (WU @ col).float()
        up = [tok.decode([i]) for i in lg.topk(NPUSH).indices.tolist()]; dn = [tok.decode([i]) for i in (-lg).topk(NPUSH).indices.tolist()]
        sp, sn = snippets(A, toks, starts, lens, tok, x, "top"), snippets(A, toks, starts, lens, tok, x, "bot")
        prompt = ("Strongest POSITIVE firings (the token in [[ ]] is where it fired):\n" + "\n".join(f"{i + 1}. {s}" for i, s in enumerate(sp))
                  + "\n\nStrongest NEGATIVE firings:\n" + "\n".join(f"{i + 1}. {s}" for i, s in enumerate(sn))
                  + f"\n\nTokens it pushes UP when positive: {json.dumps(up, ensure_ascii=False)}"
                  + f"\nTokens it pushes UP when negative: {json.dumps(dn, ensure_ascii=False)}"
                  + '\n\nIf it is about one specific nameable concept, reply {"concept": "<name>", "answer": "<the exact short name a person '
                    'would answer with>", "questions": [8 different quiz questions whose correct answer is exactly that name, '
                    'none containing the name], "evidence": "<a few words>"}. Otherwise reply {"concept": null, "evidence": "<what it '
                    'seems to respond to instead, a few words>"}.')
        jobs.append((x, grp, prompt))
    lock = threading.Lock()

    def ask(job):
        x, grp, prompt = job
        r = None
        for _ in range(3):
            try:
                r = oai.chat_json(prompt, SYSTEM); break
            except Exception as e:                                   # kept, never dropped silently
                r = {"error": repr(e)}
        with lock:
            luna[nid(x)] = {"group": grp, "reply": r, "prompt": prompt}
            if len(luna) % 25 == 0:
                lp.write_text(json.dumps(luna, indent=1, ensure_ascii=False), encoding="utf-8")
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(ask, jobs))
    lp.write_text(json.dumps(luna, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Luna: {len(luna)} neurons; usage {oai.usage}", flush=True)
    # ---- Qwen verification (resumable) ----
    vp = OUT / "verify.jsonl"; done = set()
    if vp.exists():
        done = {json.loads(l)["neuron"] for l in vp.read_text(encoding="utf-8").splitlines()}
    rng = random.Random(53); f = open(vp, "a", encoding="utf-8")
    for grp, x in order:
        u = nid(x)
        if u in done:
            continue
        r = luna[u]["reply"] or {}; row = {"neuron": u, "group": grp, "concept": r.get("concept")}
        qs = r.get("questions") or []
        if not r.get("concept") or not r.get("answer") or len(qs) < 3:
            row["status"] = "no_concept" if not r.get("concept") else "bad_reply"
            f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush(); continue
        a = str(r["answer"]).strip()
        chat = [tok.apply_chat_template([{"role": "user", "content": q + " Answer with just the name."}], tokenize=False,
                                        add_generation_prompt=True, enable_thinking=False) for q in qs]
        base_ = a[4:].strip() if a.lower().startswith("the ") else a
        forms = list(dict.fromkeys([a, a[:1].upper() + a[1:], a.title(), base_, base_[:1].upper() + base_[1:]]))
        best = None
        for fm in forms:
            t_ = tok.encode(fm); ps = pipe.batch_probs([(q, t_) for q in chat])
            sc = sum(p >= cp.MIN_P for p in ps)
            if best is None or sc > best[0]:
                best = (sc, t_, ps, fm)
        _, tid, ps, form = best
        rec = [q for q, p, raw in zip(chat, ps, qs) if p >= cp.MIN_P and not leaks(raw, a)]
        row.update({"answer": a, "form": form, "recalled": len(rec), "probs": [round(p, 4) for p in ps]})
        if len(rec) < MIN_REC:
            row["status"] = "not_recalled"; f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush(); continue
        pairs = [(q, tid) for q in rec]; base = pipe.batch_logodds(pairs)
        L, n = x // 12288, x % 12288

        def drop(units):
            with steer(m, [(LL, nn, 0.0) for LL, nn in units]):
                lo = pipe.batch_logodds(pairs)
            return [b_ - l_ for b_, l_ in zip(base, lo)]
        dn = drop([(L, n)])
        nulls = []
        while len(nulls) < 2:
            y = rng.randrange(12288)
            if L * 12288 + y not in top30_all and y != n:
                nulls.append(y)
        nd = [drop([(L, y)]) for y in nulls]
        row.update({"status": "tested", "base": [round(b_, 4) for b_ in base], "drops": [round(v, 4) for v in dn],
                    "drop_mean": float(np.mean(dn)), "null_neurons": [f"L{L}:{y}" for y in nulls],
                    "null_drop_means": [float(np.mean(v)) for v in nd]})
        f.write(json.dumps(row, ensure_ascii=False) + "\n"); f.flush()
        print(f"{grp:<7} {u:<10} {str(r.get('concept'))[:40]:<40} recalled {len(rec)} drop {np.mean(dn):+.2f} null {np.mean(nd[0]):+.2f}/{np.mean(nd[1]):+.2f}", flush=True)
    print("verification done", flush=True)


if __name__ == "__main__":
    main()

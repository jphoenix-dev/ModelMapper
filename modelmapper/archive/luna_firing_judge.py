"""Luna reads samples of concept neurons' firings in web text: own concept, broader ingredient, or unrelated?
(step 3b of the coding test; costs Luna tokens, nothing else)

v1 sample (superseded by v2 below, kept for the record): 300 concept neurons from corpus_profile (seed 21) - all 32 in layers 16-23, 70 from
0-15, 90 from 24-29, 108 from 30-35 - plus 100 CONTROLS: the paired random same-layer neuron of 100 of those 300,
judged against the SAME concept name (Luna's leniency baseline). For each neuron, 12 RANDOM firings (not the top
ones; seed 21) with +-20 tokens of context, the firing token marked [[ ]]. Luna labels each context:
  A  about the concept itself;  B  related: the same topic, or a broader category/ingredient of it;  C  unrelated.
It is not told whether a neuron is a control.
Budget: v1 assumed a pessimistic $2.50 / $10 per M tokens; v2 uses Luna's real $0.10 / $0.50 (James) and the run stops (logged, not silent) once the estimate reaches BUDGET_USD = 2.50.
v2: 6 requests in parallel (the shared token tally may undercount slightly under contention; the budget
has a ~5x margin over the expected spend). Resumable.
Usage: python src/luna_firing_judge.py   -> results/luna_firing_judge.json (+ .log)
"""
import json
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

import oai
from config import DEFAULT_MODEL, model_path

ROOT = Path(__file__).resolve().parents[1]
PRICE_IN, PRICE_OUT, BUDGET_USD = 0.10e-6, 0.50e-6, 2.50     # Luna's real price (James, 16 Oct); cached input counted at full price
N_CTX, HALF = 12, 20
PLAN = {(0, 16): 70, (16, 24): 32, (24, 30): 90, (30, 36): 108}
SYSTEM = ("You judge short web-text snippets. A language-model neuron fired on the token marked [[like this]]. "
          "For each snippet decide how it relates to the given concept: "
          "A = the snippet is about the concept itself (it is mentioned or clearly discussed); "
          "B = related but not the concept itself: the same topic or field, a closely related thing, or a broader "
          "category the concept belongs to; C = unrelated to the concept. "
          'Reply as JSON: {"labels": ["A"|"B"|"C", ... one per snippet, in order], '
          '"theme_of_unrelated": "<a few words on what the C snippets have in common, or \'none\'>"}')


def cost():
    return oai.usage["input_tokens"] * PRICE_IN + oai.usage["output_tokens"] * PRICE_OUT


def main():
    rows = json.loads((ROOT / "results/corpus_profile_units.json").read_text(encoding="utf-8"))
    prof = json.loads((ROOT / "results/corpus_profile_summary.json").read_text(encoding="utf-8"))
    rng = random.Random(21)
    concept_rows = [i for i, r in enumerate(rows) if not r["control"] and r["neuron"] in prof and prof[r["neuron"]]["n"] >= N_CTX]
    pair = {r["paired_with"]: i for i, r in enumerate(rows) if r["control"]}
    # v2 design (after James gave Luna's real price, ~20x below the assumed one): EVERY profiled concept neuron with
    # >= N_CTX firings, each with its paired random control judged against the same concept name. Shuffled (seed 21).
    jobs = [(i, i) for i in concept_rows] + [(pair[rows[i]["neuron"]], i) for i in concept_rows]
    random.Random(21).shuffle(jobs)
    z = np.load(ROOT / "results/corpus_profile.npz"); unit, doc, pos = z["unit"], z["doc"], z["pos"].astype(np.int64)
    order = np.argsort(unit, kind="stable"); unit, doc, pos = unit[order], doc[order], pos[order]
    bounds = np.searchsorted(unit, np.arange(len(rows) + 1))
    d = np.load(ROOT / f"data/atlas_{DEFAULT_MODEL.split('/')[-1]}_docs.npz")
    toks, lens = d["tokens"], d["lengths"]; starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    tok = AutoTokenizer.from_pretrained(model_path(DEFAULT_MODEL))
    out_p = ROOT / "results/luna_firing_judge.json"
    out = json.loads(out_p.read_text(encoding="utf-8")) if out_p.exists() else {"jobs": {}, "price_assumed": [PRICE_IN, PRICE_OUT]}
    u0 = out.get("usage", {"input_tokens": 0, "output_tokens": 0})          # earlier runs' tokens, re-priced
    spent_before = u0["input_tokens"] * PRICE_IN + u0["output_tokens"] * PRICE_OUT
    base_usage = dict(u0)
    log = open(ROOT / "results/luna_firing_judge.log", "a", encoding="utf-8")
    lock = threading.Lock(); stop = threading.Event()

    def one(job):
        show, name = job
        key = f"{rows[show]['neuron']}|{'control' if rows[show]['control'] else 'concept'}|{rows[name]['neuron']}"
        if key in out["jobs"] or stop.is_set():
            return
        with lock:
            if spent_before + cost() >= BUDGET_USD:
                msg = f"STOPPED at the budget: estimated ${spent_before + cost():.2f} >= ${BUDGET_USD}; {len(out['jobs'])}/{len(jobs)} judged"
                print(msg, flush=True); log.write(msg + "\n"); stop.set(); return
        sel = np.arange(bounds[show], bounds[show + 1])
        if len(sel) < N_CTX:
            with lock:
                out["jobs"][key] = {"skipped": f"only {len(sel)} firings", "control": rows[show]["control"]}
            return
        jrng = random.Random(f"21|{key}")
        ch = jrng.sample(list(sel), N_CTX)
        snippets = []
        for k in ch:
            g = starts[doc[k]] + pos[k]; lo = max(starts[doc[k]], g - HALF); hi = min(starts[doc[k]] + lens[doc[k]], g + HALF + 1)
            snippets.append((tok.decode(toks[lo:g]) + "[[" + tok.decode(toks[g:g + 1]) + "]]" + tok.decode(toks[g + 1:hi])).replace("\n", " "))
        concept = " / ".join(rows[name]["concepts"])
        prompt = f"Concept: {concept}\n\nSnippets:\n" + "\n".join(f"{j + 1}. {s_}" for j, s_ in enumerate(snippets))
        try:
            r = oai.chat_json(prompt, SYSTEM)
            labels = [str(x).strip().upper()[:1] for x in r.get("labels", [])]
            ok = len(labels) == N_CTX and all(x in "ABC" for x in labels)
        except Exception as e:                                                     # recorded, never dropped silently
            r, labels, ok = {"error": repr(e)}, [], False
        with lock:
            out["jobs"][key] = {"concept": concept, "control": rows[show]["control"], "layer": rows[show]["layer"],
                                "labels": labels, "valid": ok, "theme_of_unrelated": r.get("theme_of_unrelated"),
                                "snippets": snippets, "raw": None if ok else r}
            out["estimated_usd"] = round(spent_before + cost(), 4)
            out["usage"] = {k: base_usage.get(k, 0) + oai.usage.get(k, 0) for k in ("calls", "input_tokens", "output_tokens")}
            line = f"{key:<40} {''.join(labels) if ok else 'INVALID'} | est ${out['estimated_usd']:.3f}"
            print(line, flush=True); log.write(line + "\n"); log.flush()
            if len(out["jobs"]) % 20 == 0:
                out_p.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    with ThreadPoolExecutor(6) as ex:
        list(ex.map(one, jobs))
    out_p.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"done: {len(out['jobs'])} judged; estimated ${out.get('estimated_usd', 0):.3f} at the assumed price", flush=True)


if __name__ == "__main__":
    main()

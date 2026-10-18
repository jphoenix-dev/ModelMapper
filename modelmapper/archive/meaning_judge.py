"""How many meanings does a concept neuron have, and how related are they? Blind grouping by Luna (stage A of the
meaning / Ur-concept test, after James's question; no GPU).

Units (seed 31), all with >= 30 strong firings in the 10M-token corpus profile:
  concept   200 concept neurons, by layer band 0-15: 50, 16-23: 20, 24-29: 60, 30-35: 70
  control   60 random same-layer neurons (corpus_profile pairs)
  spliced   60 fake two-meaning units: 15 firings from each of two concept neurons owned by different concept families,
            shuffled together - KNOWN to have (at least) two unrelated meanings, for calibration
For each unit, 30 random firings (+-20 tokens, firing token marked [[ ]]); Luna is told NOTHING about any concept.
It groups the snippets by what the neuron responds to (as few groups as honestly fit; leftovers in 'misc'), labels
each group, and rates every pair of groups: 0 same idea, 1 closely related, 2 loosely related, 3 unrelated.
Validation: every snippet in exactly one group or misc, ratings for every pair; otherwise recorded as invalid.
The (doc, position) of every snippet is saved so stage C can embed the same contexts in Qwen's own space.
Usage: python src/meaning_judge.py  -> results/meaning_judge.json (+ .log)
"""
import json
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

import oai
from config import DEFAULT_MODEL, model_path
from families import family_map

ROOT = Path(__file__).resolve().parents[1]
N_SNIP, HALF, MIN_FIRINGS = 30, 20, 30
PLAN = {(0, 16): 50, (16, 24): 20, (24, 30): 60, (30, 36): 70}
SYSTEM = ("A neuron inside a language model fired on the token marked [[like this]] in each numbered snippet. Group the "
          "snippets by what the neuron seems to respond to - the meaning, topic or idea around the marked token. Use as "
          "few groups as honestly fit; put snippets that fit no group in 'misc'. Every snippet goes in exactly one group "
          "or in misc. Then rate how related each PAIR of groups is: 0 = the same idea, 1 = closely related, "
          "2 = loosely related, 3 = unrelated. Reply as JSON: "
          '{"groups": [{"label": "<few words>", "members": [snippet numbers]}], "misc": [snippet numbers], '
          '"relatedness": [{"a": <group index from 0>, "b": <group index>, "rating": 0-3}]}')


def main():
    rows = json.loads((ROOT / "results/corpus_profile_units.json").read_text(encoding="utf-8"))
    z = np.load(ROOT / "results/corpus_profile.npz"); unit, doc, pos = z["unit"], z["doc"], z["pos"].astype(np.int64)
    order = np.argsort(unit, kind="stable"); unit, doc, pos = unit[order], doc[order], pos[order]
    b = np.searchsorted(unit, np.arange(len(rows) + 1)); nf = np.diff(b)
    d = np.load(ROOT / f"data/atlas_{DEFAULT_MODEL.split('/')[-1]}_docs.npz")
    toks, lens = d["tokens"], d["lengths"]; starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    tok = AutoTokenizer.from_pretrained(model_path(DEFAULT_MODEL)); fam = family_map()
    rng = random.Random(31)
    con = [i for i, r in enumerate(rows) if not r["control"] and nf[i] >= MIN_FIRINGS]
    picks = []
    for (lo, hi), k in PLAN.items():
        band = [i for i in con if lo <= rows[i]["layer"] < hi]
        picks += rng.sample(band, min(k, len(band)))
    ctl = [i for i, r in enumerate(rows) if r["control"] and nf[i] >= MIN_FIRINGS]
    ctl = rng.sample(ctl, min(60, len(ctl)))
    fam_of = lambda i: {fam.get(c, "?") for c in rows[i]["concepts"]}
    spl = []
    while len(spl) < 60:
        x, y = rng.sample(con, 2)
        if not (fam_of(x) & fam_of(y)) and not (set(rows[x]["concepts"]) & set(rows[y]["concepts"])):
            spl.append((x, y))
    jobs = [("concept", (i,)) for i in picks] + [("control", (i,)) for i in ctl] + [("spliced", p) for p in spl]
    rng.shuffle(jobs)

    def snippet(k):
        g = starts[doc[k]] + pos[k]; lo = max(starts[doc[k]], g - HALF); hi = min(starts[doc[k]] + lens[doc[k]], g + HALF + 1)
        return (tok.decode(toks[lo:g]) + "[[" + tok.decode(toks[g:g + 1]) + "]]" + tok.decode(toks[g + 1:hi])).replace("\n", " ")

    out_p = ROOT / "results/meaning_judge.json"
    out = json.loads(out_p.read_text(encoding="utf-8")) if out_p.exists() else {"jobs": {}}
    lock = threading.Lock(); log = open(ROOT / "results/meaning_judge.log", "a", encoding="utf-8")

    def one(job):
        kind, us = job
        key = kind + "|" + "+".join(rows[i]["neuron"] for i in us)
        if key in out["jobs"]:
            return
        jr = random.Random(f"31|{key}")
        if kind == "spliced":
            ks = [(0, k) for k in jr.sample(range(b[us[0]], b[us[0] + 1]), N_SNIP // 2)] + \
                 [(1, k) for k in jr.sample(range(b[us[1]], b[us[1] + 1]), N_SNIP // 2)]
            jr.shuffle(ks)
        else:
            ks = [(0, k) for k in jr.sample(range(b[us[0]], b[us[0] + 1]), N_SNIP)]
        snips = [snippet(k) for _, k in ks]
        prompt = "Snippets:\n" + "\n".join(f"{j + 1}. {s}" for j, s in enumerate(snips))
        try:
            r = oai.chat_json(prompt, SYSTEM)
            groups = r.get("groups", []); misc = [int(x) for x in r.get("misc", [])]
            members = [int(x) for g in groups for x in g.get("members", [])] + misc
            pairs = {(int(p["a"]), int(p["b"])) for p in r.get("relatedness", [])}
            need = {(i, j) for i, j in combinations(range(len(groups)), 2)}
            ok = sorted(members) == list(range(1, N_SNIP + 1)) and all((i, j) in pairs or (j, i) in pairs for i, j in need)
        except Exception as e:                                           # recorded, never dropped silently
            r, ok = {"error": repr(e)}, False
        with lock:
            out["jobs"][key] = {"kind": kind, "units": [rows[i]["neuron"] for i in us], "layer": rows[us[0]]["layer"],
                                "concepts": [rows[i]["concepts"] for i in us], "valid": ok, "result": r,
                                "source": [s for s, _ in ks], "doc_pos": [[int(doc[k]), int(pos[k])] for _, k in ks]}
            out["usage"] = dict(oai.usage)
            line = f"{key[:60]:<60} {'ok' if ok else 'INVALID'} groups={len(r.get('groups', [])) if ok else '-'}"
            print(line, flush=True); log.write(line + "\n"); log.flush()
            if len(out["jobs"]) % 20 == 0:
                out_p.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    with ThreadPoolExecutor(6) as ex:
        list(ex.map(one, jobs))
    out_p.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    u = out.get("usage", {})
    print(f"done: {len(out['jobs'])} jobs; Luna tokens in {u.get('input_tokens', 0):,} out {u.get('output_tokens', 0):,} "
          f"(~${u.get('input_tokens', 0) * 0.1e-6 + u.get('output_tokens', 0) * 0.5e-6:.2f})", flush=True)


if __name__ == "__main__":
    main()

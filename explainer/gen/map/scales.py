"""Two-ended neurons: one scale or two jobs? (method and pass rule: SCALES.md)

    python -I -X utf8 explainer/gen/map/scales.py [--limit N] [--workers 32] [--budget 3]

Per neuron used both ways in run 1 (drop >= 0.5 for a concept firing positively and for one firing negatively):
Luna describes each end and classifies the relation from the pushed words alone (ends shuffled); Jev checks blind that
one concept from each end is assigned to the right end, and gives a second opinion on the relation. Controls: shuffled
end pairs from different neurons. Results: D:/already-clean-data/scales/ (cache + summary.json).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "site" / "map" / "data"
OUT = Path(os.environ.get("MAP_SCALES", "D:/already-clean-data/scales"))
sys.path.insert(0, os.environ.get("MAP_API_SRC", "C:/dev/already-clean/src"))
import jev  # noqa: E402
import oai  # noqa: E402
from cache import Cache  # noqa: E402

MIN_DROP = 0.5
PRICE_IN, PRICE_OUT = 0.10e-6, 0.50e-6
SYSTEM = (
    "You study one neuron in a language model. When it fires one way it pushes the model towards the tokens of end A; "
    "firing the other way pushes towards the tokens of end B (the same arrow, reversed). From the two token lists alone, "
    "describe what each end is about in a few plain words (or 'unclear'), and say how the two ends relate: "
    "'opposites' = two ends of one scale or contrast (e.g. south / north, hot / cold, buy / sell); "
    "'related' = different but clearly connected; 'unrelated' = no connection you can name; 'unclear' = you can't tell. "
    "Tokens may be in several languages or be word pieces. Be conservative. "
    'Reply as JSON: {"a": "<few words>", "b": "<few words>", "relation": "opposites"|"related"|"unrelated"|"unclear", '
    '"confidence": "high"|"medium"|"low"}')
REL_OPTS = {"opposites": "two ends of one scale or contrast", "related": "different but clearly connected",
            "unrelated": "no clear connection"}


def rng_for(nid, salt=""):
    return random.Random(int(hashlib.sha1((nid + salt).encode()).hexdigest()[:12], 16))


def load_uses():
    cj = json.loads((DATA / "concepts.json").read_text(encoding="utf-8"))
    I = {k: i for i, k in enumerate(cj["cols"])}
    uses = defaultdict(lambda: {1: [], -1: []})
    for r in cj["rows"]:
        if r[I["status"]] != "mapped" or "0" not in r[I["sources"]]:
            continue
        d = json.loads((DATA / "c" / f"{r[I['slug']]}.json").read_text(encoding="utf-8"))
        m = next((m for m in d["maps"] if m["source"] == "run1" and m.get("top")), None)
        for t in (m or {}).get("top", []):
            if t[3] is not None and (t[1] or 0) >= MIN_DROP:
                uses[t[0]][1 if t[3] >= 0 else -1].append(r[I["name"]])
    return {n: u for n, u in uses.items() if u[1] and u[-1]}


def push_words():
    out = {}
    for L in range(36):
        for i, v in json.loads((DATA / "push" / f"L{L}.json").read_text(encoding="utf-8")).items():
            out[f"L{L}:{i}"] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--budget", type=float, default=3.0)
    args = ap.parse_args()
    both, push = load_uses(), push_words()
    todo = sorted((n for n in both if n in push), key=lambda n: hashlib.sha1(n.encode()).hexdigest())
    if args.limit:
        todo = todo[:args.limit]
    OUT.mkdir(parents=True, exist_ok=True)
    cache = Cache(OUT / "scales.cache.jsonl")
    print(f"{len(todo)} two-way neurons (of {len(both)})")
    lock = threading.Lock(); spend = {"usd": 0.0}; t0 = time.time()

    def luna(key, wa, wb):
        if key in cache or spend["usd"] > args.budget:
            return
        try:
            r = oai.chat_json(f"End A tokens: {' | '.join(wa)}\nEnd B tokens: {' | '.join(wb)}", system=SYSTEM)
        except Exception as e:
            print("luna fail", key, repr(e)[:100]); return
        with lock:
            spend["usd"] = oai.usage["input_tokens"] * PRICE_IN + oai.usage["output_tokens"] * PRICE_OUT
        cache.put(key, {k: r.get(k) for k in ("a", "b", "relation", "confidence")})

    def one(n):
        flip = rng_for(n).random() < 0.5                         # which real end is shown as 'A' (sign is a convention)
        ends = (push[n][1], push[n][0]) if flip else (push[n][0], push[n][1])
        luna("luna|" + n, ends[0][:10], ends[1][:10])
        r = cache.get("luna|" + n)
        if not r:
            return
        # usage check: one concept from each real end; which description is 'A' follows the flip above
        sideA = -1 if flip else 1
        for tag, side in (("A", sideA), ("B", -sideA)):
            key = f"use{tag}|{n}"
            if key in cache:
                continue
            c = rng_for(n, tag).choice(sorted(both[n][side]))
            opts = [("A", r.get("a") or "unclear"), ("B", r.get("b") or "unclear")]
            rng_for(n, "order" + tag).shuffle(opts)
            try:
                p = jev.choice(f"A neuron pushes the model towards one of two things. End 1: {opts[0][1]}. End 2: {opts[1][1]}.",
                               f"Which end is most relevant to helping the model name: {c}?", {"1": opts[0][1], "2": opts[1][1]})
            except Exception as e:
                print("jev fail", key, repr(e)[:100]); continue
            correct = "1" if opts[0][0] == tag else "2"
            cache.put(key, {"concept": c, "p_correct": round(p.get(correct, 0.0), 4)})
        key = "rel|" + n
        if key not in cache:
            try:
                p = jev.choice(f"One end of a neuron is about: {r.get('a')}. The other end is about: {r.get('b')}.",
                               "How do the two ends relate?", REL_OPTS)
                cache.put(key, {k: round(v, 4) for k, v in p.items()})
            except Exception as e:
                print("jev fail", key, repr(e)[:100])

    with ThreadPoolExecutor(args.workers) as ex:
        for j, _ in enumerate(ex.map(one, todo)):
            if j % 200 == 0:
                print(f"{j}/{len(todo)}  ${spend['usd']:.2f}  {time.time() - t0:.0f}s", flush=True)

    # controls: end A of one neuron with end B of another
    pool = list(todo)

    def ctrl(j):
        rr = random.Random(1000 + j)
        n1, n2 = rr.sample(pool, 2)
        key = f"ctrl|{j}"
        luna(key, push[n1][0][:10], push[n2][1][:10])
        r = cache.get(key)
        if r and "ctrlrel|%d" % j not in cache:
            try:
                p = jev.choice(f"One end of a neuron is about: {r.get('a')}. The other end is about: {r.get('b')}.",
                               "How do the two ends relate?", REL_OPTS)
                cache.put("ctrlrel|%d" % j, {k: round(v, 4) for k, v in p.items()})
            except Exception as e:
                print("jev fail ctrl", j, repr(e)[:100])
    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(ctrl, range(200 if not args.limit else 20)))

    # summary and the rule from SCALES.md
    res = []
    for n in todo:
        r, ua, ub, rel = (cache.get(k + n) for k in ("luna|", "useA|", "useB|", "rel|"))
        if not (r and ua and ub and rel):
            continue
        jtop = max(rel, key=rel.get)
        use_ok = ua["p_correct"] >= 0.8 and ub["p_correct"] >= 0.8
        verdict = ("scale" if r["relation"] == "opposites" and r["confidence"] == "high" and jtop == "opposites" and rel[jtop] >= 0.5 and use_ok
                   else "two jobs" if r["relation"] == "unrelated" and r["confidence"] == "high" and jtop == "unrelated" and rel[jtop] >= 0.5 and use_ok
                   else None)
        res.append({"n": n, "luna": r["relation"], "conf": r["confidence"], "jev": jtop, "use_ok": use_ok, "verdict": verdict,
                    "a": r["a"], "b": r["b"]})
    ctl = [(cache.get(f"ctrl|{j}"), cache.get(f"ctrlrel|{j}")) for j in range(200)]
    ctl = [(a, b) for a, b in ctl if a and b]
    summ = {"two_way_neurons": len(both), "classified": len(res),
            "luna_relation": dict(Counter(x["luna"] for x in res)), "jev_relation": dict(Counter(x["jev"] for x in res)),
            "usage_check_pass": sum(x["use_ok"] for x in res),
            "scales": sum(x["verdict"] == "scale" for x in res), "two_jobs": sum(x["verdict"] == "two jobs" for x in res),
            "control_pairs": len(ctl),
            "control_luna_opposites": sum(a["relation"] == "opposites" for a, _ in ctl),
            "control_luna_opposites_high": sum(a["relation"] == "opposites" and a["confidence"] == "high" for a, _ in ctl),
            "control_jev_opposites": sum(max(b, key=b.get) == "opposites" for _, b in ctl),
            "luna_usd": round(spend["usd"], 3), "jev_usd": round(jev.cost(), 3)}
    (OUT / "summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    (OUT / "verdicts.json").write_text(json.dumps(res, ensure_ascii=False, indent=0), encoding="utf-8")
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()

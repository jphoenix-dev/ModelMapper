"""Convenience job names for mapped neurons, checked blind on a held-out concept (method: JOB_NAMES.md).

    python -I -X utf8 explainer/gen/map/job_names.py [--limit N] [--workers 12] [--budget 5]

Reads the built map data (explainer/site/map/data: n/L*.json, concepts.json), asks GPT-6 Luna for a name from the
evidence minus one held-out concept, then asks Jev which of five concepts a neuron with that name helps with (the
held-out one plus four that don't use it), and the same with another neuron's name as a control. Results go to
D:/already-clean-data/job_names/ (resumable cache + summary.json). Uses the API clients and keys of the main
checkout (MAP_API_SRC, default C:/dev/already-clean/src); keys are never printed.
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
OUT = Path(os.environ.get("MAP_JOB_NAMES", "D:/already-clean-data/job_names"))
sys.path.insert(0, os.environ.get("MAP_API_SRC", "C:/dev/already-clean/src"))
import jev  # noqa: E402
import oai  # noqa: E402
from cache import Cache  # noqa: E402

PRICE_IN, PRICE_OUT = 0.10e-6, 0.50e-6          # Luna (as map_loop.py)
SYSTEM = (
    "You give short, plain-English names to neurons in a language model, for a map that lay readers browse. "
    "A name is a convenience label for what the evidence has in common, not a claim about the neuron's true meaning. "
    "Be conservative: if the evidence has no clear common thread, give no name. Do not name a single concept unless "
    "the evidence is clearly about that concept. Use everyday words; no jargon. "
    'Reply as JSON: {"name": "<2-6 words, or empty>", "confidence": "high"|"medium"|"low", '
    '"basis": "<one short sentence: which evidence the name rests on>"}')


def seeded(nid, salt=""):
    return random.Random(int(hashlib.sha1((nid + salt).encode()).hexdigest()[:12], 16))


def odds(d):
    f = 2.718281828 ** max(0.0, d or 0.0)
    return f"{f:.0f}x" if f >= 10 else f"{f:.1f}x"


def evidence_text(nid, v, held_name, cname):
    L = int(nid[1:].split(":")[0])
    lines = [f"Neuron {nid}: layer {L} of 36 ({'early: words read from its weights are unreliable this early' if L < 20 else 'late'})."]
    if v.get("roles"):
        lines.append("Role in the maps: " + ", ".join(f"{r} x{k}" for r, k in v["roles"].items()) +
                     " ('say' = the answer is among the words it pushes most; 'attribute' = it pushes the answer up).")
    words = [w for w in (v.get("jobs") or []) if w.strip()][:15]
    if words:
        lines.append("Words it pushes the model towards (from its weights): " + " | ".join(words))
    seen, cs = set(), []
    for ci, src, rank, drop in sorted(v["uses"], key=lambda u: -(u[3] or 0)):
        name = cname[ci]
        if name == held_name or name in seen:
            continue
        seen.add(name); cs.append(f"{name} (answer {odds(drop)} less likely without it)")
    lines.append("Concepts that need it: " + "; ".join(cs[:12]) + (f"; and {len(cs) - 12} more" if len(cs) > 12 else ""))
    for f in v.get("findings", []):
        t = f.get("title", "")
        if held_name.lower() in t.lower():
            continue
        if f["kind"] == "label":
            m = f.get("metric") or {}
            lines.append(f"Blind label of where it fires in web text ({'positive' if m.get('pole') == 'pos' else 'negative'} "
                         f"direction, {'passed' if m.get('clean') else 'did not pass'} the one-job check): {t}")
        elif f["kind"] != "web firings":
            lines.append(f"Finding ({f['kind']}): {t}")
    return "\n".join(lines[:30])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--budget", type=float, default=5.0)
    args = ap.parse_args()
    cj = json.loads((DATA / "concepts.json").read_text(encoding="utf-8"))
    I = {k: i for i, k in enumerate(cj["cols"])}
    rows = cj["rows"]; cname = [r[I["name"]] for r in rows]
    mapped = [j for j, r in enumerate(rows) if r[I["status"]] == "mapped"]
    by_section = defaultdict(list)
    for j in mapped:
        by_section[rows[j][I["section"]]].append(j)

    todo = []
    for L in range(36):
        for idx, v in json.loads((DATA / "n" / f"L{L}.json").read_text(encoding="utf-8")).items():
            names = {cname[u[0]] for u in v.get("uses", [])}
            if len(names) >= 2:
                todo.append((f"L{L}:{idx}", v))
    todo.sort(key=lambda t: hashlib.sha1(t[0].encode()).hexdigest())          # a random but fixed order
    if args.limit:
        todo = todo[:args.limit]
    OUT.mkdir(parents=True, exist_ok=True)
    cache = Cache(OUT / "names.cache.jsonl")
    print(f"{len(todo)} neurons to name (>= 2 concepts); {sum(1 for t in todo if 'luna|' + t[0] in cache)} cached")

    spend = {"luna": 0.0}; lock = threading.Lock(); t0 = time.time()

    def name_one(item):
        nid, v = item
        key = "luna|" + nid
        if key in cache:
            return
        if spend["luna"] > args.budget:
            return
        uses = sorted({u[0] for u in v["uses"]})
        held = seeded(nid).choice(uses)
        prompt = evidence_text(nid, v, cname[held], cname)
        u0 = dict(oai.usage)
        try:
            r = oai.chat_json(prompt, system=SYSTEM)
        except Exception as e:                                          # keep going; it is retried on the next run
            print("luna fail", nid, repr(e)[:120]); return
        with lock:
            spend["luna"] = oai.usage["input_tokens"] * PRICE_IN + oai.usage["output_tokens"] * PRICE_OUT
        cache.put(key, {"name": (r.get("name") or "").strip(), "confidence": r.get("confidence"),
                        "basis": r.get("basis"), "held": held, "held_name": cname[held],   # row index + name: rows reorder
                        "prompt_chars": len(prompt)})

    with ThreadPoolExecutor(args.workers) as ex:
        for j, _ in enumerate(ex.map(name_one, todo)):
            if j % 250 == 0:
                print(f"luna {j}/{len(todo)}  ${spend['luna']:.2f}  {time.time() - t0:.0f}s", flush=True)

    # ---- blind check with Jev, plus a control with another neuron's name
    named = [(nid, cache.get("luna|" + nid)) for nid, _ in todo if (cache.get("luna|" + nid) or {}).get("name")]
    users = {nid: {u[0] for u in v["uses"]} for nid, v in todo}
    pool_names = [n["name"] for _, n in named]

    def check(nid, name, held, salt):
        key = f"jev{salt}|{nid}"
        if key in cache:
            return
        rng = seeded(nid, "distract")
        sec = rows[held][I["section"]]
        pool = [j for j in (by_section.get(sec) if sec else None) or mapped if j not in users[nid]]
        if len(pool) < 4:
            pool = [j for j in mapped if j not in users[nid]]
        opts = [held] + rng.sample(pool, 4)
        rng.shuffle(opts)
        letters = "ABCDE"
        try:
            p = jev.choice(f'A neuron in a language model has been given this short job name: "{name}".',
                           "Which one of these concepts is this neuron most likely to help the model name?",
                           {letters[j]: cname[c] for j, c in enumerate(opts)})
        except Exception as e:
            print("jev fail", nid, repr(e)[:120]); return
        cache.put(key, {"p_held": round(p.get(letters[opts.index(held)], 0.0), 4), "options": [cname[c] for c in opts],
                        "name": name})

    def check_pair(item):
        nid, n = item
        check(nid, n["name"], n["held"], "")
        other = seeded(nid, "control").choice(pool_names)
        check(nid, other, n["held"], "_ctrl")

    with ThreadPoolExecutor(args.workers) as ex:
        for j, _ in enumerate(ex.map(check_pair, named)):
            if j % 500 == 0:
                print(f"jev {j}/{len(named)}  {time.time() - t0:.0f}s", flush=True)

    # ---- summary
    def band(nid):
        L = int(nid[1:].split(":")[0]); return "early (0-19)" if L < 20 else "late (20-35)"
    res = []
    for nid, n in named:
        a, c = cache.get("jev|" + nid), cache.get("jev_ctrl|" + nid)
        if a and c:
            res.append({"nid": nid, "conf": n["confidence"], "p": a["p_held"], "pc": c["p_held"], "band": band(nid)})
    summ = {"neurons": len(todo), "named": len(named), "checked": len(res),
            "confidence": dict(Counter(n["confidence"] for _, n in named)),
            "luna_usd": round(spend["luna"], 3), "jev_usd": round(jev.cost(), 3) if hasattr(jev, "cost") else None,
            "rule": "confidence == high and p_held >= 0.5"}
    for grp, sel in [("all", res), ("high", [r for r in res if r["conf"] == "high"])] + \
                    [(b, [r for r in res if r["band"] == b and r["conf"] == "high"]) for b in ("early (0-19)", "late (20-35)")]:
        if sel:
            summ[grp] = {"n": len(sel), "pass": round(sum(r["p"] >= 0.5 for r in sel) / len(sel), 3),
                         "control_pass": round(sum(r["pc"] >= 0.5 for r in sel) / len(sel), 3),
                         "mean_p": round(sum(r["p"] for r in sel) / len(sel), 3),
                         "mean_p_control": round(sum(r["pc"] for r in sel) / len(sel), 3)}
    summ["shown"] = sum(1 for r in res if r["conf"] == "high" and r["p"] >= 0.5)
    (OUT / "summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
